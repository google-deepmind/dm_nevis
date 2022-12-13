# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A module implementing primitives for training by gradient descent."""

import dataclasses
import functools
import time
from typing import Any, Callable, Iterator, Mapping, Optional, Tuple

from absl import logging
import chex
from dm_nevis.benchmarker.datasets import datasets
from dm_nevis.benchmarker.datasets import tasks
from dm_nevis.benchmarker.environment import datawriter_interface
from experiments_jax.environment import pickle_checkpointer
from experiments_jax.training import models
from experiments_jax.training import resources
import haiku as hk
import jax
import optax

LOG_INTERVAL = 10
DEFAULT_MOVING_AVERAGE_ALPHA = 0.8

UpdateFn = Any
LoadParamsFn = Callable[[hk.Params, hk.State], Tuple[hk.Params, hk.Params,
                                                     hk.State]]


@chex.dataclass
class TrainState:
  rng: chex.PRNGKey
  trainable_params: hk.Params
  frozen_params: hk.Params
  state: hk.State
  opt_state: optax.OptState


def init_train_state(
    rng: chex.PRNGKey,
    model: models.Model,
    opt: optax.GradientTransformation,
    load_params_fn: Optional[LoadParamsFn] = None,
    *,
    log_model_summary: bool = True,
) -> TrainState:
  """Initializes model parameter and state.

  Args:
    rng: random seed.
    model: the model.
    opt: the optimizer.
    load_params_fn: Optional function to load pre-trained parameters and/or to
      freeze a subset of the parameters. The function takes the models randomly
      initialized parameters and state structures, and returns a
      (trainable_params, frozen_params, state) tuple.
    log_model_summary: When True, logs information about the initialized
      parameters and state.

  Returns:
    A TrainState structure.
  """
  init_rng, rng = jax.random.split(rng)
  params, state = model.init(init_rng)
  if load_params_fn:
    trainable_params, frozen_params, state = load_params_fn(params, state)
  else:
    trainable_params, frozen_params = params, {}

  opt_state = opt.init(trainable_params)

  if log_model_summary:
    logging.info("Model parameters: \n%s",
                 models.param_summary(trainable_params))
    logging.info("Frozen parameters: \n%s", models.param_summary(frozen_params))
    logging.info("Model state: \n%s", models.param_summary(state))
    logging.info("Model size (train-params/frozen-params/state): %s / %s / %s",
                 models.size_summary(trainable_params),
                 models.size_summary(frozen_params), models.size_summary(state))

  return TrainState(
      rng=rng,
      trainable_params=trainable_params,
      frozen_params=frozen_params,
      state=state,
      opt_state=opt_state)


def build_update_fn(task_key: tasks.TaskKey, model: models.Model,
                    opt: optax.GradientTransformation) -> UpdateFn:
  """Builds an update function for updating train state using an optimizer."""

  def update_fn(batch: datasets.MiniBatch,
                train_state: TrainState,
                lr_scale: float = 1.) -> Tuple[TrainState, Mapping[str, Any]]:

    @functools.partial(jax.value_and_grad, has_aux=True)
    def grad_fn(trainable_params, frozen_params, state, rng, batch):
      params = hk.data_structures.merge(trainable_params, frozen_params)
      fn = model.loss_and_metrics[task_key]
      label = None
      if task_key.kind == tasks.TaskKind.MULTI_LABEL_CLASSIFICATION:
        label = batch.multi_label_one_hot
      else:
        label = batch.label
      (loss, metrics), state = fn(params, state, rng, batch.image, label, True)
      return loss, (metrics, state)

    step_rng, next_rng = jax.random.split(train_state.rng)
    ((loss, (metrics, state)),
     grad) = grad_fn(train_state.trainable_params, train_state.frozen_params,
                     train_state.state, step_rng, batch)
    updates, opt_state = opt.update(grad, train_state.opt_state,
                                    train_state.trainable_params)

    updates = jax.tree_map(lambda x: lr_scale * x, updates)
    trainable_params = optax.apply_updates(train_state.trainable_params,
                                           updates)
    metrics = {"loss": loss, **metrics}
    metrics = jax.tree_map(lambda x: x.mean(), metrics)

    train_state = dataclasses.replace(
        train_state,
        rng=next_rng,
        trainable_params=trainable_params,
        state=state,
        opt_state=opt_state)
    return train_state, metrics

  return update_fn


def restore_train_state(train_state_checkpoint_path: str) -> TrainState:
  """Load train state from checkpoint path if it has been saved to disk."""
  if train_state_checkpoint_path is None:
    return None
  checkpointer = pickle_checkpointer.PickleCheckpointer(
      train_state_checkpoint_path)
  train_state = checkpointer.restore()
  return train_state


def save_train_state(checkpoint_file_path: str, task_key: tasks.TaskKey,
                     train_state: TrainState):
  logging.info("Saving train state for train task %s to %s", task_key.name,
               checkpoint_file_path)
  checkpointer = pickle_checkpointer.PickleCheckpointer(checkpoint_file_path)
  checkpointer.write(train_state)


def fit_params(
    train_state: TrainState,
    train_iter: Iterator[datasets.MiniBatch],
    update_fn: UpdateFn,
    num_steps: int,
    on_complete_step: Callable[[int, TrainState], None],
    metrics_writer: datawriter_interface.DataWriter,
    initial_global_step: Optional[int] = None) -> Tuple[TrainState, float]:
  """Runs gradient descent+optimizer step for the given number of steps."""

  global_step = initial_global_step or 0
  estimated_flops = None
  step_counter = StepCountEstimator()

  while global_step < num_steps:
    t = time.monotonic()

    batch = next(train_iter)
    logging.log_every_n(logging.INFO, "Step: %d/%d, Batch %s", LOG_INTERVAL,
                        global_step + 1, num_steps, batch)

    train_state, metrics = update_fn(batch, train_state)

    global_step += 1
    on_complete_step(global_step, train_state)

    metrics = {
        "global_step": global_step,
        "steps_per_second": step_counter.estimated_steps_per_second(),
        **metrics
    }
    metrics_writer.write(metrics)

    if estimated_flops is None:
      estimated_flops = resources.estimate_flops(update_fn, batch, train_state)

    step_counter.add_measurement(time.monotonic() - t)

  return train_state, (estimated_flops or 0.0) * num_steps


class StepCountEstimator:
  """Estimates how many steps per second are achieved during trainnig."""

  def __init__(self, alpha: float = DEFAULT_MOVING_AVERAGE_ALPHA) -> None:
    self._ema = None
    self._alpha = alpha

  def add_measurement(self, elapsed_seconds: float):
    if self._ema is None:
      self._ema = elapsed_seconds
    else:
      self._ema = self._alpha * self._ema + (1 - self._alpha) * elapsed_seconds

  def estimated_steps_per_second(self) -> float:
    if not self._ema:
      return float("nan")
    return 1 / self._ema
