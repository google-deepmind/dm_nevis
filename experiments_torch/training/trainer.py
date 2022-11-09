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
from typing import Any, Callable, Mapping, Optional, Tuple

from absl import logging
from dm_nevis.benchmarker.datasets import tasks
from experiments_torch.environment import pickle_checkpointer
from experiments_torch.training import models

from torch import nn
from torch import optim

LOG_INTERVAL = 10
DEFAULT_MOVING_AVERAGE_ALPHA = 0.8

UpdateFn = Any
LoadParamsFn = Callable[[nn.Module], Tuple[nn.ParameterList, nn.ParameterList]]


@dataclasses.dataclass
class TrainState:
  model: nn.Module
  optimizer: Optional[optim.Optimizer]


def init_train_state(
    model: models.Model,
    optimizer_ctor: Callable[..., optim.Optimizer],
    optimizer_kwargs: Mapping[str, Any],
    load_params_fn: Optional[LoadParamsFn] = None,
    *,
    log_model_summary: bool = True,
) -> TrainState:
  """Initialize model parameter and state.

  Args:
    model: the model.
    optimizer_ctor: the optimizer not instantiated.
    optimizer_kwargs: the optimizer arguments.
    load_params_fn: Optional function to load pre-trained parameters and/or to
      freeze a subset of the parameters. The function takes the models randomly
      initialized parameters, and returns two list of parameters.
      ([trainable_params, ...], [frozen_params, ...]) tuple.
    log_model_summary: When True, logs information about the initialized
      parameters and state.

  Returns:
    A TrainState structure.
  """
  if load_params_fn:
    trainable_params, frozen_params = load_params_fn(model)
    optimizer = optimizer_ctor([{
        "params": trainable_params,
        "lr": 0.1
    }, {
        "params": frozen_params,
        "lr": 0.
    }], **optimizer_kwargs)
  else:
    trainable_params = model.parameters()
    optimizer = optimizer_ctor(trainable_params, lr=0.1, **optimizer_kwargs)  # pytype: disable=wrong-keyword-args

  if log_model_summary:
    logging.info("Model parameters: \n%s", models.param_summary(model))
    logging.info("Model size: \n%s", models.size_summary(model))

  return TrainState(model=model, optimizer=optimizer)


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
