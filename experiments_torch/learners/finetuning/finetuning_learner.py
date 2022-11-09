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
"""A finetuning learner.

This learner supports a number of strategies for initializing the train state
for each sequential training task. One such strategy is "independent". In This
case, each model is trained independently.
"""
# pytype: disable=missing-parameter
# pytype: disable=not-callable
import copy
import dataclasses
import enum
import functools
import os
import time
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from absl import logging
from dm_nevis.benchmarker.datasets import datasets
from dm_nevis.benchmarker.datasets import streams
from dm_nevis.benchmarker.datasets import tasks
from dm_nevis.benchmarker.environment import datawriter_interface
from dm_nevis.benchmarker.learners import learner_interface
from experiments_torch import experiment
from experiments_torch.training import dataloaders
from experiments_torch.training import learning_rate_schedules
from experiments_torch.training import models
from experiments_torch.training import resources
from experiments_torch.training import trainer
import ml_collections
import numpy as np
import torch
from torch.optim import lr_scheduler


CHECKPOINT_INTERVAL = 10_000
LOG_INTERVAL_SECONDS = 5
MAX_LR_DECAY_STEPS = 4
FINETUNING_DATAFRAME_NAME = "finetuning"
DUMMY_TASK_NAME_RANDOM_PARAMS = "randomly_initialized_params"
SUPPORTED_TASK_KINDS = frozenset([
    tasks.TaskKind.CLASSIFICATION,
    tasks.TaskKind.MULTI_LABEL_CLASSIFICATION,
])


class FinetuningStrategy(enum.Enum):
  INDEPENDENT = 0  # Randomly initialize the state for each model.
  PREVIOUS = 1  # Always initialize from train state from previous task.


def learner_builder(
    dataset_lookup_builder: experiment.DatasetLookupBuilderFn,
    learner_config: ml_collections.ConfigDict
) -> Tuple[experiment.LearnerBuilderFn, experiment.ProgramStopper]:
  """Prepares the learner to run on launchpad."""

  def stopper():
    logging.info("Closing program")

  def build_learner():
    dataset_lookup, task_keys = dataset_lookup_builder()

    return build_finetuning_learner(learner_config, dataset_lookup, task_keys)

  return build_learner, stopper


@dataclasses.dataclass(frozen=True)
class TrainingContext:
  train_task_index: int
  config: ml_collections.ConfigDict
  event: streams.TrainingEvent
  initial_train_state: Optional[trainer.TrainState] = dataclasses.field(
      repr=False)


@dataclasses.dataclass(frozen=True)
class FinetuningLearnerState:
  """A dataclass to hold the state of the learner.

  Attributes:
    seen_train_events: The (ordered) sequence of train events encountered by
      this learner.
    train_states: The sequence of tasks and the achieved final train states or
      the checkpoint paths to the train states.
  """
  seen_train_events: List[streams.TrainingEvent]
  train_states: List[Tuple[tasks.TaskKey, str]]


def build_finetuning_learner(
    config: ml_collections.ConfigDict,
    dataset_lookup: experiment.DatasetLookupFn,
    task_keys: Sequence[tasks.TaskKey],
) -> learner_interface.Learner:
  """Builds the learner.

  Args:
    config: The configuration to use for this learner.
    dataset_lookup: A function used to construct train and predict datasets.
    task_keys: The tasks that the returned learner should support.

  Returns:
    A learner satisfying the learner_interface.Learner interface.
  """
  _verify_all_tasks_supported(task_keys)

  finetuning_metrics = config.get_metrics_writer("finetuning")
  cost_function = _cost_function_builder(dataset_lookup, task_keys)

  return learner_interface.Learner(
      init=functools.partial(
          init,
          config=config,
      ),
      train=functools.partial(
          train,
          dataset_lookup=dataset_lookup,
          config=config,
          cost_function=cost_function,
          finetuning_metrics=finetuning_metrics,
      ),
      predict=functools.partial(
          predict,
          config=config,
          dataset_lookup=dataset_lookup,
      ),
  )


def init(
    *,
    config: ml_collections.ConfigDict,
) -> learner_interface.LearnerState:
  """A function to initialize the train state for the learner.

  Args:
    config: The learner configuration.

  Returns:
    The initial learner state, before the learner has seen any training data.
  """
  del config

  return FinetuningLearnerState(
      seen_train_events=[],
      train_states=[],
  )


def train(
    event: streams.TrainingEvent,
    state: learner_interface.LearnerState,
    write_checkpoint: learner_interface.CheckpointFn,
    checkpoint_to_resume: learner_interface.Checkpoint = None,
    *,
    dataset_lookup: experiment.DatasetLookupFn,
    config: ml_collections.ConfigDict,
    cost_function: Any,
    finetuning_metrics: datawriter_interface.DataWriter,
) -> Tuple[learner_interface.LearnerState, learner_interface.ResourceUsage]:
  """Trains the learner given the given dataset.

  Args:
    event: The training event that the learner should read training data from.
    state: The learner's state before training began.
    write_checkpoint: A function to write intermediate checkpoints during this
      training event.
    checkpoint_to_resume: If this training event was previously interrupted,
      then this training event may be initialized from a checkpoint that was
      previously written by the write_checkpoint function.
    dataset_lookup: A lookup function for fetching the dataset by key.
    config: The learner config.
    cost_function: The function optimizing the model.
    finetuning_metrics: A metrics writer for writing the selected state that was
      finetuned from.

  Returns:
    A new learner state, containing the knowledge accrued during training, along
    with the resources used during training.
  """
  del checkpoint_to_resume

  task_key = dataset_lookup(event.train_dataset_key).task_key
  initial_train_state = _get_train_state_for_finetuning(config, task_key, state,
                                                        finetuning_metrics)

  context = TrainingContext(
      train_task_index=len(state.seen_train_events),
      config=config,
      event=event,
      initial_train_state=initial_train_state,
  )
  _, new_train_state, resources_used = cost_function(
      context, write_checkpoint=write_checkpoint)

  train_state_checkpoint_path = os.path.join(
      config.train_states_checkpoint_path,
      f"train_task_index_{len(state.seen_train_events)}_{task_key.name}.pkl")
  trainer.save_train_state(train_state_checkpoint_path, task_key,
                           new_train_state)

  return (
      dataclasses.replace(
          state,
          train_states=[
              *state.train_states, (task_key, train_state_checkpoint_path)
          ],
          seen_train_events=[*state.seen_train_events, event],
      ),
      resources_used,
  )


def predict(
    event: streams.PredictionEvent,
    state: learner_interface.LearnerState,
    *,
    config: ml_collections.ConfigDict,
    dataset_lookup: experiment.DatasetLookupFn,
) -> Iterable[learner_interface.Predictions]:
  """Computes predictions for each example in the referenced dataset.

  Args:
    event: An event containing a dataset key to compute predictions for.
    state: The state of the learner, containing all knowledge accrued by the
      learner as it was exposed to previous training events.
    config: The config of the learner.
    dataset_lookup: A function to fetch datasets by key.

  Yields:
    Batches of predictions from the model, given the learner state, over the
    dataset loaded from the event.
  """
  dataset = dataset_lookup(event.dataset_key)
  task_key = dataset.task_key
  eval_augment_fn = functools.partial(config.augmentations.eval.ctor,
                                      **config.augmentations.eval.kwargs)
  batch_iter = dataloaders.build_prediction_iterator(dataset, eval_augment_fn,
                                                     config.batch_size)

  train_state = _get_latest_train_state_for_predictions(state, task_key)
  model = train_state.model
  model.to(config.device)

  completed = 0
  for batch in batch_iter():
    logging.log_every_n_seconds(logging.INFO, "Completed predictions: %d/%d",
                                10, completed, dataset.num_examples)
    completed += batch.image.shape[0]

    probabilities = model.predict(
        torch.tensor(batch.image).to(config.device), task_key)
    yield learner_interface.Predictions(batch=batch, output=probabilities)


@dataclasses.dataclass
class FitWithEarlyStoppingState:
  step: int
  train_state: trainer.TrainState
  best_age: int
  best_metric: np.number
  best_train_state: Optional[trainer.TrainState]
  validation_metric: str


def _cost_function_builder(dataset_lookup_fn: experiment.DatasetLookupFn,
                           task_keys: Sequence[tasks.TaskKey]) -> Any:
  """Constructs the cost function used in the hyper search."""

  def cost_function(train_context,
                    *,
                    write_checkpoint,
                    checkpoint_to_resume=None):

    config = train_context.config
    logging.info("Computing cost function with learner config: %s", config)

    # Data for work-unit
    train_dataset = dataset_lookup_fn(train_context.event.train_dataset_key)
    valid_dataset = dataset_lookup_fn(train_context.event.dev_dataset_key)

    task_key = train_dataset.task_key

    train_augment_fn = functools.partial(config.augmentations.train.ctor,
                                         **config.augmentations.train.kwargs)
    eval_augment_fn = functools.partial(config.augmentations.eval.ctor,
                                        **config.augmentations.eval.kwargs)

    train_iter_fn = dataloaders.build_train_iterator(train_dataset,
                                                     train_augment_fn,
                                                     config.batch_size)
    valid_iter_fn = dataloaders.build_prediction_iterator(
        valid_dataset, eval_augment_fn, config.batch_size)

    steps_per_epoch = train_dataset.num_examples // config.batch_size + 1

    initial_train_state = _initialize_train_from_context(
        train_context, config, task_keys, config.optimization.optimizer.ctor,
        config.optimization.optimizer.kwargs)
    optimizer = initial_train_state.optimizer

    # If learning rate schedule is provided, we use it.
    learning_rate_schedule = learning_rate_schedules.build_learning_rate_schedule(
        optimizer, config.optimization.learning_rate_schedule.name,
        config.optimization.learning_rate_schedule.init_learning_rate,
        steps_per_epoch, config.max_steps,
        config.optimization.learning_rate_schedule.kwargs)

    train_metric_writer = config.get_metrics_writer(
        "learner_train", index_of_training_event=train_context.train_task_index)
    eval_metric_writer = config.get_metrics_writer(
        "learner_eval", index_of_training_event=train_context.train_task_index)

    cost, _, train_state, flops_used = fit_with_early_stopping(
        initial_train_state=initial_train_state,
        learning_rate_scheduler=learning_rate_schedule,
        task_key=task_key,
        train_iter_fn=train_iter_fn,
        valid_iter_fn=valid_iter_fn,
        validation_metric=config.validation_metric,
        run_validation_every_n_steps=config.run_validation_every_n_steps,
        early_stopping_grace=config.early_stopping_grace,
        max_steps=config.max_steps,
        device=config.device,
        train_metrics_writer=train_metric_writer,
        validation_metrics_writer=eval_metric_writer,
        write_checkpoint=write_checkpoint,
        checkpoint_to_resume=checkpoint_to_resume)

    resources_used = learner_interface.ResourceUsage(
        floating_point_operations=flops_used)

    train_metric_writer.flush()
    train_metric_writer.close()

    eval_metric_writer.flush()
    eval_metric_writer.close()

    train_state.model = train_state.model.cpu()

    return cost, train_state, resources_used

  return cost_function


def _initialize_train_from_context(train_context, config, task_keys, opt_ctor,
                                   opt_kwargs) -> trainer.TrainState:
  """Initializes trainer state based on the context."""
  if train_context.initial_train_state is not None:
    logging.info("Initializing train state from a previous state")
    model = copy.deepcopy(train_context.initial_train_state.model)
    load_params_fun = None
  else:
    logging.info("Initializing a new train state")
    model = _build_model(config, task_keys)
    load_params_fun = config.load_params_fn
    if "load_params_fn_with_kwargs" in config:
      load_params_fun = functools.partial(
          config.load_params_fn_with_kwargs.fun,
          **config.load_params_fn_with_kwargs.kwargs)
  return trainer.init_train_state(model, opt_ctor, opt_kwargs, load_params_fun)


def _run_validation(
    model: models.Model,
    task_key: tasks.TaskKey,
    state: FitWithEarlyStoppingState,
    valid_data_iter: Iterator[datasets.MiniBatch],
    device: torch.device,
    additional_diagnostics: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, Any]]:
  """Runs validation and returns the cost and metrics."""

  start_time = time.monotonic()
  metrics = _validate_batches(model, task_key, valid_data_iter, device)
  elapsed = time.monotonic() - start_time

  metrics.update(
      step=state.step,
      validation_runtime_seconds=elapsed,
  )
  if additional_diagnostics:
    metrics.update(additional_diagnostics)

  logging.info(
      "Validation completed in %.3f seconds.\n"
      "Validation metrics for step %d:\n%s", elapsed, state.step,
      "\n".join(f"  {k}: {_prettify_value(v)}" for k, v in metrics.items()))

  return float(metrics[state.validation_metric]), metrics


def _validate_batches(model: models.Model, task_key: tasks.TaskKey,
                      batch_iter: dataloaders.BatchIterator,
                      device: torch.device) -> Dict[str, float]:
  """Performs a validation run and report the metrics computed."""
  all_diagnostics = []
  for batch in batch_iter:
    targets = batch.label
    if targets is None:
      targets = batch.multi_label_one_hot

    _, diagnostics = model.loss_and_metrics(
        torch.tensor(batch.image).to(device),
        torch.tensor(targets).to(device),
        is_training=False,
        task_key=task_key)
    diagnostics = {k: np.mean(v) for k, v in diagnostics.items()}
    all_diagnostics.append(diagnostics)

  return _dict_mean(all_diagnostics)


def _dict_mean(dict_list):
  mean_dict = {}
  for key in dict_list[0].keys():
    mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
  return mean_dict


def fit_with_early_stopping(
    initial_train_state: trainer.TrainState,
    learning_rate_scheduler: lr_scheduler._LRScheduler,
    task_key: tasks.TaskKey,
    train_iter_fn: Callable[[], Iterator[datasets.MiniBatch]],
    valid_iter_fn: Callable[[], Iterator[datasets.MiniBatch]],
    validation_metric: str,
    run_validation_every_n_steps: int,
    early_stopping_grace: int,
    max_steps: int,
    device: torch.device,
    train_metrics_writer: datawriter_interface.DataWriter,
    validation_metrics_writer: datawriter_interface.DataWriter,
    write_checkpoint: Callable[[FitWithEarlyStoppingState], None],
    checkpoint_to_resume: Optional[FitWithEarlyStoppingState] = None,
    additional_diagnostics: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, Any], trainer.TrainState, float]:
  """Fits model with early stopping and dynamic LR schduling."""
  additional_diagnostics = additional_diagnostics or {}

  if checkpoint_to_resume is None:
    logging.info("Starting new train loop...")
    state = FitWithEarlyStoppingState(  # pytype: disable=wrong-arg-types  # mapping-is-not-sequence
        step=0,
        best_age=0,
        best_metric=np.inf,
        train_state=initial_train_state,
        best_train_state=None,
        validation_metric=validation_metric,
    )
  else:
    logging.info("Resuming train loop from checkpoint...")
    state: FitWithEarlyStoppingState = checkpoint_to_resume

  step_timer = trainer.StepCountEstimator()
  train_iter = train_iter_fn()

  model, optimizer = initial_train_state.model, initial_train_state.optimizer
  model.to(device)

  while state.step < max_steps:
    start_time = time.monotonic()
    batch = next(train_iter)
    logging.log_every_n_seconds(logging.INFO,
                                "Step: %d/%d, Batch %s, Steps per second: %f",
                                LOG_INTERVAL_SECONDS, state.step + 1, max_steps,
                                batch, step_timer.estimated_steps_per_second())

    targets = batch.label
    if targets is None:
      targets = batch.multi_label_one_hot

    optimizer.zero_grad()
    loss, train_metrics = model.loss_and_metrics(
        torch.tensor(batch.image).to(device),
        torch.tensor(targets).to(device),
        is_training=True,
        task_key=task_key)
    loss.backward()

    optimizer.step()
    learning_rate_scheduler.step()

    train_metrics = {k: np.mean(v) for k, v in train_metrics.items()}
    train_metrics.update(
        step=state.step,
        steps_per_second=step_timer.estimated_steps_per_second())
    train_metrics.update(additional_diagnostics)
    train_metrics_writer.write(train_metrics)

    if state.step % CHECKPOINT_INTERVAL == 0 and state.step != 0:
      logging.info("Writing checkpoint at step %d", state.step)
      write_checkpoint(state)

    if state.step % run_validation_every_n_steps == 0:
      validation_metric, valid_metrics = _run_validation(
          model, task_key, state, valid_iter_fn(), device)
      validation_metrics_writer.write(valid_metrics)
      if validation_metric < state.best_metric:
        state.best_metric = validation_metric
        state.best_train_state = trainer.TrainState(
            model=copy.deepcopy(model).cpu(), optimizer=None)
        state.best_age = 0
      else:
        state.best_age += 1

      if state.best_age >= early_stopping_grace:
        logging.info("Validation metrics plateaued, stopping training.")
        break

    step_timer.add_measurement(time.monotonic() - start_time)

    state.step += 1

  logging.info("Running final validation.")
  validation_metric, valid_metrics = _run_validation(model, task_key, state,
                                                     valid_iter_fn(), device)
  validation_metrics_writer.write(valid_metrics)
  if validation_metric < state.best_metric:
    state.best_metric = validation_metric
    state.best_train_state = trainer.TrainState(
        model=copy.deepcopy(model).cpu(), optimizer=None)
    state.best_age = 0

  train_flops = state.step * _estimate_flops(model, batch, task_key)
  return validation_metric, valid_metrics, state.best_train_state, train_flops


def _estimate_flops(model: models.Model, batch: datasets.MiniBatch,
                    task_key: tasks.TaskKey) -> float:
  """Estimates number of flops doing forward/backward on a single batch."""
  targets = batch.label
  if targets is None:
    targets = batch.multi_label_one_hot
  x, y = torch.tensor(batch.image), torch.tensor(targets)

  def function():
    loss, _ = model.loss_and_metrics(x, y, is_training=True, task_key=task_key)
    loss.backward()

  flops = resources.estimate_flops(model, function)
  return flops


def _get_train_state_for_finetuning(
    config: ml_collections.ConfigDict,
    task_key: tasks.TaskKey,
    state: FinetuningLearnerState,
    finetuning_metrics: datawriter_interface.DataWriter,
) -> Optional[trainer.TrainState]:
  """Optionally returns a train state to fine tune from."""

  if config.finetuning.strategy is FinetuningStrategy.INDEPENDENT:
    logging.info("For independent training, no initial train state is used %s",
                 task_key)
    _write_finetuning_entry(finetuning_metrics, state, task_key, None)
    return None

  elif config.finetuning.strategy is FinetuningStrategy.PREVIOUS:
    if not state.train_states:
      logging.info(
          "Finetuning enabled for %s, but there are no previous tasks.",
          task_key)
      _write_finetuning_entry(finetuning_metrics, state, task_key, None)
      return None
    else:
      source_task, train_state_checkpoint_path = state.train_states[-1]
      train_state = trainer.restore_train_state(train_state_checkpoint_path)
      logging.info("Finetuning %s from previous task: %s.", task_key,
                   source_task)
      _write_finetuning_entry(finetuning_metrics, state, task_key, source_task)
      return train_state

  raise ValueError(f"Unsupported strategy: {config.finetuning_strategy}")


def _verify_all_tasks_supported(task_keys: Iterable[tasks.TaskKey]) -> None:
  unsupported_tasks = set(key.kind for key in task_keys) - SUPPORTED_TASK_KINDS
  if unsupported_tasks:
    raise NotImplementedError(
        f"Got unsupported tasks: {unsupported_tasks}. "
        "If required, you may use streams.FilteredStream "
        "to construct a stream that removes cetain tasks.")


def _get_latest_train_state_for_predictions(
    state: FinetuningLearnerState,
    task_key: tasks.TaskKey) -> trainer.TrainState:

  for key, train_state_checkpoint_path in reversed(state.train_states):
    if key == task_key:
      return trainer.restore_train_state(train_state_checkpoint_path)

  raise ValueError(
      f"Cannot compute predicions for task that has not been trained: {task_key}"
  )


def _build_model(config: ml_collections.ConfigDict,
                 task_keys: Sequence[tasks.TaskKey]) -> models.Model:
  """Constructs the parameterized, trainable model."""

  # In this learner, every task has its own set of parameters, and
  # so the backbone should be identical for all heads.
  return models.build_model(
      functools.partial(config.model.ctor, **config.model.kwargs),
      supported_tasks=task_keys,
      head_kwargs={"label_smoothing": config.label_smoothing})


def _write_finetuning_entry(
    finetuning_metrics: datawriter_interface.DataWriter,
    state: FinetuningLearnerState,
    current_task: tasks.TaskKey,
    finetune_from_task: Optional[tasks.TaskKey],
) -> None:
  """Writes the selected task to finetune from."""

  if finetune_from_task:
    finetune_from_task_name = finetune_from_task.name
  else:
    finetune_from_task_name = None

  finetuning_metrics.write({
      "index_of_train_event": len(state.train_states),
      "current_task": current_task.name,
      "finetune_from_task": finetune_from_task_name,
  })


def _prettify_value(value):
  try:
    return f"{value:.2f}"
  except ValueError:
    return f"{value}"
