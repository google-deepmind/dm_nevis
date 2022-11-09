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

"""Defines the environment which is used to run a benchmark."""

import dataclasses
import datetime
import time
from typing import Callable, NamedTuple, Optional, Tuple

from absl import logging
import chex
from dm_nevis.benchmarker.datasets import streams
from dm_nevis.benchmarker.learners import learner_interface
from dm_nevis.benchmarker.metrics import metrics_aggregators


@chex.dataclass(frozen=True)
class EnvironmentState:
  """Represents the state of the environment.

  The environment state stores the synchronized model checkpoint and current
  position in the stream.

  Attributes:
    number_of_completed_events: The number of events completed when this state
      was written.
    within_training_event_checkpoint: If the learner writes a checkpoint during
      a train event, then we write an environment checkpoint with this field set
      to that checkpoint.
    learner_state: The state of the learner.
    metrics_state: The accumulated metrics computed by the environment.
    train_resources_used: The accumulated resource usage during training.
  """
  number_of_completed_events: int
  within_training_event_checkpoint: Optional[learner_interface.Checkpoint]
  learner_state: learner_interface.LearnerState
  metrics_state: metrics_aggregators.State
  train_resources_used: learner_interface.ResourceUsage


CheckpointWriterFn = Callable[[EnvironmentState], None]


class RunResult(NamedTuple):
  results: metrics_aggregators.Results
  train_resources_used: learner_interface.ResourceUsage
  final_learner_state: learner_interface.LearnerState


def no_op_checkpointer(state: EnvironmentState) -> None:
  """A checkpointer function that ignores the state."""
  del state


def run(
    learner: learner_interface.Learner,
    stream: streams.Stream,
    metrics: metrics_aggregators.MetricsAggregator,
    *,
    write_checkpoint: CheckpointWriterFn = no_op_checkpointer,
    checkpoint_to_resume: Optional[EnvironmentState] = None,
) -> RunResult:
  """Runs the interaction of a learner with a stream and computes metrics.

  Args:
    learner: The learner that will be exposed to the datasets in the stream.
    stream: A stream containing an iterable sequence of events to feed to the
      learner. To support resuming an environment from a checkpoint, the event
      sequence returned by the stream, up to the resumed point must be
      determnistic and identical for future runs.
    metrics: Defines the metrics aggregator that will be used to compute and
      publish the metrics resuting from this benchmarker run.
    write_checkpoint: A callable that stores intermediate environment state to a
      checkpoint.
    checkpoint_to_resume: If provided, the environment run is resumed from the
      given checkpointed state.

  Returns:
    The result of the metrics aggregator applied to the accumulated state
    computed across all prediction tasks, along with the resource usage
    during the run.
  """

  if not checkpoint_to_resume:
    state = EnvironmentState(
        number_of_completed_events=0,
        within_training_event_checkpoint=None,
        learner_state=learner.init(),
        metrics_state=metrics.init(),
        train_resources_used=learner_interface.ResourceUsage(
            floating_point_operations=0.0,
            peak_parameter_count=0,
            peak_parameter_size_bytes=0))
  else:
    logging.info("Restoring run from checkpoint...")
    state = checkpoint_to_resume

  for index_in_stream, event in enumerate(stream.events()):
    if index_in_stream < state.number_of_completed_events:
      logging.info("Skipping step %d: %s", index_in_stream, event)
      continue

    step_start_time = time.monotonic()
    logging.info("Step %d: %s", index_in_stream, event)

    if isinstance(event, streams.TrainingEvent):
      learner_state, resources = _train(state, event, learner, write_checkpoint)
      metrics_state = metrics.aggregate_train_event(state.metrics_state, event,
                                                    resources)
      state = dataclasses.replace(
          state,
          metrics_state=metrics_state,
          learner_state=learner_state,
          train_resources_used=state.train_resources_used.combine(resources),
      )
    elif isinstance(event, streams.PredictionEvent):
      predictions = learner.predict(event, state.learner_state)
      metrics_state = metrics.aggregate_predict_event(state.metrics_state,
                                                      event, predictions)
      state = dataclasses.replace(state, metrics_state=metrics_state)
    else:
      raise ValueError(f"Unknown stream task type {type(event)}")

    state = dataclasses.replace(
        state,
        within_training_event_checkpoint=None,
        number_of_completed_events=index_in_stream + 1,
    )

    write_checkpoint(state)

    logging.info(
        "Completed step %d: %s in %s",
        index_in_stream,
        event,
        datetime.timedelta(seconds=time.monotonic() - step_start_time),
    )

  return RunResult(
      results=metrics.compute_results(state.metrics_state),
      train_resources_used=state.train_resources_used,
      final_learner_state=state.learner_state,
  )


def _train(
    state: EnvironmentState,
    event: streams.TrainingEvent,
    learner: learner_interface.Learner,
    write_checkpoint: CheckpointWriterFn,
) -> Tuple[learner_interface.LearnerState, learner_interface.ResourceUsage]:
  """Runs a train dataset."""

  def write_train_event_checkpoint(learner_train_checkpoint):
    write_checkpoint(
        dataclasses.replace(
            state, within_training_event_checkpoint=learner_train_checkpoint))

  learner_state, resources_used = learner.train(
      event,
      state.learner_state,
      write_checkpoint=write_train_event_checkpoint,
      checkpoint_to_resume=state.within_training_event_checkpoint)

  logging.info("Resources used during train event: %s", resources_used)
  return learner_state, resources_used
