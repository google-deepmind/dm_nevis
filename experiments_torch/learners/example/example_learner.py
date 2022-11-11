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
"""A learner implemented for the baseline."""

import dataclasses
import functools
from typing import Iterable, Optional, Tuple

from absl import logging
from dm_nevis.benchmarker.datasets import streams
from dm_nevis.benchmarker.datasets import tasks
from dm_nevis.benchmarker.learners import learner_interface
from experiments_torch import experiment
import ml_collections
import numpy as np
import tensorflow_datasets as tfds


def learner_builder(
    dataset_lookup_builder: experiment.DatasetLookupBuilderFn,
    learner_config: ml_collections.ConfigDict
) -> Tuple[experiment.LearnerBuilderFn, experiment.ProgramStopper]:
  """Prepares the learner to run on launchpad."""
  del learner_config

  def _learner_builder():
    dataset_lookup, _ = dataset_lookup_builder()
    return build_example_learner(dataset_lookup)

  def _stopper():
    return

  return _learner_builder, _stopper


@dataclasses.dataclass(frozen=True)
class ExampleLearnerState:
  """The state for the example learner."""


def build_example_learner(
    dataset_lookup: experiment.DatasetLookupFn) -> learner_interface.Learner:
  return learner_interface.Learner(
      init=init,
      train=functools.partial(train, dataset_lookup=dataset_lookup),
      predict=functools.partial(predict, dataset_lookup=dataset_lookup),
  )


def init() -> ExampleLearnerState:
  return ExampleLearnerState()


def train(
    event: streams.TrainingEvent,
    state: ExampleLearnerState,
    write_checkpoint: learner_interface.CheckpointFn,
    *,
    checkpoint_to_resume: Optional[learner_interface.Checkpoint] = None,
    dataset_lookup: experiment.DatasetLookupFn,
) -> Tuple[ExampleLearnerState, learner_interface.ResourceUsage]:
  """Trains the learner given the given dataset."""

  del write_checkpoint, checkpoint_to_resume
  dataset = dataset_lookup(event.train_dataset_key)
  logging.info("Got train task: %s with %s examples", dataset.task_key,
               dataset.num_examples)

  return state, learner_interface.ResourceUsage(
      floating_point_operations=0.0,
      peak_parameter_count=0,
      peak_parameter_size_bytes=0)


def predict(
    event: streams.PredictionEvent,
    state: ExampleLearnerState,
    *,
    dataset_lookup: experiment.DatasetLookupFn,
) -> Iterable[learner_interface.Predictions]:
  """Computes predictions for each example in the referenced dataset."""

  del state

  dataset = dataset_lookup(event.dataset_key)
  logging.info("Got predict task: %s with %s examples", dataset.task_key,
               dataset.num_examples)

  batch_size = 1
  ds = dataset.builder_fn(shuffle=False).batch(batch_size=batch_size)

  for batch in tfds.as_numpy(ds):
    # For now, we make empty predictions.

    if dataset.task_key.kind == tasks.TaskKind.MULTI_LABEL_CLASSIFICATION:
      output = [
          np.zeros((batch_size,))
          for _ in range(dataset.task_key.metadata.num_classes)
      ]
    elif dataset.task_key.kind == tasks.TaskKind.CLASSIFICATION:
      output = [np.zeros((batch_size, dataset.task_key.metadata.num_classes))]
    else:
      raise ValueError(f"Unknown task kind: {dataset.task_key.kind}")
    yield learner_interface.Predictions(batch=batch, output=output)
