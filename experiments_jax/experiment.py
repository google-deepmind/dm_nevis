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

"""Creates a launchpad program for benchmarking learners.

This module provides an interface for constructing a launchpad program that can
benchmark learners using the benchmarker. Learners are configured using
instances of the `ExperimentConfig` class. This class may be initialized from an
`ml_collections.ConfigDict` using the `config_from_config_dict` function.

To benchmark a learner on launchpad, users must provide a function
`launchpad_learner_builder` that,

1) adds any learner-specific launchpad nodes to the launchpad program and,
2) returns a callable for building the learner.

We opt to pass around callable "builder" functions rather than passing around
instantiated objects, since these builder functions may be called directly on
the launchpad nodes where the objects are to be instantiated. This means that we
do not require that the object instances be serializable, and means that
launcher program does not have to initialize the stream or learner in order to
build the programs being launched.

When running on XManager, launchpad requires information on the resource types
to use for each program group. Every program defined by this module will
contain an `environment` group, which is the leader thread running the outer
loop of the environment. A single node will be started in this group
containing the environment. This node will also instantiate the learner
using the builder function returned by `launchpad_learner_builder`. For learners
that require only a single node, it may suffice to allocate a sufficiently large
resource type to the environment group.
"""

import dataclasses
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Protocol

from absl import logging
from dm_nevis.benchmarker.datasets import datasets
from dm_nevis.benchmarker.datasets import streams
from dm_nevis.benchmarker.datasets import tasks
from dm_nevis.benchmarker.environment import datawriter_interface
from dm_nevis.benchmarker.environment import environment
from dm_nevis.benchmarker.learners import learner_interface
from experiments_jax.environment import noop_checkpointer
from experiments_jax.metrics import lscl_metrics
import ml_collections


ProgramStopper = Callable[[], None]
LearnerBuilderFn = Callable[[], learner_interface.Learner]
StreamBuilderFn = Callable[[], streams.Stream]
DatasetLookupFn = Callable[[streams.DatasetKey], datasets.Dataset]
DatasetLookupBuilderFn = Callable[[], Tuple[DatasetLookupFn,
                                            Sequence[tasks.TaskKey]]]

BENCHMARKER_DATAFRAME = "benchmark"


class MetaLearnerBuilderFn(Protocol):
  """The signature of the function that prepares a learner to run on launchpad.

  Learners are given access to the launchpad program, which allows them to
  add auxiliary nodes to the launchpad program. The function then returns a
  callable used to initialize the learner. Note that the returned callable will
  be executed diretly on the node running the environment, this means that the
  learner does not need to be serializable. Similarly, the dataset lookup is
  wrapped in a builder function. This builder function must be serializable,
  but the actual dataset lookup returned from the builder need not be.

  In order to ensure graceful termination when using launchpad with threads
  (which we use for running tests on TAP), learners can provide a function for
  gracefully terminating any resources that they have spawned.
  """

  def __call__(
      self, *, dataset_lookup_builder: DatasetLookupBuilderFn,
      learner_config: ml_collections.ConfigDict
  ) -> Tuple[LearnerBuilderFn, ProgramStopper]:
    """Callable used to initialize the learner.

    Args:
      dataset_lookup_builder: A function that returns a dataset lookup, and the
        sequence of training task keys that will be fed to the learner. This is
        a 'builder' function since we want to be able to construct the objects
        directly on the machines where they will run, rather than constructing
        them in the launchpad main process and then pickling the functions.
      learner_config: The learner-specific configuration.

    Returns:
      A function for constructing a learner satisfying the learner interface,
      and a function for gracefully stopping the learner's resources.
    """


@dataclasses.dataclass
class LearnerConfig:
  learner_builder: MetaLearnerBuilderFn
  config: ml_collections.ConfigDict


@dataclasses.dataclass
class StreamConfig:
  ctor: Callable[..., streams.Stream]
  kwargs: Mapping[str, Any]


@dataclasses.dataclass
class ExperimentConfig:
  resume_from_checkpoint_path: str
  stream: StreamConfig
  learner: LearnerConfig


def config_from_config_dict(cfg: ml_collections.ConfigDict) -> ExperimentConfig:
  """Constructs a typed experiment config from an untyped config dict."""
  resume_from_checkpoint_path = cfg.resume_from_checkpoint_path
  stream_config = StreamConfig(**cfg.stream)
  learner_config = LearnerConfig(**cfg.learner)

  return ExperimentConfig(
      resume_from_checkpoint_path=resume_from_checkpoint_path,
      stream=stream_config,
      learner=learner_config,
  )


def _stopper():
  return


def run_program(config: ExperimentConfig):
  """Prepares a launchpad program to be executed."""

  stream_builder, dataset_lookup_builder = _stream_builders(config.stream)

  logging.info("Building learner to run on launchpad")
  learner_builder, learner_stopper = config.learner.learner_builder(
      dataset_lookup_builder=dataset_lookup_builder,
      learner_config=config.learner.config,
  )
  benchmark_metrics_writer = config.learner.config.get_metrics_writer(
      "benchmarker")
  return _run_environment(config.resume_from_checkpoint_path, stream_builder,
                          learner_builder, learner_stopper, _stopper,
                          benchmark_metrics_writer)


def _run_environment(checkpoint_restore_path: Optional[str],
                     stream_builder: StreamBuilderFn,
                     learner_builder: LearnerBuilderFn,
                     learner_stopper: ProgramStopper, stopper: ProgramStopper,
                     benchmark_metrics_writer: datawriter_interface.DataWriter):
  """Runs the environment."""

  learner = learner_builder()
  stream = stream_builder()

  checkpointer = noop_checkpointer.NoOpCheckpointer(
      restore_path=checkpoint_restore_path)

  metrics = lscl_metrics.lscl_metrics(stream.get_dataset_by_key,
                                      benchmark_metrics_writer)
  optional_checkpoint_to_resume = checkpointer.restore()
  output = environment.run(
      learner,
      stream,
      metrics,
      write_checkpoint=checkpointer.write,
      checkpoint_to_resume=optional_checkpoint_to_resume,
  )

  metrics = {
      **output.results,
      **dataclasses.asdict(output.train_resources_used)
  }
  logging.info("Benchmark Results: %s", metrics)

  benchmark_metrics_writer.close()  # Flush and close metrics writer

  logging.info("Stopping Launchpad...")
  learner_stopper()
  stopper()


def _stream_builders(
    config: StreamConfig) -> Tuple[StreamBuilderFn, DatasetLookupBuilderFn]:
  """Builds functions that can instantiate the stream and dataset lookup."""

  def stream_builder():
    return config.ctor(**config.kwargs)

  def dataset_lookup_builder():
    stream = stream_builder()
    task_keys = _all_train_task_keys(stream)
    return stream.get_dataset_by_key, task_keys

  return stream_builder, dataset_lookup_builder


def _all_train_task_keys(stream: streams.Stream) -> Sequence[tasks.TaskKey]:
  task_keys = []

  # TODO: Consider adding this to the stream interface.
  for event in stream.events():
    if isinstance(event, streams.TrainingEvent):
      dataset = stream.get_dataset_by_key(event.train_dataset_key)
      task_keys.append(dataset.task_key)

  return task_keys
