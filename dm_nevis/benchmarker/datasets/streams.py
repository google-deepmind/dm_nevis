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

"""Define interface for streams.

Streams are defined to be sequences of datasets containing either train or
predict entries. Each of these has a description, and a key which may be used
to instantiate the relevant dataset.
"""

from typing import Any, Callable, Iterable, Iterator, Mapping, NamedTuple, Sequence, Union, Protocol

from absl import logging
from dm_nevis.benchmarker.datasets import datasets
from dm_nevis.benchmarker.datasets import tasks


DatasetKey = str


class TrainingEvent(NamedTuple):
  """A stream event corresponding to training a learner with new data.

  Attributes:
    train_dataset_key: The main default dataset to use for training. This does
      not include data from the `dev_dataset_key` dataset given below.
    train_and_dev_dataset_key: This is the full training dataset to be used by
      learners that wish to use their own logic to define a dev / train split.
    dev_dataset_key: A predefined dev (validation) dataset, to be used within a
      learner's train step for hyperparameter searching.
  """
  train_dataset_key: DatasetKey
  train_and_dev_dataset_key: DatasetKey
  dev_dataset_key: DatasetKey


class PredictionEvent(NamedTuple):
  """A stream event corresponding to running prediction with a learner.

  Attributes:
    dataset_key: The dataset key corresponding to either the `test` or
      `dev-test` split of the data (depending on the stream being executed).
      These events are used to measure the performance of learners on unseen
      data.
  """
  dataset_key: DatasetKey


Event = Union[TrainingEvent, PredictionEvent]


class Stream(Protocol):

  def get_dataset_by_key(self, dataset_key: str) -> datasets.Dataset:
    ...

  def events(self) -> Iterator[Event]:
    ...


class FilteredStream:
  """A stream for wrapping other streams, and removing unsupported tasks.

  This is provided to test learners that do not support all tasks in the input
  stream.
  """

  def __init__(
      self,
      stream_ctor: Callable[..., Stream],
      supported_task_kinds: Iterable[tasks.TaskKind],
      **stream_kwargs: Mapping[str, Any],
  ):

    self._stream = stream_ctor(**stream_kwargs)
    self._supported_task_kinds = set(supported_task_kinds)

  def get_dataset_by_key(self, dataset_key: str) -> datasets.Dataset:
    result = self._stream.get_dataset_by_key(dataset_key)
    assert result.task_key.kind in self._supported_task_kinds
    return result

  def events(self) -> Iterator[Event]:
    """Returns an iterator over the (filtered) stream events."""
    for event in self._stream.events():

      # This relies on datasets having consistent task_keys
      # across all of the keys. If this assumption were to fail, an assertion
      # error would be raised in get_dataset_by_key() above.
      if isinstance(event, PredictionEvent):
        dataset = self._stream.get_dataset_by_key(event.dataset_key)
      else:
        dataset = self._stream.get_dataset_by_key(event.train_dataset_key)

      if dataset.task_key.kind not in self._supported_task_kinds:
        logging.warning("Skipping unsupported event: %s, task key: %s", event,
                        dataset.task_key)
        continue

      yield event


def all_dataset_keys(event: Event) -> Sequence[DatasetKey]:
  """Returns all dataset keys for an event."""

  if isinstance(event, TrainingEvent):
    return [
        event.train_dataset_key,
        event.train_and_dev_dataset_key,
        event.dev_dataset_key,
    ]

  elif isinstance(event, PredictionEvent):
    return [event.dataset_key]

  raise ValueError(f"Unknown event type: {type(event)}")
