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

"""Split-Imagenet stream that consists of 100 ten-way classification tasks."""
from concurrent import futures
from typing import Dict, Iterator, Mapping, Sequence, Tuple
from dm_nevis.benchmarker.datasets import datasets
from dm_nevis.benchmarker.datasets import streams
from dm_nevis.benchmarker.datasets.builders import split_imagenet
from dm_nevis.streams import lscl_stream
import numpy as np

DEFAULT_THREADPOOL_WORKERS = 30


class SplitImagenetStream:
  """The LSCL benchmark stream.

  The stream adds a train event for each instance of the train data in the
  stream.

  Additionally, a predict event is added containing the test dataset after
  every instance of a train dataset.

  Once the stream is complete, a further predict event is added for every
  seen train event. This makes it possible to compare the performance on tasks
  from train time to the end of the stream.
  """

  def __init__(
      self,
      *,
      predict_event_splits: Sequence[lscl_stream.Split] = (
          lscl_stream.Split.DEV_TEST,),
      shuffle_seed: int = 1,
      shuffle_tasks_order: bool = False,
  ):
    """Instantiate a Split-Imagenet stream.

    Imagenet has 1000 classes, we split them into 100 disjoint subsets with
    10 classes each to create 100 ten-way classification tasks.

    Args:
      predict_event_splits: Sequence of splits to use for prediction.
      shuffle_seed: An integer denoting a seed for shuffling logic when
        `shuffle_datasets_order` are active.
      shuffle_tasks_order: Whether to shuffle the order of datasets.
    """
    self._events, self._datasets_by_key = _get_events_and_lookup(
        predict_event_splits=predict_event_splits,
        shuffle_seed=shuffle_seed,
        shuffle_datasets_order=shuffle_tasks_order)

  def get_dataset_by_key(self,
                         dataset_key: streams.DatasetKey) -> datasets.Dataset:
    return self._datasets_by_key[dataset_key]

  def events(self) -> Iterator[streams.Event]:
    return iter(self._events)


def _get_events_and_lookup(
    *,
    predict_event_splits: Sequence[lscl_stream.Split] = (
        lscl_stream.Split.DEV_TEST,),
    shuffle_seed: int = 1,
    shuffle_datasets_order: bool = False,
) -> Tuple[Sequence[streams.Event], Mapping[streams.DatasetKey,
                                            datasets.Dataset]]:
  """Constructs a sequence of events and a dataset lookup."""
  events = []
  lookup = {}
  datasets_by_key = {}
  task_indices = list(range(split_imagenet.N_TASKS))
  lookup = _build_lookup(task_indices)

  rng = np.random.default_rng(shuffle_seed)

  if shuffle_datasets_order:
    task_indices = list(task_indices)
    rng.shuffle(task_indices)

  for task_index in task_indices:
    result = lookup[task_index]
    if result is None:
      raise ValueError(f'Unable to read dataset for task_index = {task_index}')

    train_event = streams.TrainingEvent(
        train_dataset_key=result.train.key,
        dev_dataset_key=result.dev.key,
        train_and_dev_dataset_key=result.train_and_dev.key)
    events.append(train_event)

    for split in predict_event_splits:
      events.append(
          streams.PredictionEvent(lscl_stream.split_to_key(split, result)))

    datasets_by_key[result.train.key] = result.train.dataset
    datasets_by_key[result.test.key] = result.test.dataset
    datasets_by_key[result.dev.key] = result.dev.dataset
    datasets_by_key[result.dev_test.key] = result.dev_test.dataset
    datasets_by_key[result.train_and_dev.key] = result.train_and_dev.dataset

  return events, datasets_by_key


def _build_lookup(
    task_indices: Sequence[int]) -> Dict[int, lscl_stream.DatasetSplits]:
  """Creates a lookup for given dataset names."""

  with futures.ThreadPoolExecutor(
      max_workers=DEFAULT_THREADPOOL_WORKERS) as executor:
    result = executor.map(_get_dataset_splist_by_task, task_indices)

  return dict(zip(task_indices, result))


def _get_dataset_splist_by_task(task_index: int) -> lscl_stream.DatasetSplits:
  """Construct key and dataset for tfds dataset."""
  train_dataset = split_imagenet.get_dataset(
      task_index=task_index, split='train')
  dev_dataset = split_imagenet.get_dataset(task_index=task_index, split='dev')
  train_and_dev_dataset = split_imagenet.get_dataset(
      task_index=task_index, split='train_and_dev')
  dev_test_dataset = split_imagenet.get_dataset(
      task_index=task_index, split='dev_test')
  test_dataset = split_imagenet.get_dataset(task_index=task_index, split='test')

  dataset_key_prefix = train_dataset.task_key.name
  train_key = f'{dataset_key_prefix}_train'
  dev_key = f'{dataset_key_prefix}_dev'
  train_and_dev_key = f'{dataset_key_prefix}_train_and_dev'
  dev_test_key = f'{dataset_key_prefix}_dev_test'
  test_key = f'{dataset_key_prefix}_test'

  return lscl_stream.DatasetSplits(
      train=lscl_stream.KeyAndDataset(train_key, train_dataset),
      dev=lscl_stream.KeyAndDataset(dev_key, dev_dataset),
      train_and_dev=lscl_stream.KeyAndDataset(train_and_dev_key,
                                              train_and_dev_dataset),
      dev_test=lscl_stream.KeyAndDataset(dev_test_key, dev_test_dataset),
      test=lscl_stream.KeyAndDataset(test_key, test_dataset),
  )
