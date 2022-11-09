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

"""Defines a stream where all data is constructed at runtime.

This dataset has been created for use in unit tests.
"""

from typing import Iterator

from dm_nevis.benchmarker.datasets import datasets
from dm_nevis.benchmarker.datasets import streams
from dm_nevis.benchmarker.datasets.builders import test_dataset


class TestStream:
  """An stream over in-memory test datasets."""

  def __init__(self):

    self._datasets_by_key = {
        'train_0_10':
            test_dataset.get_dataset(split='train', start=0, end=5),
        'test':
            test_dataset.get_dataset(split='test', start=0, end=5),
        'train_10_20':
            test_dataset.get_dataset(split='train', start=2, end=7),
    }

    self._events = [
        streams.TrainingEvent('train_0_10', 'train_0_10', 'train_0_10'),
        streams.PredictionEvent('test'),
        streams.TrainingEvent('train_10_20', 'train_10_20', 'train_10_20'),
        streams.PredictionEvent('test'),
    ]

  def get_dataset_by_key(self,
                         dataset_key: streams.DatasetKey) -> datasets.Dataset:
    return self._datasets_by_key[dataset_key]

  def events(self) -> Iterator[streams.Event]:
    return iter(self._events)
