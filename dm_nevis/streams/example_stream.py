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

"""Defines a stream where all data is available in tfds."""

from typing import Iterator

from dm_nevis.benchmarker.datasets import datasets
from dm_nevis.benchmarker.datasets import streams
from dm_nevis.benchmarker.datasets.builders import tfds_builder

DEV_SIZE = 1_000


class ExampleStream:
  """An example stream over tfds datasets."""

  def __init__(self):

    # These datasets do not have predefined dev splits, so we create
    # our own splits, where the dev set is assigned `DEV_SIZE` examples.
    self._datasets_by_key = {
        # MNIST
        'mnist_train_and_dev':
            tfds_builder.get_dataset(dataset_name='mnist', split='train'),
        'mnist_train':
            tfds_builder.get_dataset(
                dataset_name='mnist', split='train', start=DEV_SIZE),
        'mnist_dev':
            tfds_builder.get_dataset(
                dataset_name='mnist', split='train', end=DEV_SIZE),
        'mnist_test':
            tfds_builder.get_dataset(dataset_name='mnist', split='test'),
        # CIFAR10
        'cifar10_train_and_dev':
            tfds_builder.get_dataset(dataset_name='cifar10', split='train'),
        'cifar10_train':
            tfds_builder.get_dataset(
                dataset_name='cifar10', split='train', start=DEV_SIZE),
        'cifar10_dev':
            tfds_builder.get_dataset(
                dataset_name='cifar10', split='train', end=DEV_SIZE),
        'cifar10_test':
            tfds_builder.get_dataset(dataset_name='cifar10', split='test'),
        # CIFAR100
        'cifar100_train_and_dev':
            tfds_builder.get_dataset(dataset_name='cifar100', split='train'),
        'cifar100_train':
            tfds_builder.get_dataset(
                dataset_name='cifar100', split='train', start=DEV_SIZE),
        'cifar100_dev':
            tfds_builder.get_dataset(
                dataset_name='cifar100', split='train', end=DEV_SIZE),
        'cifar100_test':
            tfds_builder.get_dataset(dataset_name='cifar100', split='test'),
    }

    self._events = [
        # MNIST
        streams.TrainingEvent('mnist_train', 'mnist_train_and_dev',
                              'mnist_dev'),
        streams.PredictionEvent('mnist_test'),
        # CIFAR10
        streams.TrainingEvent('cifar10_train', 'cifar10_train_and_dev',
                              'cifar10_dev'),
        streams.PredictionEvent('cifar10_test'),
        # CIFAR100
        streams.TrainingEvent('cifar100_train', 'cifar100_train_and_dev',
                              'cifar100_dev'),
        streams.PredictionEvent('cifar100_test'),
    ]

  def get_dataset_by_key(self,
                         dataset_key: streams.DatasetKey) -> datasets.Dataset:
    return self._datasets_by_key[dataset_key]

  def events(self) -> Iterator[streams.Event]:
    return iter(self._events)
