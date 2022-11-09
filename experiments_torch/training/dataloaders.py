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

"""A module to split and load training data."""

from typing import Callable, Iterator

from absl import logging
from dm_nevis.benchmarker.datasets import datasets
import tensorflow as tf
import tensorflow_datasets as tfds

BatchIterator = Iterator[datasets.MiniBatch]
PreprocFn = Callable[[datasets.MiniBatch], datasets.MiniBatch]

# For datasets containing fewer than this number of elements, we cache the
# dataset in memory, before preprocessing is applied. This avoids problematic
# cases where very small datasets require many requests to the underlying
# file storage.
DATASET_SIZE_TO_CACHE = 5_000


def build_train_iterator(dataset: datasets.Dataset, preproc_fn: PreprocFn,
                         batch_size: int) -> Callable[[], BatchIterator]:
  """Builds functions to iterate over train and validation data."""

  def build_iterator() -> BatchIterator:
    ds = dataset.builder_fn(shuffle=True)

    if dataset.num_examples < DATASET_SIZE_TO_CACHE:
      logging.info("Caching dataset with %d elements", dataset.num_examples)
      ds = ds.cache()
      buffer_size = min(DATASET_SIZE_TO_CACHE, dataset.num_examples)
      ds = ds.shuffle(buffer_size, reshuffle_each_iteration=True)

    ds = ds.repeat()
    ds = ds.map(
        preproc_fn,
        num_parallel_calls=tf.data.AUTOTUNE)  #, deterministic=False)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(10)
    return iter(tfds.as_numpy(ds))

  return build_iterator


def build_prediction_iterator(dataset: datasets.Dataset, preproc_fn: PreprocFn,
                              batch_size: int) -> Callable[[], BatchIterator]:
  """Builds an iterator over batches for use in prediction."""

  def build_iterator():
    ds = dataset.builder_fn(shuffle=False)
    ds = ds.map(preproc_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(10)
    return iter(tfds.as_numpy(ds))

  return build_iterator
