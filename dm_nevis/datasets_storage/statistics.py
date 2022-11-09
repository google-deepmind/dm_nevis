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

"""Accumulates and calculates statistics."""

import dataclasses
from typing import List
from dm_nevis.datasets_storage.handlers import types
import numpy as np

_FULL_DATASET_STATS_NAME = 'full_dataset_stats'


def _incremental_mean(x: np.ndarray, prev_mu: np.ndarray, n: int) -> np.ndarray:
  return 1. / n * ((n - 1) * prev_mu + x)


def _incremental_sigma_sq_with_sq_diff(x: np.ndarray, prev_mu: np.ndarray,
                                       mu: np.ndarray, prev_sq_diff: np.ndarray,
                                       n: int):
  sq_diff = prev_sq_diff + (x - prev_mu) * (x - mu)
  sigma_sq = sq_diff / n
  return sq_diff, sigma_sq


def _incremental_min(x: np.ndarray, prev_min: np.ndarray, n: int) -> np.ndarray:
  if n == 1:
    return x
  return np.minimum(x, prev_min)


def _incremental_max(x: np.ndarray, prev_max: np.ndarray, n: int) -> np.ndarray:
  if n == 1:
    return x
  return np.maximum(x, prev_max)


@dataclasses.dataclass
class StatisticsAccumulator:
  """Data structure to accumulate all the dataset statistics."""
  mean_per_channel: np.ndarray
  sigma_sq_per_channel: np.ndarray
  sq_diff_per_channel: np.ndarray
  max_per_channel: np.ndarray
  min_per_channel: np.ndarray
  num_examples_per_class: np.ndarray
  num_examples: int
  min_label: int
  max_label: int


class StatisticsCalculator(object):
  """Aggregates statistics over the whole dataset for each split."""

  def __init__(self, splits: List[str], metadata: types.DatasetMetaData):
    self._accumulator = dict()
    self._num_classes = metadata.num_classes
    self._num_channels = metadata.num_channels

    for split in tuple(splits) + (_FULL_DATASET_STATS_NAME,):
      self._accumulator[split] = StatisticsAccumulator(
          mean_per_channel=np.zeros((self._num_channels,), dtype=float),
          sigma_sq_per_channel=np.zeros((self._num_channels,), dtype=float),
          sq_diff_per_channel=np.zeros((self._num_channels,), dtype=float),
          max_per_channel=np.zeros((self._num_channels,), dtype=float),
          min_per_channel=np.zeros((self._num_channels,), dtype=float),
          num_examples_per_class=np.zeros((self._num_classes,), dtype=int),
          num_examples=0,
          min_label=self._num_classes + 1,
          max_label=-1)

  def _accumulate_for_split(self, image, label, split_accumulator):
    """Accumulates the statistics and updates `split_accumulator`."""
    split_accumulator.num_examples_per_class[label] += 1
    split_accumulator.num_examples += 1

    split_accumulator.min_label = min(label, split_accumulator.min_label)
    split_accumulator.max_label = max(label, split_accumulator.max_label)

    prev_mu = split_accumulator.mean_per_channel

    flatten_image = np.copy(image).reshape((-1, self._num_channels))

    x_mu = np.mean(flatten_image, axis=0)
    mu = _incremental_mean(x_mu, prev_mu, split_accumulator.num_examples)
    split_accumulator.mean_per_channel = mu

    prev_sq_diff = split_accumulator.sq_diff_per_channel

    x_sigma = np.mean(flatten_image, axis=0)
    sq_diff, sigma_sq = _incremental_sigma_sq_with_sq_diff(
        x_sigma, prev_mu, mu, prev_sq_diff, split_accumulator.num_examples)

    split_accumulator.sq_diff_per_channel = sq_diff
    split_accumulator.sigma_sq_per_channel = sigma_sq

    x_max = np.max(flatten_image, axis=0)
    split_accumulator.max_per_channel = _incremental_max(
        x_max, split_accumulator.max_per_channel,
        split_accumulator.num_examples)

    x_min = np.min(flatten_image, axis=0)
    split_accumulator.min_per_channel = _incremental_min(
        x_min, split_accumulator.min_per_channel,
        split_accumulator.num_examples)

  def accumulate(self, image, label, split):
    """Accumulates the statistics for the given split."""
    self._accumulate_for_split(image, label, self._accumulator[split])
    self._accumulate_for_split(image, label,
                               self._accumulator[_FULL_DATASET_STATS_NAME])

  def merge_statistics(self):
    """Merges all the statistics together."""
    merged_statistics = dict()
    for split, split_accumulator in self._accumulator.items():

      std_per_channel = np.sqrt(split_accumulator.sigma_sq_per_channel)
      label_imbalance = np.max(
          split_accumulator.num_examples_per_class) - np.min(
              split_accumulator.num_examples_per_class)

      dict_split_accumulator = dataclasses.asdict(split_accumulator)
      dict_split_accumulator['std_per_channel'] = std_per_channel
      dict_split_accumulator['label_imbalance'] = label_imbalance
      merged_statistics[split] = dict_split_accumulator
    return merged_statistics
