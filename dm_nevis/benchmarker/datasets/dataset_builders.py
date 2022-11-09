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

"""Wrappers to unify the interface of existing datasets."""

from typing import Any, Callable, NamedTuple, Optional, Tuple

from absl import logging
from dm_nevis.benchmarker.datasets import datasets
import tensorflow as tf
import tensorflow_datasets as tfds


class TFDSMetadata(NamedTuple):
  num_examples: Optional[int]


def tfds_dataset_builder_fn(
    tfds_name: str,
    *,
    split: str,
    start: Optional[int],
    end: Optional[int],
    shuffle_buffer_size: int,
    to_minibatch_fn: Callable[[Any], datasets.MiniBatch],
) -> Tuple[datasets.DatasetBuilderFn, TFDSMetadata]:
  """Provides a standard way to initialize a builder_fn from a tfds dataset.

  TODO: Explicit test for this function that uses a mock dataset.

  For efficiency, slicing indices are materialized and combined lazily when
  the dataset is actually constructed, this allows for the most efficient
  possible construction of the dataset.

  Args:
    tfds_name: The dataset name to load from tfds.
    split: The name of the split to load.
    start: The start index in the underlying data.
    end: The maximum end index in the underlying data.
    shuffle_buffer_size: The size of the shuffle buffer to use.
    to_minibatch_fn: A function to be mapped to the underlying dataset and
      create a datasets.MiniBatch matching the other datasets.

  Returns:
    a dataset builder function, along with metadata about the dataset.
  """

  builder = tfds.builder(tfds_name)
  metadata = _metadata_from_tfds_info(split, start, end, builder.info)

  outer_start, outer_end = start, end
  del start, end

  def builder_fn(shuffle: bool,
                 start: Optional[int] = None,
                 end: Optional[int] = None) -> tf.data.Dataset:

    indices = combine_indices(outer_start, start, outer_end, end)
    split_with_indices = _slice_str(split, *indices)

    builder.download_and_prepare()
    ds = builder.as_dataset(split=split_with_indices, shuffle_files=shuffle)

    if shuffle:
      ds = ds.shuffle(shuffle_buffer_size)

    return ds.map(
        to_minibatch_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=not shuffle)

  return builder_fn, metadata


def combine_indices(
    outer_start: Optional[int], inner_start: Optional[int],
    outer_end: Optional[int],
    inner_end: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
  """Combine starts and ends together.

  For an underlying sequence, this function combines an outer set of start:end
  and an inner set of start:end indices together, so that they may be applied
  in a single operation.

  Semantically: first the outer_start:outer_end sequence is selected,
  and then the inner_start:inner_end sequence is sampled from the result.

  In the diagram below, the returned (start, end) region is represented by
  the X's.

  |==============================================|
       ^                                   ^
       |================XXXXXXXXXX=========|
       |                ^        ^         |
        ` outer_start   |        |          ` outer_end
                        |        |
                        |        |
           inner_start '          ` inner_end

  Args:
    outer_start: Optional start index in input sequence.
    inner_start: Optional start index in input sequence, relative to
      outer_start.
    outer_end: Optional end index in input sequence.
    inner_end: Optional end index in input sequence, relative to outer_start.

  Returns:
    The region combing the start and end indices together.
  """

  # TODO: Support negative indices.
  assert (outer_start or 0) >= 0 and (inner_start or 0) >= 0
  assert (outer_end or 0) >= 0 and (inner_end or 0) >= 0

  combined_start = None
  if outer_start is not None or inner_start is not None:
    combined_start = (outer_start or 0) + (inner_start or 0)

  ends = []
  if outer_end is not None:
    ends.append(outer_end)

  if inner_end is not None:
    ends.append((outer_start or 0) + inner_end)

  if ends:
    combined_end = min(ends)
  else:
    combined_end = None

  if combined_end is not None and combined_start is not None:
    combined_start = min(combined_start, combined_end)

  return combined_start, combined_end


def _metadata_from_tfds_info(split: str, start: int, end: int,
                             info: tfds.core.DatasetInfo) -> TFDSMetadata:
  try:
    split_info = info.splits[_slice_str(split, start, end)]
  except KeyError:
    logging.warning("Cannot extract info for split. Is this mock data?")
    return TFDSMetadata(num_examples=None)

  return TFDSMetadata(num_examples=split_info.num_examples)


def _slice_str(split: str, start: Optional[int], end: Optional[int]) -> str:
  if start is None and end is None:
    return split
  return f"{split}[{start or ''}:{'' if end is None else end}]"
