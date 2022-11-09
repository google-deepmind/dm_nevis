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

"""Dataset loader."""

import codecs
import functools
import os
from typing import Callable, NamedTuple, Optional
import dill
from dm_nevis.datasets_storage import download_util as du
from dm_nevis.datasets_storage import encoding
from dm_nevis.datasets_storage import handlers
from dm_nevis.datasets_storage import paths
from dm_nevis.datasets_storage import version_control as vc
from dm_nevis.datasets_storage.handlers import types
import tensorflow as tf

from tensorflow.io import gfile

MAXIMUM_DATA_FETCH_CONCURRENT_REQUESTS = 16
DEFAULT_BLOCK_LENGTH = 16
DEFAULT_SHUFFLE_BUFFER_SIZE = 5_000
DEFAULT_LRU_CACHE_SIZE = 1_000


class Dataset(NamedTuple):
  builder_fn: Callable[..., tf.data.Dataset]
  num_examples: int


class DatasetNotAvailableError(ValueError):
  """Raised when a dataset is not known or available."""


class DatasetNotReadyError(ValueError):
  """Raised when a dataset is known, but was not ready."""


def check_dataset_is_ready(dataset_name: str, version: str) -> None:
  """Raises an exception if the dataset is not found or not ready."""

  if not handlers.is_dataset_available(dataset_name):
    raise DatasetNotAvailableError(f'Dataset `{dataset_name}` not available.')

  path = vc.get_dataset_path(dataset_name, version=version)
  status = du.try_read_status(path)
  if status != du.DatasetDownloadStatus.READY:
    raise DatasetNotReadyError(
        f'Dataset `{dataset_name}` not ready to be used.')


@functools.lru_cache(maxsize=DEFAULT_LRU_CACHE_SIZE)
def get_metadata_from_path(path: str) -> types.DatasetMetaData:
  with gfile.GFile(os.path.join(path, paths.METADATA_FNAME), 'rb') as f:
    # tf.io.gfile has a bug with pickle, thus we write it to a temporary buffer
    # in base64 before decoding the pickle object.
    # It's only metadata of limited size thus the overhead is acceptable.
    # See https://stackoverflow.com/a/67171328
    pickled_object = f.read()
    return dill.loads(codecs.decode(pickled_object, 'base64'))


def get_metadata(dataset_name: str,
                 version: str = 'stable') -> types.DatasetMetaData:
  check_dataset_is_ready(dataset_name, version=version)
  path = vc.get_dataset_path(dataset_name, version=version)
  return get_metadata_from_path(path)


def load_dataset(dataset_name: str,
                 split: str,
                 version: str = 'stable') -> Dataset:
  """Loads the dataset given a name and a split."""
  check_dataset_is_ready(dataset_name, version=version)
  path = vc.get_dataset_path(dataset_name, version=version)
  return load_dataset_from_path(path, split)


def load_dataset_from_path(dataset_dir: str, split: str) -> Dataset:
  """Loads the dataset for a given split from a path."""

  metadata = get_metadata_from_path(dataset_dir)
  num_examples = _num_examples_from_metadata(split, metadata)
  num_classes = metadata.num_classes
  num_channels = metadata.num_channels
  encoding.get_default_features(num_channels, num_classes)
  # This is required for backwards compatibility. Later on, we will remove need
  # for default features.
  features = metadata.features or encoding.get_default_features(
      num_channels, num_classes)

  split_path = os.path.join(dataset_dir, split)
  shards = gfile.glob(os.path.join(split_path, 'data-*-of-*'))
  if not shards:
    raise ValueError(f'No shards found for split `{split}` at {dataset_dir}')

  def builder_fn(*,
                 shuffle: bool,
                 start: Optional[int] = None,
                 end: Optional[int] = None) -> tf.data.Dataset:

    # TODO: It's possible to support this by carefully keeping track
    # of the offsets and mapping them to the dataloaders, as tfds does.
    # We opt not to do this for now, however.
    if shuffle and (start is not None or end is not None):
      raise NotImplementedError(
          'Currently, offsets are not supported when shuffling')

    ds = tf.data.Dataset.from_tensor_slices(shards)

    if shuffle:
      # Shuffling the shards helps randomize across the full dataset.
      ds = ds.shuffle(buffer_size=len(shards))

    ds = ds.interleave(
        tf.data.TFRecordDataset,
        cycle_length=MAXIMUM_DATA_FETCH_CONCURRENT_REQUESTS,
        block_length=DEFAULT_BLOCK_LENGTH,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not shuffle)

    if start is not None:
      ds = ds.skip(start)
    if end is not None:
      ds = ds.take(end - start or 0)

    if shuffle:
      ds = ds.shuffle(DEFAULT_SHUFFLE_BUFFER_SIZE)

    example_decoder = encoding.build_example_decoder(features=features)
    ds = ds.map(
        example_decoder,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not shuffle)

    return ds

  return Dataset(num_examples=num_examples, builder_fn=builder_fn)


def _num_examples_from_metadata(split: str,
                                metadata: types.DatasetMetaData) -> int:
  points_per_split = metadata.additional_metadata['num_data_points_per_split']
  try:
    return points_per_split[split]
  except KeyError as e:
    raise ValueError(f'Split `{split}` not found in {points_per_split}') from e
