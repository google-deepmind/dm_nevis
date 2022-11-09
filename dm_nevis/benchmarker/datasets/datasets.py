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

"""Interface to access datasets."""

from typing import Any, NamedTuple, Optional, Union, Protocol

import chex
from dm_nevis.benchmarker.datasets import tasks
import numpy as np
import tensorflow as tf


Array = Union[tf.Tensor, np.ndarray]


class DatasetBuilderFn(Protocol):

  def __call__(self,
               *,
               shuffle: bool,
               start: Optional[int] = None,
               end: Optional[int] = None) -> tf.data.Dataset:
    """A function to build a tf.data.Dataset with an offset and max length.

    Args:
      shuffle: Whether or not to shuffle the data within the sampled region.
      start: Start at this index in the underlying dataset sequence
      end: Stop reading if the index in the underlying stream reaches this
        value.

    Returns:
      A tf.data.Dataset over ``dataset.MiniBatch`` objects of single unbatched
      examples.
    """


class Dataset(NamedTuple):
  """A handle for building a tf.data.Dataset.

  Attributes:
    builder_fn: A pure function to instantiate the dataset. Allows specifying of
      basic operations, such as enabling shuffling, and reading from specific
      sub-sequences of the underlying data. Each time the builder is called, a
      new separate dataset reader context is created, so users may call this
      multiple times to obtain separate dataset readers.
    task_key: Identifies the task that is containined in this dataset.
    num_examples: If available, provides the number of examples in iterable
      dataset constructed by the builder_fn.
  """
  builder_fn: DatasetBuilderFn
  task_key: tasks.TaskKey
  num_examples: Optional[int]


@chex.dataclass
class MiniBatch:
  """A shared MiniBatch representation for all tasks and datasets.

  By definition, a minibatch may have any number of batch dimensions, including
  zero batch dimensions (which is the default case for datasets returned by
  dataset builder functions).

  Attributes:
    image: If the dataset has an image, it will be stored here.
    label: The task specific label, if available.
    multi_label_one_hot: Multi-label in a one-hot format, if available.
  """
  image: Optional[Array]
  label: Optional[Any]
  multi_label_one_hot: Optional[Any]

  def __repr__(self):
    return _batch_repr(self)


def _batch_repr(batch: MiniBatch) -> str:
  """Writes a human-readable representation of a batch to a string."""

  feats = []
  if batch.image is not None:
    feats.append(f"image ({batch.image.shape})")

  parts = [f"features: {', '.join(feats)}"]

  if batch.label is not None:
    parts.append(f"label: {batch.label}")

  if batch.multi_label_one_hot is not None:
    parts.append(f"multi_label_one_hot: {batch.multi_label_one_hot}")

  return f"<MiniBatch: {', '.join(parts)}>"
