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

"""Builders for Nevis datasets."""

import dataclasses
import os
from typing import Callable, Optional

from absl import logging
from dm_nevis.benchmarker.datasets import dataset_builders
from dm_nevis.benchmarker.datasets import datasets
from dm_nevis.benchmarker.datasets import tasks
from dm_nevis.datasets_storage import dataset_loader
from dm_nevis.datasets_storage import encoding
import tensorflow as tf

_MULTI_LABEL_DATASETS = [
    "sun_attributes",
    "voc_actions",
    "pascal_voc2007",
    "iaprtc12",
    "nih_chest_xray",
    "awa2",
    "biwi",
]


def _nevis_builder_fn(
    name: str,
    split: str,
    version: str,
    start: Optional[int],
    end: Optional[int],
    to_minibatch_fn: Callable[..., datasets.MiniBatch],
    path: Optional[str] = None,
) -> datasets.DatasetBuilderFn:
  """Builds a builder_fn for nevis datasets."""
  outer_start, outer_end = start, end
  del start, end

  def builder_fn(shuffle: bool,
                 start: Optional[int] = None,
                 end: Optional[int] = None) -> tf.data.Dataset:
    # Combine inner and outer interval boundaries
    start, end = dataset_builders.combine_indices(outer_start, start, outer_end,
                                                  end)

    if path:
      dataset = dataset_loader.load_dataset_from_path(path, split)
    else:
      dataset = dataset_loader.load_dataset(name, split, version)

    ds = dataset.builder_fn(shuffle=shuffle, start=start, end=end)

    return ds.map(
        to_minibatch_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  return builder_fn


def get_dataset(dataset: str,
                split: str,
                *,
                root_dir: Optional[str] = None,
                version: str = "stable",
                start: Optional[int] = None,
                end: Optional[int] = None) -> datasets.Dataset:
  """Get a Nevis dataset.

  Args:
    dataset: The name of the dataset to load.
    split: The name of the split to load.
    root_dir: If provided, loads data from this root directory, instead of from
      the default paths.
    version: If no root_dir is provided, load the version specified by this
      name.
    start: Offset from the start of the dataset to load.
    end: Offset from the end of the dataset.

  Returns:
    A dataset in the form of a datasets.Dataset.
  """

  # TODO: Consider only supporting loading by path.
  path = os.path.join(root_dir, dataset) if root_dir else None

  if path:
    metadata = dataset_loader.get_metadata_from_path(path)
  else:
    metadata = dataset_loader.get_metadata(dataset, version=version)

  available_splits = metadata.additional_metadata["splits"]
  num_classes = metadata.num_classes

  if split not in available_splits:
    raise ValueError(f"Requested nevis dataset `{dataset}`, split `{split}` "
                     f"but available splits are {available_splits}")

  max_end = metadata.additional_metadata["num_data_points_per_split"][split]
  num_examples = (end or max_end) - (start or 0)

  if dataset in _MULTI_LABEL_DATASETS:
    task_metadata = tasks.MultiLabelClassificationMetadata(
        num_classes=num_classes)
    task_key = tasks.TaskKey(dataset, tasks.TaskKind.MULTI_LABEL_CLASSIFICATION,
                             task_metadata)

    def to_minibatch(data) -> datasets.MiniBatch:
      multi_label_one_hot = tf.reduce_sum(
          tf.one_hot(data["multi_label"], num_classes), axis=0)

      return datasets.MiniBatch(
          image=data[encoding.DEFAULT_IMAGE_FEATURE_NAME],
          label=None,
          multi_label_one_hot=multi_label_one_hot,
      )
  else:
    task_metadata = tasks.ClassificationMetadata(num_classes=num_classes)
    task_key = tasks.TaskKey(dataset, tasks.TaskKind.CLASSIFICATION,
                             task_metadata)

    def to_minibatch(data) -> datasets.MiniBatch:
      return datasets.MiniBatch(
          image=data[encoding.DEFAULT_IMAGE_FEATURE_NAME],
          label=data[encoding.DEFAULT_LABEL_FEATURE_NAME],
          multi_label_one_hot=None,
      )

  # TODO: Remove this workaround.
  if dataset == "biwi":
    logging.warning("Applying patch to BIWI labels")
    patched_to_minibatch = lambda x: _patch_biwi_minibatch(to_minibatch(x))
  else:
    patched_to_minibatch = to_minibatch

  builder_fn = _nevis_builder_fn(
      name=dataset,
      split=split,
      version=version,
      start=start,
      end=end,
      to_minibatch_fn=patched_to_minibatch,
      path=path)

  return datasets.Dataset(
      builder_fn=builder_fn, task_key=task_key, num_examples=num_examples)


@tf.function
def _patch_biwi_minibatch(batch: datasets.MiniBatch) -> datasets.MiniBatch:
  """Fix an off-by-one-error in BIWI.

  TODO: Fix this in the underlying data.

  Due to a boundary error, angles that should have been [0,0,0,0,1] were
  assigned the value [0,0,0,0,0], and the 1 was added to the following bucket.
  This function fixes the bug.

  Args:
    batch: The batch to fix.

  Returns:
    A batch with the fix in place.
  """

  def fix_angle(angle):
    if tf.reduce_max(angle) == 0:
      return tf.constant([0, 0, 0, 0, 1], dtype=angle.dtype)
    elif tf.reduce_sum(angle) == 2:
      return angle - tf.constant([1, 0, 0, 0, 0], dtype=angle.dtype)
    else:
      return angle

  label = batch.multi_label_one_hot
  a1 = fix_angle(label[0:5])
  a2 = fix_angle(label[5:10])
  a3 = fix_angle(label[10:15])
  fixed_label = tf.concat([a1, a2, a3], axis=0)

  return dataclasses.replace(batch, multi_label_one_hot=fixed_label)
