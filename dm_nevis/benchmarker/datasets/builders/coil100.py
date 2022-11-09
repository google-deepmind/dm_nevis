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

"""Dataset builders for COIL100."""

from typing import Optional

from dm_nevis.benchmarker.datasets import dataset_builders
from dm_nevis.benchmarker.datasets import datasets
from dm_nevis.benchmarker.datasets import tasks
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


DATASET_NAME = "coil100"
TASK_METADATA = tasks.ClassificationMetadata(num_classes=1_00)
TASK_KEY = tasks.TaskKey(DATASET_NAME, tasks.TaskKind.CLASSIFICATION,
                         TASK_METADATA)
SHUFFLE_BUFFER_SIZE = 10_000
# Angle ranges are between 0 and 71, since each unit corresponds to an increment
# of 5 degrees (360/5=72). For instance, 0 means between 0 and 4, 1 between 5
# and 9, etc.
# We take frontal views for training (around 0 degrees).
# We take side views for dev and dev-test (opposite side),
# and use for test all the reamining views.
NUM_ANGLE_RANGES = 72
NUM_OBJS = 100
SPLIT_ANGLE_RANGES = {"train": [70, 71, 0, 1, 2], "dev": [16, 17, 18, 19, 20],
                      "dev_test": [50, 51, 52, 53, 54]}


def _keep(i: int) -> bool:
  return (i not in SPLIT_ANGLE_RANGES["train"] and
          (i not in SPLIT_ANGLE_RANGES["dev"])
          and (i not in SPLIT_ANGLE_RANGES["dev_test"]))

SPLIT_ANGLE_RANGES["test"] = [i for i in range(NUM_ANGLE_RANGES) if _keep(i)]
SPLIT_ANGLE_RANGES["train_and_dev"] = SPLIT_ANGLE_RANGES[
    "train"] + SPLIT_ANGLE_RANGES["dev"]


def get_dataset(split: str,
                *,
                outer_start: Optional[int] = None,
                outer_end: Optional[int] = None) -> datasets.Dataset:
  """Get the COIL100 dataset."""

  builder = tfds.builder(DATASET_NAME)
  # Since there are 100 images per pose, we calculate the number of sample for
  # each split.
  # TODO: Support use of start and stop indexes.
  num_samples = NUM_OBJS * len(SPLIT_ANGLE_RANGES[split])
  metadata = dataset_builders.TFDSMetadata(num_samples)

  def builder_fn(shuffle: bool,
                 start: Optional[int] = None,
                 end: Optional[int] = None) -> tf.data.Dataset:
    # start/end are used to slice a dataset in the stream (done by
    # the learner), while outer_start/outer_end are used to slice a dataset
    # while constructing a stream (done by the designer of the stream).

    builder.download_and_prepare()
    ds = builder.as_dataset(split="train", shuffle_files=False)
    split_array = np.zeros((NUM_ANGLE_RANGES,))  # (0, …,0)
    # Turn the list of angle ranges into a binary vector with 1's indicating
    # which angle ranges we select for that split.
    # For instance, there is going to be a 1 in the first position if there are
    # images with views between [0, 4] degrees.
    np.put(split_array, SPLIT_ANGLE_RANGES[split], 1)  # (0,0, 1, 0, 0, 0, 1..)

    def _filter_fn(example):
      angle = example["angle_label"]  # integer from 0 to NUM_ANGLE_RANGES - 1
      result = tf.gather(tf.convert_to_tensor(split_array), angle)  # -> 0 | 1
      return tf.cast(result, tf.bool)

    ds = ds.filter(_filter_fn)  # leave the elements with desired angle

    # Slice if needed.
    indices = dataset_builders.combine_indices(outer_start, start, outer_end,
                                               end)
    # NOTE: Do not shuffle the data. Dataset needs to be deterministic for this
    # construction to work.
    if indices[0] is not None:
      ds = ds.skip(indices[0])

    if indices[1] is not None:
      ds = ds.take(indices[1] - (indices[0] or 0))

    if shuffle:
      # Note: We entirely rely on the shuffle buffer to randomize the order.
      ds = ds.shuffle(SHUFFLE_BUFFER_SIZE)

    def _to_minibatch_fn(data) -> datasets.MiniBatch:
      return datasets.MiniBatch(
          image=data["image"],
          label=data["object_id"],
          multi_label_one_hot=None,
      )

    return ds.map(_to_minibatch_fn)

  return datasets.Dataset(
      builder_fn=builder_fn,
      task_key=TASK_KEY,
      num_examples=metadata.num_examples)
