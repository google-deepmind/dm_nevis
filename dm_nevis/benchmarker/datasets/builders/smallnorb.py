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

"""Dataset builders for Small Norb."""

from typing import Optional

from dm_nevis.benchmarker.datasets import dataset_builders
from dm_nevis.benchmarker.datasets import datasets
from dm_nevis.benchmarker.datasets import tasks
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


DATASET_NAME = 'smallnorb'
SHUFFLE_BUFFER_SIZE = 10_000
TASK_METADATA = tasks.ClassificationMetadata(num_classes=5)
TASK_KEY = tasks.TaskKey(DATASET_NAME, tasks.TaskKind.CLASSIFICATION,
                         TASK_METADATA)


# pylint:disable=missing-function-docstring
def get_dataset(split: str,
                *,
                outer_start: Optional[int] = None,
                outer_end: Optional[int] = None) -> datasets.Dataset:
  """Small Norb dataset."""
  # There are 5 object categories, and each category has 10 object instances.
  # 5 instances are used for testing, and 5 for training in the original
  # dataset. We are going to split the original training set into: train, dev
  # and dev-test. We are going to use images from a particular instance for dev,
  # and images from another particular instance for dev-test. Images from the
  # remaining 3 instances are used for the train split. This way we guarantee
  # that we test for actual generalization.

  def _to_minibatch_fn(data) -> datasets.MiniBatch:
    return datasets.MiniBatch(
        image=data['image'],
        label=data['label_category'],
        multi_label_one_hot=None,
    )

  if split == 'test':  # We can use original test split.
    builder_fn, metadata = dataset_builders.tfds_dataset_builder_fn(
        tfds_name='smallnorb',
        shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
        split=split,
        start=None,
        end=None,
        to_minibatch_fn=_to_minibatch_fn)
    return datasets.Dataset(
        builder_fn=builder_fn,
        task_key=TASK_KEY,
        num_examples=metadata.num_examples)

  builder = tfds.builder(DATASET_NAME)
  builder.download_and_prepare()
  ds = builder.as_dataset(split='train')
  instance_ids = [4, 6, 7, 8, 9]  # Instances present in original train split
  ids = dict()
  ids['dev'] = [instance_ids[0]]
  ids['dev_test'] = [instance_ids[1]]
  ids['train'] = instance_ids[2:]
  ids['train_and_dev'] = ids['train'] + ids['dev']
  num_samples = {'dev': 0, 'dev_test': 0, 'train': 0}
  for el in ds.as_numpy_iterator():
    el_id = el['instance']
    if el_id in ids['dev']:
      num_samples['dev'] += 1
    elif el_id in ids['dev_test']:
      num_samples['dev_test'] += 1
    elif el_id in ids['train']:
      num_samples['train'] += 1
    else:
      raise ValueError(f'Unknown instance id {el_id}')
  num_samples['train_and_dev'] = num_samples['train'] + num_samples['dev']
  metadata = dataset_builders.TFDSMetadata(num_samples[split])

  def builder_fn_train(shuffle: bool,
                       start: Optional[int] = None,
                       end: Optional[int] = None) -> tf.data.Dataset:
    # start/end are used to slice a dataset in the stream (done by
    # the learner), while outer_start/outer_end are used to slice a dataset
    # while constructing a stream (done by the designer of the stream).
    builder.download_and_prepare()
    ds = builder.as_dataset(split='train', shuffle_files=False)
    split_array = np.zeros((10,))  # (0, …,0)
    np.put(split_array, ids[split], 1)  # (0,0, 1, 0, 0, 0, 1..)

    def _filter_fn(example):
      instance_id = example['instance']
      result = tf.gather(tf.convert_to_tensor(split_array), instance_id)
      return tf.cast(result, tf.bool)

    ds = ds.filter(_filter_fn)  # leave the elements with desired instance id

    # Slice if needed.
    indices = dataset_builders.combine_indices(outer_start, start, outer_end,
                                               end)
    if indices[0] is not None:
      ds = ds.skip(indices[0])

    if indices[1] is not None:
      ds = ds.take(indices[1] - (indices[0] or 0))

    if shuffle:
      ds = ds.shuffle(SHUFFLE_BUFFER_SIZE)

    return ds.map(_to_minibatch_fn)

  return datasets.Dataset(
      builder_fn=builder_fn_train,
      task_key=TASK_KEY,
      num_examples=metadata.num_examples)
