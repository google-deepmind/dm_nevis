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

"""Dataset builders for DomainNet."""

from typing import List, Optional, Tuple

from dm_nevis.benchmarker.datasets import dataset_builders
from dm_nevis.benchmarker.datasets import datasets
from dm_nevis.benchmarker.datasets import tasks
from dm_nevis.benchmarker.datasets.builders import tfds_builder
import tensorflow as tf
import tensorflow_datasets as tfds


DATASET_NAME = 'domainnet'
DEFAULT_SHUFFLE_BUFFER_SIZE = 5_000
TASK_METADATA = tasks.ClassificationMetadata(num_classes=345)

TRAIN_DOMAINS = ['real', 'painting', 'clipart', 'quickdraw', 'infograph']

TEST_DOMAIN_DATASET = DATASET_NAME + '/sketch'
TASK_KEY = tasks.TaskKey(DATASET_NAME, tasks.TaskKind.CLASSIFICATION,
                         TASK_METADATA)


def merge_datasets(all_datasets: List[tf.data.Dataset]) -> tf.data.Dataset:
  assert all_datasets, 'The list of datasets to be merged is empty!'
  ds = all_datasets[0]
  for ds_i in all_datasets[1:]:
    ds = ds.concatenate(ds_i)
  return ds


def get_dataset_from_domains(
    domain_split_ls: List[Tuple[str, str]]) -> datasets.Dataset:
  """Gets a dataset given a list of (domain, split) tuples."""
  label_key = 'label'
  to_minibatch_fn = lambda x: _to_minibatch_single_label(x, label_key)
  num_examples = 0
  num_examples_ls = []
  builder_ls = []

  # Compute the num_examples and collect builders
  for curr_domain, curr_split in domain_split_ls:
    domain_dataset = f'{DATASET_NAME}/{curr_domain}'
    builder = tfds.builder(domain_dataset)
    builder_ls.append(builder)
    num_examples_ls.append(builder.info.splits[curr_split].num_examples)
    num_examples += builder.info.splits[curr_split].num_examples

  def builder_fn(shuffle: bool,
                 start: Optional[int] = None,
                 end: Optional[int] = None) -> tf.data.Dataset:
    all_datasets = []
    start = 0 if start is None else start
    end = num_examples if end is None else end
    for count, (_, curr_split) in enumerate(domain_split_ls):
      curr_domain_start = sum(num_examples_ls[:count])
      curr_domain_end = sum(num_examples_ls[:count + 1])
      # if the indices overlap with the current domain
      if start < curr_domain_end and end > curr_domain_start:
        local_start = max(start, curr_domain_start) - curr_domain_start
        local_end = min(end, curr_domain_end) - curr_domain_start
        indices = dataset_builders.combine_indices(None, local_start, None,
                                                   local_end)
        split_with_indices = _slice_str(curr_split, *indices)

        builder_ls[count].download_and_prepare()
        all_datasets.append(builder_ls[count].as_dataset(
            split=split_with_indices, shuffle_files=shuffle))
    # concatenate the datasets from different domains
    ds = merge_datasets(all_datasets)
    if shuffle:
      ds = ds.shuffle(DEFAULT_SHUFFLE_BUFFER_SIZE)

    return ds.map(to_minibatch_fn)

  return datasets.Dataset(
      builder_fn=builder_fn, task_key=TASK_KEY, num_examples=num_examples)


def get_dataset(split: str) -> datasets.Dataset:
  """Gets the DomainNet dataset."""
  if split == 'dev_test':
    dataset = tfds_builder.get_dataset(TEST_DOMAIN_DATASET, split='train')
    return dataset._replace(task_key=TASK_KEY)
  elif split == 'test':
    dataset = tfds_builder.get_dataset(TEST_DOMAIN_DATASET, split='test')
    return dataset._replace(task_key=TASK_KEY)
  elif split == 'dev':
    domain_splits = [(domain, 'test') for domain in TRAIN_DOMAINS]
    return get_dataset_from_domains(domain_splits)
  elif split == 'train':
    domain_splits = [(domain, 'train') for domain in TRAIN_DOMAINS]
    return get_dataset_from_domains(domain_splits)
  else:
    train_domain_splits = [(domain, 'train') for domain in TRAIN_DOMAINS]
    test_domain_splits = [(domain, 'test') for domain in TRAIN_DOMAINS]
    # Interleave the two (domain, split) lists
    domain_splits = []
    for train_domain_split, test_domain_split in zip(train_domain_splits,
                                                     test_domain_splits):
      domain_splits.append(train_domain_split)
      domain_splits.append(test_domain_split)

    return get_dataset_from_domains(domain_splits)


def _slice_str(split: str, start: Optional[int], end: Optional[int]) -> str:
  if start is None and end is None:
    return split
  return f"{split}[{start or ''}:{'' if end is None else end}]"


def _to_minibatch_single_label(data, label_key) -> datasets.MiniBatch:
  return datasets.MiniBatch(
      image=data['image'],
      label=data[label_key],
      multi_label_one_hot=None,
  )
