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

"""MNIST-m handler."""

import os
import re
import tarfile
from dm_nevis.datasets_storage.handlers import extraction_utils as eu
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
from PIL import Image

_DATA_FNAME = 'mnist_m.tar.gz'
_TRAIN_LABELS_FNAME = 'mnist_m/mnist_m_train_labels.txt'
_TEST_LABELS_FNAME = 'mnist_m/mnist_m_test_labels.txt'
_FNAME_AND_LABEL_REGEX = r'([\d]+.png) ([\d]+)'


def _parse_labels(labels_fname, tf):
  """Parses the labels and filenames for given label_fname from a tarfile."""
  read_buffer = tf.extractfile(labels_fname)
  if read_buffer is None:
    raise ValueError(f'Failed to read {labels_fname}')
  fname_to_label_list = read_buffer.read().decode('utf-8').split('\n')
  parsed_labels = dict()
  for fname_to_label in fname_to_label_list:
    if not fname_to_label:
      continue
    regex_match = re.search(_FNAME_AND_LABEL_REGEX, fname_to_label)
    if regex_match is None:
      raise ValueError('Regex match returned None result.')
    fname, label = regex_match.groups()
    label = int(label)
    parsed_labels[fname] = label
    # parsed_labels.append((fname, label))
  return parsed_labels


def mnist_m_handler(dataset_path: str) -> types.HandlerOutput:
  """Handler for MNIST-m dataset."""

  with tarfile.open(os.path.join(dataset_path, _DATA_FNAME)) as tf:
    train_fname_labels = _parse_labels(_TRAIN_LABELS_FNAME, tf)
    test_fname_labels = _parse_labels(_TEST_LABELS_FNAME, tf)

  def gen(fname_to_labels):
    with tarfile.open(os.path.join(dataset_path, _DATA_FNAME), 'r:gz') as tf:
      for member in tf.getmembers():
        image_fname = os.path.basename(member.path)
        if image_fname not in fname_to_labels:
          continue
        image = Image.open(tf.extractfile(member))
        image.load()
        label = fname_to_labels[image_fname]
        yield (image, label)

  metadata = types.DatasetMetaData(
      num_classes=10,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          task_type='classification',
          image_type='ocr',
      ))

  # TODO: Make more efficient deduplication algorithm.
  merged_fname_labels = train_fname_labels
  merged_fname_labels.update(test_fname_labels)
  make_gen_fn = eu.deduplicate_data_generator(gen(merged_fname_labels))

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)

  return metadata, per_split_gen


mnist_m_dataset = types.DownloadableDataset(
    name='mnist_m',
    download_urls=[
        types.DownloadableArtefact(
            url='https://drive.google.com/uc?export=download&id=0B_tExHiYS-0veklUZHFYT19KYjg&confirm=t',
            checksum='859df31c91afe82e80e5012ba928f279')
    ],
    website_url='http://yaroslav.ganin.net/',
    handler=mnist_m_handler)
