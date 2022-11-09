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

"""Not-MNIST handler."""

import os

from dm_nevis.datasets_storage.handlers import extraction_utils as utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types

from tensorflow.io import gfile

_IGNORED_FILES_REGEX = '|'.join([
    utils.DEFAULT_IGNORED_FILES_REGEX,
    r'notMNIST_small/notMNIST_small/A/RGVtb2NyYXRpY2FCb2xkT2xkc3R5bGUgQm9sZC50dGY=.png',
    r'notMNIST_small/notMNIST_small/F/Q3Jvc3NvdmVyIEJvbGRPYmxpcXVlLnR0Zg==.png'
])


def _path_to_label_fn(path: str, label_to_id):
  label = os.path.basename(os.path.dirname(path))
  return label_to_id[label]


def not_mnist_handler(dataset_path: str) -> types.HandlerOutput:
  """Not-MNIST dataset handler."""
  files = gfile.listdir(dataset_path)

  labels = [
      'A',
      'B',
      'C',
      'D',
      'E',
      'F',
      'G',
      'H',
      'I',
      'J',
  ]
  label_to_id = dict(
      ((label, label_id) for label_id, label in enumerate(labels)))

  metadata = types.DatasetMetaData(
      num_classes=len(labels),
      num_channels=1,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_to_id=label_to_id,
          task_type='classification',
          image_type='ocr',
      ))

  def make_gen_fn():
    return utils.generate_images_from_zip_files(
        dataset_path,
        files,
        path_to_label_fn=lambda path: _path_to_label_fn(path, label_to_id),
        ignored_files_regex=_IGNORED_FILES_REGEX,
        path_filter=lambda x: x.startswith('notMNIST_small/notMNIST_small'),
        convert_mode='L')

  # TODO: Make more efficient deduplication algorithm.
  make_unique_gen_fn = utils.deduplicate_data_generator(make_gen_fn())

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_unique_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)

  return metadata, per_split_gen


not_mnist_dataset = types.DownloadableDataset(
    name='not_mnist',
    download_urls=[types.KaggleDataset(
        dataset_name='jwjohnson314/notmnist',
        checksum='e2a47bb2a88c2c6bcae60d9f95223ace')],
    website_url='https://www.kaggle.com/jwjohnson314/notmnist',
    handler=not_mnist_handler)
