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

"""Pneumonia chest X-RAY dataset."""

import os

from dm_nevis.datasets_storage.handlers import extraction_utils as utils
from dm_nevis.datasets_storage.handlers import splits as su
from dm_nevis.datasets_storage.handlers import types

from tensorflow.io import gfile

_IGNORED_FILES_REGEX = '|'.join([
    utils.DEFAULT_IGNORED_FILES_REGEX,
    r'metadata.xlsx',
    r'README.md.txt',
    r'__MACOSX',
    r'DS_Store',
])

_LABELS = ['NORMAL', 'PNEUMONIA']


def _path_to_label_fn(path: str, label_to_id):
  label = os.path.basename(os.path.dirname(path))
  return label_to_id[label]


# pylint:disable=missing-function-docstring
def pneumonia_chest_xray_handler(dataset_path: str) -> types.HandlerOutput:
  files = gfile.listdir(dataset_path)
  label_to_id = dict(
      ((label, label_id) for label_id, label in enumerate(_LABELS)))

  metadata = types.DatasetMetaData(
      num_classes=2,
      num_channels=1,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_to_id=label_to_id,
          task_type='classification',
          image_type='xray',
      ))

  # pylint:disable=g-long-lambda
  def make_gen_fn(split, label_to_id):
    return utils.generate_images_from_zip_files(
        dataset_path,
        files,
        path_to_label_fn=lambda path: _path_to_label_fn(path, label_to_id),
        path_filter=lambda x: x.startswith(
            os.path.join('chest_xray/chest_xray', split)),
        ignored_files_regex=_IGNORED_FILES_REGEX,
        convert_mode='L')

  train_split_gen_fn = lambda: make_gen_fn('train', label_to_id)
  # Train and dev.
  per_split_gen = su.random_split_generator_into_splits_with_fractions(
      train_split_gen_fn, su.SPLIT_WITH_FRACTIONS_FOR_TRAIN_AND_DEV_ONLY,
      su.MERGED_TRAIN_AND_DEV)
  per_split_gen['dev-test'] = make_gen_fn('val', label_to_id)
  per_split_gen['test'] = make_gen_fn('test', label_to_id)

  return metadata, per_split_gen


pneumonia_chest_xray_dataset = types.DownloadableDataset(
    name='pneumonia_chest_xray',
    download_urls=[
        types.KaggleDataset(
            dataset_name='paultimothymooney/chest-xray-pneumonia',
            checksum='930763e3580e76de9c2c849ec933b5d6')
    ],
    website_url='https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia',
    handler=pneumonia_chest_xray_handler)
