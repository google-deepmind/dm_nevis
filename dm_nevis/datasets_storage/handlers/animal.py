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

"""Animal handler."""

import os
import zipfile

from dm_nevis.datasets_storage.handlers import extraction_utils as utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types

from tensorflow.io import gfile


def _path_to_label_fn(path: str, label_to_id):
  label = os.path.split(os.path.split(path)[0])[1]
  return label_to_id[label]


# pylint:disable=missing-function-docstring
def animal_handler(dataset_path: str) -> types.HandlerOutput:
  files = gfile.listdir(dataset_path)
  labels = []

  for file in files:
    with zipfile.ZipFile(os.path.join(dataset_path, file), 'r') as zf:
      labels.extend(set([member.split('/')[1] for member in zf.namelist()[1:]]))

  num_classes = len(labels)
  num_channels = 1
  label_to_id = dict(((label, idx) for idx, label in enumerate(labels)))

  metadata = types.DatasetMetaData(
      num_classes=num_classes,
      num_channels=num_channels,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(label_to_id=label_to_id))

  def make_gen_fn():
    return utils.generate_images_from_zip_files(
        dataset_path,
        files,
        path_to_label_fn=lambda path: _path_to_label_fn(path, label_to_id))

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)

  return (metadata, per_split_gen)


animal_dataset = types.DownloadableDataset(
    name='animal',
    download_urls=[
        types.DownloadableArtefact(
            url='http://xiang.bai.googlepages.com/non_rigid_shape_A.zip',
            checksum='d88d44dc6d2382de3a5857d86fb5d430'),
        types.DownloadableArtefact(
            url='http://xiang.bai.googlepages.com/non_rigid_shape_B.zip',
            checksum='dae4a05b9797a3109078d1553173b9a8')
    ],
    handler=animal_handler,
)
