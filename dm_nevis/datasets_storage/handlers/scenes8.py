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

"""Scenes8 handler."""

import os

from dm_nevis.datasets_storage.handlers import extraction_utils as utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types

from tensorflow.io import gfile


def _path_to_label_fn(path: str, label_to_id):
  label = os.path.split(path)[1].split('_')[0]
  return label_to_id[label]


# pylint:disable=missing-function-docstring
def scenes8_handler(dataset_path: str) -> types.HandlerOutput:
  files = gfile.listdir(dataset_path)

  labels = [
      'highway', 'street', 'tallbuilding', 'forest', 'insidecity',
      'opencountry', 'mountain', 'coast'
  ]
  label_to_id = dict(
      ((label, label_id) for label_id, label in enumerate(labels)))

  metadata = types.DatasetMetaData(
      num_classes=len(labels),
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_to_id=label_to_id,
          task_type='classification',
          image_type='scene',
      ))

  def make_gen_fn():
    return utils.generate_images_from_zip_files(
        dataset_path,
        files,
        path_to_label_fn=lambda path: _path_to_label_fn(path, label_to_id))

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)

  return metadata, per_split_gen


scenes8_dataset = types.DownloadableDataset(
    name='scenes8',
    download_urls=[
        types.DownloadableArtefact(
            url='https://people.csail.mit.edu/torralba/code/spatialenvelope/spatial_envelope_256x256_static_8outdoorcategories.zip',
            checksum='c26fe529d49848091a759d7aadd267f5')
    ],
    handler=scenes8_handler)
