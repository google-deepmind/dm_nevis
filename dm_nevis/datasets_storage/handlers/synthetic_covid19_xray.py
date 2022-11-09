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

"""Synthetic COVID-19 X-Ray dataset."""

import os

from typing import Dict

from dm_nevis.datasets_storage.handlers import extraction_utils as utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types

from tensorflow.io import gfile


def _path_to_label_fn(path: str, label_to_id: Dict[str, int]):
  dirname = os.path.dirname(path)
  if dirname == 'G1':
    return label_to_id['Normal']

  if dirname == 'G2':
    return label_to_id['Pneumonia']

  raise ValueError('Unknown label.')


# pylint:disable=missing-function-docstring
def synthetic_covid19_xray_handler(dataset_path: str) -> types.HandlerOutput:
  files = gfile.listdir(dataset_path)

  labels = ['Normal', 'Pneumonia']

  label_to_id = dict(
      ((label, label_id) for label_id, label in enumerate(labels)))

  metadata = types.DatasetMetaData(
      num_classes=len(labels),
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_to_id=label_to_id,
          task_type='classification',
          image_type='xray',
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


synthetic_covid19_xray_dataset = types.DownloadableDataset(
    name='synthetic_covid19_xray',
    download_urls=[
        types.DownloadableArtefact(
            url='https://github.com/hasibzunair/synthetic-covid-cxr-dataset/releases/download/v0.1/G_NC.zip',
            checksum='bd82149d00283fca892bc41e997a3070'),
        types.DownloadableArtefact(
            url='https://github.com/hasibzunair/synthetic-covid-cxr-dataset/releases/download/v0.1/G_PC.zip',
            checksum='acf56668f97be81fd8c05c4308f80c61')
    ],
    handler=synthetic_covid19_xray_handler)
