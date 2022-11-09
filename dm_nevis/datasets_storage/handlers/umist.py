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

"""UMIST handler."""

import os

from typing import Dict

from dm_nevis.datasets_storage.handlers import extraction_utils as eu
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types

from tensorflow.io import gfile


def _path_to_label_fn(path: str, label_to_id: Dict[str, int]) -> int:
  label = os.path.dirname(path)
  return label_to_id[label]


def umist_handler(dataset_path: str) -> types.HandlerOutput:
  """Handler for UMIST dataset."""
  files = gfile.listdir(dataset_path)

  labels = [
      '1s', '1r', '1n', '1i', '1d', '1e', '1q', '1c', '1k', '1l', '1h', '1o',
      '1a', '1b', '1t', '1m', '1g', '1f', '1p', '1j'
  ]

  label_to_id = dict(
      ((label, label_id) for label_id, label in enumerate(labels)))

  metadata = types.DatasetMetaData(
      num_classes=20,
      num_channels=1,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_to_id=label_to_id,
          task_type='classification',
          image_type='face',
      ))

  def make_gen_fn():
    return eu.generate_images_from_tarfiles(
        *files,
        path_to_label_fn=lambda path: _path_to_label_fn(path, label_to_id),
        working_directory=dataset_path)

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)

  return metadata, per_split_gen


umist_dataset = types.DownloadableDataset(
    name='umist',
    download_urls=[
        types.DownloadableArtefact(
            url='http://eprints.lincoln.ac.uk/id/eprint/16081/1/face.tar.gz',
            checksum='11011ab5dded043f5e4331711c2407c8')
    ],
    website_url='http://eprints.lincoln.ac.uk/id/eprint/16081',
    handler=umist_handler)
