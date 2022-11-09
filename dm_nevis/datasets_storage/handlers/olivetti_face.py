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

"""Olivetti handler."""
from dm_nevis.datasets_storage.handlers import extraction_utils as utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types


ARCHIVE_NAME = 'the-orl-database-for-training-and-testing.zip'


def _path_to_label_fn(path: str) -> int:
  label = int(path.split('/')[-1].split('.')[0].split('_')[1]) - 1
  return label


def olivetti_face_handler(dataset_path: str) -> types.HandlerOutput:
  """Olivetti dataset handler."""
  num_classes = 41

  metadata = types.DatasetMetaData(
      num_classes=num_classes,
      num_channels=1,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          task_type='classification',
          image_type='face',
      ))

  def make_gen_fn():
    return utils.generate_images_from_zip_files(
        dataset_path, [ARCHIVE_NAME],
        path_to_label_fn=_path_to_label_fn,
        convert_mode='L')

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)

  return metadata, per_split_gen


olivetti_face_dataset = types.DownloadableDataset(
    name='olivetti_face',
    download_urls=[
        types.KaggleDataset(
            dataset_name='tavarez/the-orl-database-for-training-and-testing',
            checksum='09871495160825a485b0f2595ba2bb34')
    ],
    website_url='https://www.kaggle.com/tavarez/the-orl-database-for-training-and-testing',
    handler=olivetti_face_handler)
