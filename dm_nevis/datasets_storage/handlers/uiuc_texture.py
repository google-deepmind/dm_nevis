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

"""UIUC Texture dataset handler."""

import os
from typing import List
import zipfile

from dm_nevis.datasets_storage.handlers import extraction_utils as utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types

from tensorflow.io import gfile


def _get_class_names_from_file(dataset_path: str, fname: str) -> List[str]:
  names = []
  with zipfile.ZipFile(os.path.join(dataset_path, fname), 'r') as zf:
    for name in sorted(zf.namelist()):
      f = zf.getinfo(name)

      if f.is_dir():
        names.append(os.path.split(f.filename)[0])
  return names


def _get_all_class_names(dataset_path: str, filenames: List[str]) -> List[str]:
  names = []
  for fname in filenames:
    names += _get_class_names_from_file(dataset_path, fname)
  return sorted(names)


def _path_to_label_fn(class_names: List[str]) -> utils.PathToLabelFn:
  def _path_to_label(fname):
    class_name = os.path.split(fname)[0]
    return class_names.index(class_name)
  return _path_to_label


def uiuc_texture_handler(dataset_path: str) -> types.HandlerOutput:
  """Imports UIUC texture dataset.

  The dataset is split over 5 zip files, each of them contains a subset of
  classes. To generate the images, we go through the zip files sequentially and
  yield the images and their corresponding labels. The dataset does not provide
  random splits. The splits are then generated at random.

  Link:
  https://web.archive.org/web/20070829035029/http://www-cvr.ai.uiuc.edu/ponce_grp/data/index.html

  Args:
    dataset_path: Path with downloaded datafiles.

  Returns:
    Metadata and generator functions.
  """

  filenames = gfile.listdir(dataset_path)
  class_names = _get_all_class_names(dataset_path, filenames)

  metadata = types.DatasetMetaData(
      num_channels=1,
      num_classes=len(class_names),
      image_shape=(),  # Ignored for now.
      preprocessing='random_crop',  # select random crops in the images
      additional_metadata=dict(
          labels=class_names,
          task_type='classification',
          image_type='texture'
      ))

  def gen():
    return utils.generate_images_from_zip_files(
        dataset_path=dataset_path,
        zip_file_names=filenames,
        path_to_label_fn=_path_to_label_fn(class_names))

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      gen, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)

  return (metadata, per_split_gen)


uiuc_texture_dataset = types.DownloadableDataset(
    name='uiuc_texture',
    download_urls=[
        types.DownloadableArtefact(
            url='https://web.archive.org/web/20070829035029/http://www-cvr.ai.uiuc.edu/ponce_grp/data/texture_database/T01-T05.zip',
            checksum='e622e4708e336d51b3bbd45503618af1'),
        types.DownloadableArtefact(
            url='https://web.archive.org/web/20070829035029/http://www-cvr.ai.uiuc.edu/ponce_grp/data/texture_database/T06-T10.zip',
            checksum='50ab35564058d3f1d05e6f5d767db5df'),
        types.DownloadableArtefact(
            url='https://web.archive.org/web/20070829035029/http://www-cvr.ai.uiuc.edu/ponce_grp/data/texture_database/T11-T15.zip',
            checksum='d1c7584aeb0ab1e41c7157c60e84c3ad'),
        types.DownloadableArtefact(
            url='https://web.archive.org/web/20070829035029/http://www-cvr.ai.uiuc.edu/ponce_grp/data/texture_database/T16-T20.zip',
            checksum='d527ba7d820e1eeaef49f9c07b82aa34'),
        types.DownloadableArtefact(
            url='https://web.archive.org/web/20070829035029/http://www-cvr.ai.uiuc.edu/ponce_grp/data/texture_database/T21-T25.zip',
            checksum='74a29af10123f2be70ea481e5af3ec36')
    ],
    handler=uiuc_texture_handler,
    paper_title='A Training-free Classification Framework for Textures, Writers, and Materials',
    authors='R. Timofte1 and L. Van Gool',
    year='2012')
