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

"""VisTex dataset handler."""

import os
import re
from typing import Any, List, Optional
from dm_nevis.datasets_storage.handlers import extraction_utils as utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
import ncompress
from PIL import Image

from tensorflow.io import gfile

_VALID_IMAGES_REGEX = r'^.+/Images/Reference/(.+)/.+.ppm'


def _decompress_archives(filenames: List[str],
                         foldername: str) -> None:
  for filename in filenames:
    if filename.endswith('tar.Z'):
      path = os.path.join(foldername, filename)
      with gfile.GFile(path, 'rb') as f:
        buffer = ncompress.decompress(f.read())
      with gfile.GFile(path[:-2], 'wb') as out:
        out.write(buffer)
      os.remove(path)


def _unpack_archives(dataset_path: str, foldername: str) -> None:
  filenames = gfile.listdir(os.path.join(dataset_path, foldername))
  for filename in filenames:
    if '.tar' in filename:
      utils.unpack_file(filename, os.path.join(dataset_path, foldername))


def _extract_classname_from_path(path: str) -> Optional[str]:
  match = re.match(_VALID_IMAGES_REGEX, path)
  if not match:
    raise ValueError(f'Failed to match class for {path}')
  return match.groups()[0]


def _get_all_filenames(path: str) ->  List[str]:
  all_filenames = []
  for path, _, files in os.walk(path):
    for filename in files:
      all_filenames.append(os.path.join(path, filename))
  return all_filenames


def _extract_all_classnames(all_filenames: List[str]) -> List[Any]:
  """Extracts all the names of classes present in the dataset."""
  classnames = []
  for filename in all_filenames:
    try:
      c = _extract_classname_from_path(filename)
      if c not in classnames:
        classnames.append(c)
    except ValueError:
      continue
  return classnames


def _extract_label_from_path(classname_list: List[str],
                             path: str) -> Optional[int]:
  c = _extract_classname_from_path(path)
  label = classname_list.index(c)
  return label


def vistex_handler(dataset_path: str) -> types.HandlerOutput:
  """Imports VisTex dataset.

  The dataset provides two types of images: reference textures and texture
  scenes. We load only the reference textures.

  The archive contains a collection of LZW-compressed archives.
  The data is organized in 2 subdirectories:
  - A subdirectory organised relatively to the images types (reference texture
  or texture scene) and classes. It contains symbolic links to the actual
  images.
  - A subdirectory that contains the actual images arranged according to their
  resolution.
  It is therefore necessary to unpack all the archives to be able to load the
  images we are interested in (i.e. reference textures) and assign the correct
  labels to them.

  Link:
  https://vismod.media.mit.edu/vismod/imagery/VisionTexture/vistex.html

  Args:
    dataset_path: Path with downloaded datafiles.

  Returns:
    Metadata and generator functions.
  """

  # TODO: Revisit to avoid writing data to disk.

  archive_path = gfile.listdir(dataset_path)[0]
  utils.unpack_file(archive_path, dataset_path)
  unpacked_folder_name_1 = 'VisionTexture'
  unpacked_folder_name_2 = 'VisTex'
  filenames = gfile.listdir(os.path.join(dataset_path, unpacked_folder_name_1))
  _decompress_archives(filenames,
                       os.path.join(dataset_path, unpacked_folder_name_1))
  _unpack_archives(dataset_path, unpacked_folder_name_1)

  all_filenames = _get_all_filenames(
      os.path.join(dataset_path, unpacked_folder_name_1,
                   unpacked_folder_name_2))

  class_names = _extract_all_classnames(all_filenames)

  metadata = types.DatasetMetaData(
      num_channels=3,
      num_classes=len(class_names),
      image_shape=(),  # Ignored for now.
      preprocessing='random_crop',  # select random crops in the images
      additional_metadata=dict(
          labels=class_names,
          task_type='classification',
          image_type='texture'
      ))

  def gen():
    for filename in all_filenames:
      if not re.match(_VALID_IMAGES_REGEX, filename):
        continue

      image = Image.open(filename)
      image.load()
      label = _extract_label_from_path(class_names, filename)
      yield (image, label)

  # TODO: review split function to make sure each class is present
  # at least in train and test.

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      gen, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)

  return (metadata, per_split_gen)


vistex_dataset = types.DownloadableDataset(
    name='vistex',
    download_urls=[
        types.DownloadableArtefact(
            url='http://vismod.media.mit.edu/pub/VisTex/VisTex.tar.gz',
            checksum='f176ad5c9383141f981e3668b232add7')
    ],
    handler=vistex_handler,
)
