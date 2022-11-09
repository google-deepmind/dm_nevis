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

"""Graz 02 dataset handler."""  # NOTYPO

import itertools
import os
import zipfile

from dm_nevis.datasets_storage.handlers import extraction_utils as utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types


ZIP_FNAMES = {'bikes': 'ig02-v1.0-bikes.zip',
              'cars': 'ig02-v1.0-cars.zip',
              'people': 'ig02-v1.0-people.zip'}

CLASS_NAME_TO_LABEL = {'bikes': 0,
                       'cars': 1,
                       'people': 2}

IMAGE_NAME_SUFFIX = '.image.png'


# pylint:disable=missing-function-docstring
def ig02_handler(artifacts_path: str) -> types.HandlerOutput:

  metadata = types.DatasetMetaData(
      num_classes=3,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata={
          'class_names': list(CLASS_NAME_TO_LABEL.keys()),
      })

  def gen(split):
    # path is in the format of "{class_name}/{fname}{IMAGE_NAME_SUFFIX}".
    if split in ['train', 'split']:
      split_fnames = []
      for class_name, zip_fname in ZIP_FNAMES.items():
        with zipfile.ZipFile(
            os.path.join(artifacts_path, zip_fname), 'r') as zf:
          fnames_path = f'{class_name}_{split}.txt'
          with zf.open(fnames_path, 'r') as f:
            for line in f:
              fname = line.decode().strip()
              split_fnames.append(f'{class_name}/{fname}{IMAGE_NAME_SUFFIX}')
      split_fnames = set(split_fnames)
      path_filter = lambda path: path in split_fnames
    else:
      # Include all image files.
      path_filter = lambda path: path.endswith(IMAGE_NAME_SUFFIX)

    return utils.generate_images_from_zip_files(
        artifacts_path, list(ZIP_FNAMES.values()),
        path_to_label_fn=_label_from_path,
        path_filter=path_filter)

  # TODO: Make more efficient deduplication algorithm.
  make_gen_fn = utils.deduplicate_data_generator(
      itertools.chain(gen('train'), gen('test')))

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)

  return metadata, per_split_gen


def _label_from_path(path: str) -> types.Label:
  # path is in the format of "{class_name}/{fname}.image.png".
  return CLASS_NAME_TO_LABEL[os.path.dirname(path)]


ig02_dataset = types.DownloadableDataset(
    name='ig02',  # NOTYPO
    download_urls=[
        types.DownloadableArtefact(
            url='http://lear.inrialpes.fr/people/marszalek/data/ig02/ig02-v1.0-bikes.zip',
            checksum='13266fdf968176fa3aebdd439184254f'),
        types.DownloadableArtefact(
            url='http://lear.inrialpes.fr/people/marszalek/data/ig02/ig02-v1.0-cars.zip',
            checksum='34de933832755ee009c6f0e9d9c6426e'),
        types.DownloadableArtefact(
            url='http://lear.inrialpes.fr/people/marszalek/data/ig02/ig02-v1.0-people.zip',
            checksum='f80d8d21f018197979c72b977986fd2f')
    ],  # NOTYPO
    handler=ig02_handler)  # NOTYPO
