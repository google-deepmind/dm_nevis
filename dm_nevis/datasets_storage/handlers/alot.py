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

"""ALOT dataset handler."""  # NOTYPO

import functools
import os
import tarfile

from dm_nevis.datasets_storage.handlers import extraction_utils as utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types

from tensorflow.io import gfile


# pylint:disable=missing-function-docstring
def alot_handler(dataset_path: str,
                 is_grey: bool = False) -> types.HandlerOutput:
  files = gfile.listdir(dataset_path)
  assert len(files) == 1
  alot_file = files[0]

  # Calculate number of classes.
  with tarfile.open(os.path.join(dataset_path, alot_file)) as tfile:
    num_classes = len(
        set([member.path.split('/')[1] for member in tfile.getmembers()[1:]]))

  metadata = types.DatasetMetaData(
      num_channels=1 if is_grey else 3,
      num_classes=num_classes,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict())

  def gen_split():
    return utils.generate_images_from_tarfiles(
        alot_file, working_directory=dataset_path,
        path_to_label_fn=lambda x: int(x.split('/')[1]) - 1)

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      gen_split, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)

  return (metadata, per_split_gen)


alot_dataset = types.DownloadableDataset(
    name='alot',  # NOTYPO
    download_urls=[
        types.DownloadableArtefact(
            url='http://aloi.science.uva.nl/public_alot/tars/alot_png4.tar',
            checksum='c3f489c9072d3469e43225c5906e01e2')
    ],  # NOTYPO
    handler=alot_handler)  # NOTYPO


alot_grey_dataset = types.DownloadableDataset(
    name='alot_grey',  # NOTYPO
    download_urls=[
        types.DownloadableArtefact(
            url='http://aloi.science.uva.nl/public_alot/tars/alot_grey4.tar',
            checksum='0f5b72b1fe1ac8381821d45e5087dd99')
    ],  # NOTYPO
    handler=functools.partial(alot_handler, is_grey=True))  # NOTYPO
