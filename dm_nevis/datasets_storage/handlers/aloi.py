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

"""ALOI dataset handler."""

import functools

from dm_nevis.datasets_storage.handlers import extraction_utils as utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types

from tensorflow.io import gfile

_TRAIN_CATEGORIES = ('col', 'view')
_TEST_CATEGORIES = ('ill',)


# TODO: handle 'masks' file as we for now don't know what it is for.
# pylint:disable=missing-function-docstring
def aloi_handler(
    dataset_path: str,
    is_grey: bool = False,
) -> types.HandlerOutput:
  files = gfile.listdir(dataset_path)

  metadata = types.DatasetMetaData(
      num_channels=1 if is_grey else 3,
      num_classes=1000,  # We know that.
      image_shape=(),  # Ignored for now.
      additional_metadata=dict())

  def _get_fname_for_category(files, category):
    for fname in files:
      if category in fname:
        return fname

  def gen_data_for_categories(categories, files):
    fnames = [
        _get_fname_for_category(files, category) for category in categories
    ]
    return utils.generate_images_from_tarfiles(
        *fnames,
        working_directory=dataset_path,
        path_to_label_fn=lambda x: int(x.split('/')[1]) - 1)

  train_gen_fn = lambda: gen_data_for_categories(_TRAIN_CATEGORIES, files)

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      train_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN,
      splits.MERGED_TRAIN_AND_DEV)
  per_split_gen['test'] = gen_data_for_categories(_TEST_CATEGORIES, files)

  return (metadata, per_split_gen)


aloi_dataset = types.DownloadableDataset(
    name='aloi',
    download_urls=[
        types.DownloadableArtefact(
            url='http://aloi.science.uva.nl/tars/aloi_red4_ill.tar',
            checksum='0f72c3acf66908e61b6f8d5bf5981193'),
        types.DownloadableArtefact(
            url='http://aloi.science.uva.nl/tars/aloi_red4_col.tar',
            checksum='c0486588258de8a26df4ed7aff0b34f8'),
        types.DownloadableArtefact(
            url='http://aloi.science.uva.nl/tars/aloi_red4_view.tar',
            checksum='ca29f342bb0c1fa4788e9f76f591ec59'),
        types.DownloadableArtefact(
            url='http://aloi.science.uva.nl/tars/aloi_red4_stereo.tar',
            checksum='44fef81680375641ea374c32dbe6d68d')
        # TODO: Need to handle masks somehow, but it is not specified
        # on the website
        # 'http://aloi.science.uva.nl/tars/aloi_mask4.tar',
    ],
    handler=aloi_handler)

aloi_grey_dataset = types.DownloadableDataset(
    name='aloi_grey',
    download_urls=[
        types.DownloadableArtefact(
            url='http://aloi.science.uva.nl/tars/aloi_grey_red4_ill.tar',
            checksum='70eca95ebfcfb253eb7be199f158f10c'),
        types.DownloadableArtefact(
            url='http://aloi.science.uva.nl/tars/aloi_grey_red4_col.tar',
            checksum='91a31a2692797c899973e84783c08e92'),
        types.DownloadableArtefact(
            url='http://aloi.science.uva.nl/tars/aloi_grey_red4_view.tar',
            checksum='0347be95470dbb22cc02e441d8b7e956'),
        types.DownloadableArtefact(
            url='http://aloi.science.uva.nl/tars/aloi_grey_red4_stereo.tar',
            checksum='bc059412fd6620fd49feb89655284f4a')
        # TODO: Need to handle masks somehow, but it is not specified
        # on the website
        # 'http://aloi.science.uva.nl/tars/aloi_mask4.tar',
    ],
    handler=functools.partial(aloi_handler, is_grey=True))
