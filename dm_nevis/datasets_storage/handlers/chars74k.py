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

"""Chars74k dataset handler."""

import itertools
import os
import tarfile
from typing import Iterable
from dm_nevis.datasets_storage.handlers import extraction_utils as eu
from dm_nevis.datasets_storage.handlers import splits as su
from dm_nevis.datasets_storage.handlers import types
import numpy as np
import pandas as pd
from PIL import Image
import scipy.io

from tensorflow.io import gfile


def chars74k_handler(
    dataset_path: str,
    handle_categories: Iterable[str] = frozenset({'Img', 'Hnd', 'Fnt'}),
) -> types.HandlerOutput:
  """Imports Chars74k dataset.

  Link: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
  The dataset comes with train/test(/val) splits. We keep the same test set and
  consider all the remaining images as training samples.
  Possible improvements for future iterations:
    - The category Img (see below) comes as Good and Bad images (regarding the
    image quality). In the current version, all images are taken into
    consideration but it would be possible to load them separately.
    - For now, we consider only the dataset of Latin characters. The website
    provides another dataset of Kannada characters that can be interesting to
    consider in the next iteration.
    - Some categories have richer annotations (segmentation, stroke trajectory,
    ...)
    Note: The images of the category Fnt are grey images, while the two other
    categories are RGB. All images are therefore converted into RGB to have a
    consistent format.

  Args:
    dataset_path: Path with downloaded datafiles.
    handle_categories: Images categories - The data come in 3 types:
      - Img: Cropped real images
      - Hnd: Hand written images captured from a tablet
      - Fnt: Generated characters from different computer fonts

  Returns:
    Metadata and generator functions.
  """

  files = gfile.listdir(dataset_path)

  img_markers = {
      'Img': 'Bmp',
      'Hnd': 'Img',
      'Fnt': '',
  }

  splits_columns = ['ALLnames', 'is_good', 'ALLlabels', 'classlabels',
                    'classnames', 'NUMclasses', 'TSTind', 'VALind',
                    'TXNind', 'TRNind']

  metadata = types.DatasetMetaData(
      num_channels=3,
      num_classes=62,  # We know that.
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          task_type='classification',
          image_type='ocr'
      ))

  def _get_fname_for_category(files, category):
    for fname in files:
      if category in fname:
        return fname

  def _get_fname_for_splits(files):
    for fname in files:
      if 'Lists' in fname:
        return fname

  def _get_splits_df_from_file(f_obj):
    splits = scipy.io.loadmat(f_obj)
    df = pd.DataFrame(splits['list'][0], columns=splits_columns)
    return df

  def _get_test_filenames(files):
    fname = _get_fname_for_splits(files)
    test_filenames = {}
    with tarfile.open(os.path.join(dataset_path, fname)) as tfile:
      for member in tfile.getmembers():
        if member.isdir():
          continue
        if 'English' in member.path:
          if 'lists.mat' in member.path or 'lists_var_size.mat' in member.path:
            name = member.path.split('/')[-2]
            f_obj = tfile.extractfile(member)
            df = _get_splits_df_from_file(f_obj)
            test_filenames[name] = df['ALLnames'][0][df['TSTind'][0][:, -1] - 1]
    return test_filenames

  def _is_test_image(path, category, test_filenames):
    return any(name in path for name in test_filenames[category])

  def _extract_label_from_path(path):
    starting_pos = path.find('Sample') + 6
    return path[starting_pos:starting_pos+3]

  def gen_data_for_categories_splits(categories, files, select_test=False):
    test_filenames = _get_test_filenames(files)
    for category in categories:
      fname = _get_fname_for_category(files, category)
      with tarfile.open(os.path.join(dataset_path, fname)) as tfile:
        for member in tfile.getmembers():
          if member.isdir():
            continue
          if img_markers[category] in member.path and '.png' in member.path:
            if _is_test_image(
                member.path, category, test_filenames) == select_test:
              f_obj = tfile.extractfile(member)
              label = np.array(int(_extract_label_from_path(member.path))) -1
              image = Image.open(f_obj).convert('RGB')
              image.load()
              yield (image, label)

  train_gen = gen_data_for_categories_splits(handle_categories, files)
  test_gen = gen_data_for_categories_splits(
      handle_categories, files, select_test=True)
  make_unique_gen_fn = eu.deduplicate_data_generator(
      itertools.chain(train_gen, test_gen))

  per_split_gen = su.random_split_generator_into_splits_with_fractions(
      make_unique_gen_fn, su.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      su.MERGED_TRAIN_AND_DEV)

  return (metadata, per_split_gen)

chars74k_dataset = types.DownloadableDataset(
    name='chars74k',
    download_urls=[
        types.DownloadableArtefact(
            url='http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishImg.tgz',
            checksum='85d157e0c58f998e1cda8def62bcda0d'),
        types.DownloadableArtefact(
            url='http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishHnd.tgz',
            checksum='95e9ee6d929809ef4c2f509f86388c72'),
        types.DownloadableArtefact(
            url='http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz',
            checksum='f6710b1302a7e9693d904b16389d9d8a'),
        types.DownloadableArtefact(
            url='http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/Lists.tgz',
            checksum='8dfe916169fa10b86466776a4a1d7c5f')
    ],
    handler=chars74k_handler,
)
