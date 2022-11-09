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

"""PACS dataset.

PACS (Photo-Art-Cartoon-Sketch) is a dataset for testing generalization.

The original split by the authors are not done by domains, so we use sketch
for dev-test / test.
"""

import os
from dm_nevis.datasets_storage.handlers import extraction_utils as eu
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
import h5py
import numpy as np
from PIL import Image

from tensorflow.io import gfile

IMAGE_SHAPE = (227, 227)

CATEGORIES = frozenset([
    'photo',
    'art_painting',
    'cartoon',
    'sketch',
])

CLASS_NAMES = (
    'dog',
    'elephant',
    'giraffe',
    'guitar',
    'horse',
    'house',
    'person',
)


SPLIT_WITH_FRACTIONS_FOR_TEST_AND_DEV_TEST = {'test': 0.5, 'dev-test': 0.5}


def handler(dataset_path: str) -> types.HandlerOutput:
  """A handler to download the PACS dataset."""

  metadata = types.DatasetMetaData(
      num_classes=len(CLASS_NAMES),
      num_channels=3,
      image_shape=IMAGE_SHAPE,
      additional_metadata={
          'class_names': CLASS_NAMES,
          'categories': CATEGORIES,
      })

  def gen(categories):
    for filename in gfile.listdir(dataset_path):
      # check if fname contains a domain in domain_ls
      if any(category_name in filename for category_name in categories):

        with h5py.File(os.path.join(dataset_path, filename), 'r') as f:
          # Get the data
          images = list(f['images'])
          labels = list(f['labels'])
          for image, label in zip(images, labels):
            # change image from BGR to RGB and label to start from 0
            yield Image.fromarray(image[..., ::-1].astype('uint8')), label-1

  train_make_gen_fn = eu.deduplicate_data_generator(
      gen(categories=['photo', 'art_painting', 'cartoon']))
  test_make_gen_fn = eu.deduplicate_data_generator(gen(categories=['sketch']))

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      train_make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN_AND_DEV_ONLY,
      splits.MERGED_TRAIN_AND_DEV)

  test_dev_test_gen = splits.random_split_generator_into_splits_with_fractions(
      test_make_gen_fn, SPLIT_WITH_FRACTIONS_FOR_TEST_AND_DEV_TEST)

  per_split_gen = per_split_gen | test_dev_test_gen

  return metadata, per_split_gen


def write_fixture(path: str) -> None:
  """Writes a fixture to the given path."""

  # For each category and class, write a single fixture image.
  for category in CATEGORIES:
    with h5py.File(os.path.join(path, category + '.hdf5'), 'w') as hf:
      images = list(
          np.random.randint(256, size=(len(CLASS_NAMES),) + IMAGE_SHAPE + (3,)))
      labels = [i + 1 for i in list(range(len(CLASS_NAMES)))]
      hf.create_dataset('images', data=images, dtype='uint16')
      hf.create_dataset('labels', data=labels, dtype='uint16')


pacs_dataset = types.DownloadableDataset(
    name='pacs',
    download_urls=[
        types.DownloadableArtefact(
            url='https://drive.google.com/u/0/uc?id=1e2WfiUmpv25FzRHYrA_8rooqEWicwbGA&export=download&confirm=y',
            checksum='7cd392ecb9e0ab0f0e8be9d8fc5ed5a2'),
        types.DownloadableArtefact(
            url='https://drive.google.com/u/0/uc?id=1qvJeF3WgfZgBgNBnzJGOLVOMncLgi5uN&export=download&confirm=y',
            checksum='d5df8be042fd2525efeb29cfb2252026'),
        types.DownloadableArtefact(
            url='https://drive.google.com/u/0/uc?id=10yRj3m8bB_PAiKqOcct1viTGu0DuT5un&export=download&confirm=y',
            checksum='a91c0ee93df8278028ff49072317e24a'),
        types.DownloadableArtefact(
            url='https://drive.google.com/u/0/uc?id=1ID0Y-v0EvKz1VL7XIKfZtb2FOxF89gVQ&export=download&confirm=y',
            checksum='e9205c7d19484ea8b5082abe1560dad3'),
        types.DownloadableArtefact(
            url='https://drive.google.com/u/0/uc?id=1BpaNvaSRXZ09xwnC5TWBv3ktOBj36mp7&export=download&confirm=y',
            checksum='988c767ea2e542268be87044e3da60f5'),
        types.DownloadableArtefact(
            url='https://drive.google.com/u/0/uc?id=16pF2YwohULpkXV3NNRiBvDy4SBWyhxvz&export=download&confirm=y',
            checksum='e1c23f2990c290b38a07c970750b6226'),
        types.DownloadableArtefact(
            url='https://drive.google.com/u/0/uc?id=1gNHdceC8tS1JLcb6sZGaT7w6zwwTkiXp&export=download&confirm=y',
            checksum='4578bcf9207ffa2ad9608976e8f4cf37'),
        types.DownloadableArtefact(
            url='https://drive.google.com/u/0/uc?id=14_xcxAYTsURyhBKS2FBNwqQFNGIkDbP7&export=download&confirm=y',
            checksum='d2d58d2df269ffa2f79d68f5942e4109'),
        types.DownloadableArtefact(
            url='https://drive.google.com/u/0/uc?id=1e8PvYV1Rbc3uDDKt0iADNNT6fXH95FIC&export=download&confirm=y',
            checksum='77fb7329500a70150d1b4637652720b9'),
        types.DownloadableArtefact(
            url='https://drive.google.com/u/0/uc?id=1xj-PJhD4xBtPv6EETGImlA0Pju7KdIH0&export=download&confirm=y',
            checksum='52c846632c2b903536c097e6ccd91c39'),
        types.DownloadableArtefact(
            url='https://drive.google.com/u/0/uc?id=1li1j1-315EmjXbuqRnMIxiH_u7Kpj81b&export=download&confirm=y',
            checksum='44293bc45b2a41fba17cf163c8a01c0a'),
        types.DownloadableArtefact(
            url='https://drive.google.com/u/0/uc?id=1OQnAweOPYbwhNt9uQ07aVV1GNmkTAcD2&export=download&confirm=y',
            checksum='9ea4965e0c61ad295437be4b6cf10681')
    ],
    website_url='https://dali-dl.github.io/project_iccv2017.html',
    handler=handler,
    paper_title='Deeper, Broader and Artier Domain Generalization',
    authors='Da Li, Yongxin Yang, Yi-Zhe Song, Timothy M. Hospedales',
    year=2017,
    fixture_writer=write_fixture)
