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

"""Ofxord flowers 17 handler."""

import os
import re
import tarfile
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
import numpy as np
from PIL import Image
import scipy.io

_IMAGE_FILE_NAME = '17flowers.tgz'
_SPLIT_FILE_NAME = 'datasplits.mat'
_NUM_EXAMPLES_PER_CLASS = 80


def oxford_flowers_17_handler(dataset_path: str) -> types.HandlerOutput:
  """Oxford flowers dataset with 17 classes."""
  split_data = scipy.io.loadmat(os.path.join(dataset_path, _SPLIT_FILE_NAME))

  train_ids = split_data['trn1'].flatten()
  val_ids = split_data['val1'].flatten()
  test_ids = split_data['tst1'].flatten()

  # Image file names are sorted such that every class has excatly 80 examples.
  id_to_label_fn = lambda x: (x - 1) // _NUM_EXAMPLES_PER_CLASS

  def gen(ids, id_to_label_fn):
    with tarfile.open(os.path.join(dataset_path, _IMAGE_FILE_NAME), 'r') as tf:
      for member in tf:
        if member.isdir() or 'image' not in member.path:
          continue
        idx = int(re.search(r'jpg/image_([\d]+).jpg', member.path)[1])
        if idx not in ids:
          continue
        image = Image.open(tf.extractfile(member))
        image.load()
        label = id_to_label_fn(idx)
        yield (image, label)

  metadata = types.DatasetMetaData(
      num_classes=17,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          task_type='classification',
          image_type='object',
      ))

  make_gen_fn = lambda: gen(train_ids, id_to_label_fn)
  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN_AND_DEV_ONLY,
      splits.MERGED_TRAIN_AND_DEV)
  per_split_gen['dev-test'] = gen(val_ids, id_to_label_fn)
  per_split_gen['test'] = gen(test_ids, id_to_label_fn)
  per_split_gen['all'] = gen(np.arange(start=1, stop=1360 + 1), id_to_label_fn)

  return metadata, per_split_gen


oxford_flowers_17_dataset = types.DownloadableDataset(
    name='oxford_flowers_17',
    download_urls=[
        types.DownloadableArtefact(
            url='https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz',
            checksum='b59a65d8d1a99cd66944d474e1289eab'),
        types.DownloadableArtefact(
            url='https://www.robots.ox.ac.uk/~vgg/data/flowers/17/datasplits.mat',
            checksum='4828cddfd0d803c5abbdebcb1e148a1e')
    ],
    website_url='https://www.robots.ox.ac.uk/~vgg/data/flowers/',
    handler=oxford_flowers_17_handler)
