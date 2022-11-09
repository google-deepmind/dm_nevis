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

"""MNIST Rotation handler."""

import gzip
import os
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
import numpy as np
from PIL import Image
import scipy.ndimage


_TRAIN_IMAGES_FILE = 'train-images-idx3-ubyte.gz'
_TRAIN_LABELS_FILE = 'train-labels-idx1-ubyte.gz'
_TEST_IMAGES_FILE = 't10k-images-idx3-ubyte.gz'
_TEST_LABELS_FILE = 't10k-labels-idx1-ubyte.gz'
_NUM_CLASSES = 10
_SEED = 99_999


# pylint:disable=missing-function-docstring
def mnist_rotation_handler(dataset_path: str) -> types.HandlerOutput:

  np.random.seed(_SEED)

  metadata = types.DatasetMetaData(
      num_classes=_NUM_CLASSES,
      num_channels=1,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          task_type='classification',
          image_type='ocr'))

  def gen_fn(images_file, labels_file, rotate=True):
    images_path = os.path.join(dataset_path, images_file)
    labels_path = os.path.join(dataset_path, labels_file)
    with gzip.open(images_path, 'rb') as f:
      images = np.frombuffer(f.read(), np.uint8, offset=16)
      images = images.reshape((-1, 28, 28))
    with gzip.open(labels_path, 'rb') as f:
      labels = np.frombuffer(f.read(), np.uint8, offset=8)
    for np_image, label in zip(images, labels):
      if rotate:
        np_image = scipy.ndimage.rotate(np_image,
                                        np.random.randint(0, high=360),
                                        reshape=False)
      image = Image.fromarray(np_image)
      yield (image, label)

  gen_tr = lambda: gen_fn(_TRAIN_IMAGES_FILE, _TRAIN_LABELS_FILE)

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      gen_tr, splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN,
      splits.MERGED_TRAIN_AND_DEV)
  per_split_gen['test'] = gen_fn(
      _TEST_IMAGES_FILE, _TEST_LABELS_FILE, rotate=False)

  return metadata, per_split_gen


mnist_rotation_dataset = types.DownloadableDataset(
    name='mnist_rotation',
    download_urls=[
        types.DownloadableArtefact(
            url='http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            checksum='f68b3c2dcbeaaa9fbdd348bbdeb94873'),
        types.DownloadableArtefact(
            url='http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            checksum='d53e105ee54ea40749a09fcbcd1e9432'),
        types.DownloadableArtefact(
            url='http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            checksum='9fb629c4189551a2d022fa330f9573f3'),
        types.DownloadableArtefact(
            url='http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
            checksum='ec29112dd5afa0611ce80d1b7f02629c')
    ],
    paper_title='SO(2)-equivariance in Neural networks using Fourier nonlinearity',
    authors='Muthuvel Murugan and K. V. Subrahmanyam',
    year='2019',
    handler=mnist_rotation_handler,
)
