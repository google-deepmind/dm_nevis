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

"""Path MNIST handler.

Colorectal Histology MNIST dataset.

"""

import os
import zipfile
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
import numpy as np
import pandas as pd
from PIL import Image

_HMNIST_FNAME = 'hmnist_28_28_RGB.csv'
_ARCHIVE_FNAME = 'colorectal-histology-mnist.zip'


def _path_to_label_fn(path: str, label_to_id):
  label = os.path.split(path)[1].split('_')[0]
  return label_to_id[label]


def path_mnist_handler(dataset_path: str) -> types.HandlerOutput:
  """Colorectal Histology MNIST handler."""
  datafile = os.path.join(dataset_path, _ARCHIVE_FNAME)

  with zipfile.ZipFile(os.path.join(dataset_path, datafile), 'r') as zf:
    data = pd.read_csv(zf.open(_HMNIST_FNAME))

  def gen():
    for _, row in data.iterrows():

      img = np.array([row[f'pixel{i:04d}'] for i in range(28 * 28 * 3)
                     ]).reshape((28, 28, 3))

      label = row['label'] - 1
      img = Image.fromarray(img.astype('uint8'))
      yield img, label

  metadata = types.DatasetMetaData(
      num_classes=8,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          task_type='classification',
          image_type='medical',
      ))

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      gen, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)

  return metadata, per_split_gen


path_mnist_dataset = types.DownloadableDataset(
    name='path_mnist',
    download_urls=[
        types.KaggleDataset(
            dataset_name='kmader/colorectal-histology-mnist',
            checksum='e03501016bd54719567dfb954fe982fe')
    ],
    website_url='https://www.kaggle.com/kmader/colorectal-histology-mnist',
    handler=path_mnist_handler)
