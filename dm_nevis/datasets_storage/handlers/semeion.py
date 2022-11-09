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

"""Semeion handler."""

import os
from typing import Tuple
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
import numpy as np
from PIL import Image

from tensorflow.io import gfile


_SEMION_FNAME = 'semeion.data'
_NUM_ATTRIBUTES = 256
_IMG_SHAPE = (16, 16)
_NUM_CLASSES = 10


def semeion_handler(dataset_path: str) -> types.HandlerOutput:
  """Semeion dataset handler."""
  metadata = types.DatasetMetaData(
      num_classes=10,
      num_channels=1,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          task_type='classification',
          image_type='ocr',
      ))

  def make_gen_fn():
    with gfile.GFile(os.path.join(dataset_path, _SEMION_FNAME), 'r') as f:
      for line in f:
        image, label = _parse_image_and_label(line)
        yield (image, label)

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)

  return metadata, per_split_gen


def _parse_image_and_label(line: str) -> Tuple[Image.Image, int]:
  """Parses an image and a label from the line."""
  # The data in the line corresponds to 256 binary (float) pixel values followed
  # by one-hot encoding of the label which is represented by a binary vector of
  # size 10.
  unparsed_line = line.strip().split(' ')
  image_data = []

  for i in range(_NUM_ATTRIBUTES):
    image_data.append(int(float(unparsed_line[i])))
  image_array = np.reshape(np.array(image_data), _IMG_SHAPE).astype(np.uint8)
  # Original array is in range [0,1]
  image_array *= 255
  image = Image.fromarray(image_array)
  labels = []
  for i in range(_NUM_CLASSES):
    labels.append(int(unparsed_line[i + _NUM_ATTRIBUTES]))
  label = np.argmax(labels)
  return (image, label)


semeion_dataset = types.DownloadableDataset(
    name='semeion',
    download_urls=[
        types.DownloadableArtefact(
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data',
            checksum='cb545d371d2ce14ec121470795a77432')
    ],
    website_url='https://archive.ics.uci.edu/ml/datasets/semeion+handwritten+digit',
    handler=semeion_handler)
