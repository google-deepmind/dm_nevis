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

"""USPS handler."""

import bz2
import os
from typing import Tuple
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
import numpy as np
from PIL import Image

_ORIG_IMG_RANGE = (-1., 1.)
_NEW_IMG_RANGE = (0., 255.)
_TRAIN_FNAME = 'usps.bz2'
_TEST_FNAME = 'usps.t.bz2'


def _rescale_image_back_to_pixel_range(
    img: np.ndarray, orig_img_range: Tuple[float, float],
    new_img_range: Tuple[float, float]) -> np.ndarray:
  """Rescales image from original image range to a new one."""
  assert len(orig_img_range) == 2
  assert len(new_img_range) == 2
  delta_orig_range = orig_img_range[1] - orig_img_range[0]
  delta_new_range = new_img_range[1] - new_img_range[0]
  # [0, 1]]
  normalied_img = (img - orig_img_range[0]) / delta_orig_range
  # [0, 255]
  new_range_img = normalied_img * delta_new_range + new_img_range[0]
  return new_range_img


def _parse_label_and_image_from_line(line: str):
  """Parses label and image from a parsed line."""
  label, *img_data = line.split(' ')
  label = int(label)
  # `img_data` contains pairs x:y, where x is a pixel number and y is the value.
  image_array = np.array([float(el.split(':')[1]) for el in img_data[:-1]])
  image_array = _rescale_image_back_to_pixel_range(
      image_array, _ORIG_IMG_RANGE, _NEW_IMG_RANGE).astype(np.uint8)
  label = label - 1
  return image_array, label


def usps_handler(dataset_path: str) -> types.HandlerOutput:
  """USPS handler."""

  def make_gen(split):
    fname = _TRAIN_FNAME if split == 'train' else _TEST_FNAME
    with bz2.BZ2File(os.path.join(dataset_path, fname)) as bzf:
      for line in bzf:
        if not line:
          continue
        image_array, label = _parse_label_and_image_from_line(
            line.decode('utf-8'))
        image = Image.fromarray(image_array.reshape((16, 16)))
        yield (image, label)

  metadata = types.DatasetMetaData(
      num_classes=10,
      num_channels=1,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          task_type='classification',
          image_type='ocr',
      ))

  make_gen_fn = lambda: make_gen('train')
  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN,
      splits.MERGED_TRAIN_AND_DEV)
  per_split_gen['test'] = make_gen('test')

  return metadata, per_split_gen


usps_dataset = types.DownloadableDataset(
    name='usps',
    download_urls=[
        types.DownloadableArtefact(
            url='https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2',
            checksum='ec16c51db3855ca6c91edd34d0e9b197'),
        types.DownloadableArtefact(
            url='https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2',
            checksum='8ea070ee2aca1ac39742fdd1ef5ed118')
    ],
    website_url='https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps',
    handler=usps_handler)
