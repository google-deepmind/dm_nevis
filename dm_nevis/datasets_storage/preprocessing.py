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

"""Preprocessing functions for data extraction."""

from typing import Any, Optional
from dm_nevis.datasets_storage.handlers import types
import numpy as np

STANDARD_IMAGE_SIZE = (64, 64)


def preprocess_image_fn(image: types.Image,
                        preprocessing: str,
                        rng: Optional[Any] = None,
                        seed: int = 0) -> types.Image:
  """Image preprocessing function."""

  if not preprocessing:
    pass
  elif preprocessing == 'random_crop':
    image = _random_crop_image(image, rng=rng, seed=seed)
  else:
    raise ValueError('Preprocessing: %s not supported' % preprocessing)
  return image


def preprocess_metadata_fn(
    metadata: types.DatasetMetaData) -> types.DatasetMetaData:
  return metadata


def _random_crop_image(image, rng=None, seed=0):
  """Randomly crops the image to standard size."""
  # TODO: Consider cropping to a larger size to homgenize resizing step
  # across all datasets once we move it to the learner.
  width, height = image.size
  assert width - STANDARD_IMAGE_SIZE[0] > 0
  assert height - STANDARD_IMAGE_SIZE[1] > 0
  if rng is None:
    rng = np.random.default_rng(seed=seed)
  left = rng.integers(0, width - STANDARD_IMAGE_SIZE[0])
  top = rng.integers(0, height - STANDARD_IMAGE_SIZE[1])
  # Crops to STANDARD_IMAGE_SIZE[0]xSTANDARD_IMAGE_SIZE[1]
  return image.crop(
      (left, top, left + STANDARD_IMAGE_SIZE[0], top + STANDARD_IMAGE_SIZE[1]))
