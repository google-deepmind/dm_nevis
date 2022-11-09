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

"""Tests for dm_nevis.datasets_storage.handlers.extraction_utils."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_nevis.datasets_storage.handlers import extraction_utils
import numpy as np
from PIL import Image

_IMG_VALUES_A = np.array([1 for _ in range(4)], dtype=np.uint8)
_IMG_VALUES_B = np.array([2 for _ in range(4)], dtype=np.uint8)


class ExtractionUtilsTest(parameterized.TestCase):

  def test_deduplicate_data_generator(self):
    duplicated_image_values = (
        (Image.fromarray(_IMG_VALUES_A), 1),
        (Image.fromarray(_IMG_VALUES_A), 1),
        (Image.fromarray(_IMG_VALUES_A), 1),
        (Image.fromarray(_IMG_VALUES_B), 2),
        (Image.fromarray(_IMG_VALUES_B), 2),
    )

    expected_image_values = (
        (Image.fromarray(_IMG_VALUES_A), 1),
        (Image.fromarray(_IMG_VALUES_B), 2),
    )

    unique_gen = extraction_utils.deduplicate_data_generator(
        duplicated_image_values)()
    unique_values = tuple(unique_gen)

    self.assertLen(unique_values, 2)
    self.assertEqual(unique_values, expected_image_values)

  @parameterized.parameters([
      (16, 16, 16, 16, 16),
      (512, 16, 16, 16, 16),
      (256, 869, 1079, 206, 256),
      (256, 1079, 869, 256, 206),
  ])
  def test_resize_to_max_size(self, max_size, width, height, expected_width,
                              expected_height):
    image = Image.new('RGB', (width, height))
    result_image = extraction_utils.resize_to_max_size(image, max_size)
    self.assertEqual(result_image.size, (expected_width, expected_height))


if __name__ == '__main__':
  absltest.main()
