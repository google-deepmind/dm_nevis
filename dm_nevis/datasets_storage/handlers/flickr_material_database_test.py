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

"""Tests for dm_nevis.datasets_storage.handlers.flickr_material_database."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_nevis.datasets_storage.handlers import flickr_material_database


class FlickrMaterialDatabaseTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(path='image/wood/wood_object_021_new.jpg', label=9),
      dict(path='image/water/water_moderate_023_new.jpg', label=8),
  ])
  def test_path_to_label(self, path, label):
    result = flickr_material_database._path_to_label(path)
    self.assertEqual(result, label)

  def test_handler(self):
    fixture_path = self.create_tempdir().full_path
    flickr_material_database.write_fixture(fixture_path)

    dataset = flickr_material_database.flickr_material_database_dataset
    metadata, generators = dataset.handler(fixture_path)

    self.assertEqual(metadata.num_classes, 10)

    samples = list()
    for split in ['train', 'dev', 'dev-test', 'test']:
      self.assertIn(split, generators)
      samples.extend(list(generators[split]))

    self.assertLen(samples, 11)

    self.assertTrue(
        all(img.size == metadata.image_shape for img, label in samples))


if __name__ == '__main__':
  absltest.main()
