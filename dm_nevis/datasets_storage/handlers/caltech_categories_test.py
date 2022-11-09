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

"""Tests for dm_nevis.datasets_storage.handlers.caltech_categories."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_nevis.datasets_storage.handlers import caltech_categories


class CaltechCategoriesTest(parameterized.TestCase):

  def test_caltech_categories_handler(self):
    fixture_path = self.create_tempdir().full_path
    caltech_categories.write_fixture(fixture_path)
    dataset = caltech_categories.caltech_categories_dataset
    metadata, generators = dataset.handler(fixture_path)

    self.assertEqual(metadata.num_classes, 2)
    samples = list()
    for split in ['train', 'dev', 'dev-test', 'test']:
      self.assertIn(split, generators)
      samples.extend(list(generators[split]))

    self.assertLen(samples, 5 * len(caltech_categories.AVAILABLE_CATEGORIES))

    self.assertTrue(all(img.size == (50, 50) for img, label in samples))

    seen_labels = set(label for img, label in samples)
    self.assertEqual(seen_labels, {0, 1})

    self.assertEqual(metadata.additional_metadata['class_names'],
                     ['car', 'motorcycle'])

  def test_caltech_categories_handler_cars_only(self):
    fixture_path = self.create_tempdir().full_path
    caltech_categories.write_fixture(fixture_path)
    dataset = caltech_categories.caltech_categories_dataset
    metadata, generators = dataset.handler(
        fixture_path, categories=['cars_2001'])

    self.assertEqual(metadata.num_classes, 1)

    samples = list(generators['train'])
    seen_labels = set(label for img, label in samples)
    self.assertEqual(seen_labels, {0})


if __name__ == '__main__':
  absltest.main()
