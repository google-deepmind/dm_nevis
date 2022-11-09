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

"""Tests for dm_nevis.datasets_storage.handlers.belgium_tsc."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_nevis.datasets_storage.handlers import belgium_tsc


class BelgiumTscTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(path='Training/00061/01957_00002.ppm', expected=61),
      dict(path='Test/00009/01957_00002.ppm', expected=9),
  ])
  def test_label_from_path(self, path, expected):
    result = belgium_tsc._label_from_path(path)
    self.assertEqual(result, expected)

  def test_belgium_tsc_handler(self):
    artifact_dir = self.create_tempdir().full_path
    belgium_tsc.write_fixture(artifact_dir)

    metadata, generators = belgium_tsc.belgium_tsc_handler(artifact_dir)
    self.assertLen(metadata.additional_metadata['class_names'], 62)

    self.assertEqual(metadata.num_classes, 62)
    for split in ['train', 'dev', 'dev-test', 'test']:
      self.assertIn(split, generators)

    test_examples = list(generators['test'])
    self.assertLen(test_examples, 4)

    samples = list()
    for split in ['train', 'dev', 'dev-test']:
      samples.extend(generators[split])
    self.assertLen(samples, 14)


if __name__ == '__main__':
  absltest.main()
