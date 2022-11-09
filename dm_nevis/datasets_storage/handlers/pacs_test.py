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

"""Tests for dm_nevis.datasets_storage.handlers.pacs."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_nevis.datasets_storage.handlers import pacs


class PacsTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(categories=pacs.CATEGORIES),
  ])
  def test_handler(self, categories):
    fixture_dir = self.create_tempdir().full_path
    dataset = pacs.pacs_dataset

    dataset.fixture_writer(fixture_dir)
    metadata, generator = dataset.handler(fixture_dir)
    self.assertLen(metadata.additional_metadata['categories'], len(categories))

    samples = list()
    for split in ['train', 'dev', 'dev-test', 'test']:
      self.assertIn(split, generator)
      samples.extend(list(generator[split]))

    # Each category and class has a single image in the fixture.
    self.assertLen(samples, len(categories) * metadata.num_classes)


if __name__ == '__main__':
  absltest.main()
