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

"""Tests for dm_nevis.datasets_storage.dataset_loader."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_nevis.datasets_storage import dataset_loader
from dm_nevis.datasets_storage import download_util
from dm_nevis.datasets_storage import handlers
from dm_nevis.datasets_storage import preprocessing
from dm_nevis.datasets_storage.handlers import types

DEFAULT_NUM_SHARDS = 5
DEFAULT_BUFFER_SIZE = 100


class DatasetLoaderTest(parameterized.TestCase):

  # We can include any dataset that has a fixture writer.
  @parameterized.parameters([
      # dict(dataset_name='belgium_tsc'),
      # dict(dataset_name='pacs'),
      dict(dataset_name='caltech_categories'),
      # dict(dataset_name='flickr_material_database'),
  ])
  def test_load_dataset_from_path(self, dataset_name: str):
    artifacts_path = self.create_tempdir().full_path
    dataset_path = self.create_tempdir().full_path

    downloadable_dataset = handlers.get_dataset(dataset_name)
    self.assertIsNotNone(downloadable_dataset.fixture_writer)

    _extract_dataset(downloadable_dataset, artifacts_path, dataset_path)

    metadata = dataset_loader.get_metadata_from_path(dataset_path)
    self.assertNotEmpty(metadata.additional_metadata['splits'])

    for split in metadata.additional_metadata['splits']:
      with self.subTest(split):
        dataset = dataset_loader.load_dataset_from_path(dataset_path, split)
        examples = list(dataset.builder_fn(shuffle=False))
        self.assertNotEmpty(examples)
        self.assertLen(examples, dataset.num_examples)
        examples = list(dataset.builder_fn(shuffle=True))
        self.assertNotEmpty(examples)
        self.assertLen(examples, dataset.num_examples)
        examples = list(dataset.builder_fn(shuffle=False, start=2))
        self.assertLen(examples, dataset.num_examples - 2)


def _extract_dataset(downloadable_dataset: types.DownloadableDataset,
                     artifacts_path: str, output_path: str) -> None:
  """Writes a fixture and then extracts a dataset from the fixture."""
  downloadable_dataset.fixture_writer(artifacts_path)
  download_util.extract_dataset_and_update_status(
      downloadable_dataset,
      artifacts_path,
      output_path,
      DEFAULT_NUM_SHARDS,
      DEFAULT_BUFFER_SIZE,
      compute_statistics=True,
      preprocess_image_fn=preprocessing.preprocess_image_fn,
      preprocess_metadata_fn=preprocessing.preprocess_metadata_fn)


if __name__ == '__main__':
  absltest.main()
