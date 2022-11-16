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

"""Integration tests for stable datasets in Nevis dataset storage.

The purpose of these tests is to guarantee that a dataset specified in
`DATASETS_TO_RUN_INTEGRATION_TESTS_FOR` is available in the `stable` version
(see README.md) and have specified properties:
* A dataset can be read as TFRecordDataset.
* A dataset contains metadata.
* Each image in the dataset has `image_shape` and `num_channels` as specified in
metadata.
* Each pixel of the image is in range [0,255].
* Labels are in [0,num_classes - 1].
* Each class has at least one data point.
"""

import collections
import os
from absl.testing import absltest
from absl.testing import parameterized
from dm_nevis.datasets_storage import dataset_loader
from dm_nevis.datasets_storage import download_util as du
from dm_nevis.datasets_storage import version_control as vc
import tensorflow as tf

DATASETS_TO_RUN_INTEGRATION_TESTS_FOR = [
    'animal',
]


@absltest.skipThisClass('Timeout for OSS')
class StableDatasetsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests on the `stable` version of Nevis datasets.

  These tests go over all the specified datasets in
  `DATASETS_TO_RUN_INTEGRATION_TESTS_FOR` in the `stable` dataset directory
  and ensure that these datasets satisfy given (in the tests) constraints.
  That will ensure that the datasets which are actually used in the benchmark
  produce expected data.
  """

  @parameterized.parameters(sorted(DATASETS_TO_RUN_INTEGRATION_TESTS_FOR))
  def test_datasets_available(self, dataset_name):
    data_dir = vc.get_nevis_stable_data_dir()
    dataset_path = os.path.join(data_dir, dataset_name)
    status = du.try_read_status(dataset_path)
    self.assertEqual(status, du.DatasetDownloadStatus.READY)

  @parameterized.parameters(sorted(DATASETS_TO_RUN_INTEGRATION_TESTS_FOR))
  def test_dataset_splits_metadata_not_empty(self, dataset_name):
    metadata = dataset_loader.get_metadata(dataset_name)
    splits = metadata.additional_metadata['splits']
    self.assertNotEmpty(splits)

  @parameterized.parameters(sorted(DATASETS_TO_RUN_INTEGRATION_TESTS_FOR))
  def test_datasets_each_class_has_at_least_one_element(self, dataset_name):
    metadata = dataset_loader.get_metadata(dataset_name)
    num_classes = metadata.num_classes
    splits = metadata.additional_metadata['splits']
    per_class_num_points = collections.defaultdict(int)
    for split in splits:
      ds = dataset_loader.load_dataset(dataset_name, split)
      ds = ds.builder_fn().as_numpy_iterator()
      for elem in ds:
        label = elem.label
        per_class_num_points[label] += 1
    self.assertLen(per_class_num_points, num_classes)

  @parameterized.parameters(sorted(DATASETS_TO_RUN_INTEGRATION_TESTS_FOR))
  def test_datasets_elements_have_expected_shapes(self, dataset_name):
    metadata = dataset_loader.get_metadata(dataset_name)
    num_channels = metadata.num_channels
    num_classes = metadata.num_classes
    image_shape = metadata.image_shape
    splits = metadata.additional_metadata['splits']
    for split in splits:
      ds = dataset_loader.load_dataset(dataset_name, split)
      ds = ds.builder_fn().as_numpy_iterator()
      for elem in ds:
        label = elem.label
        self.assertEqual(elem.image.shape, image_shape + (num_channels,))
        self.assertAllInRange(label, 0, num_classes - 1)
        # TODO: Make sure that the test fails when image is in (0,1).
        self.assertAllInRange(elem.image, 0, 255)


if __name__ == '__main__':
  tf.test.main()
