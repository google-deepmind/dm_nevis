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

"""Tests for dm_nevis.benchmarker.datasets.builders.test_dataset."""


from absl.testing import absltest
from absl.testing import parameterized
from dm_nevis.benchmarker.datasets.builders import test_dataset
import tensorflow_datasets as tfds


class TestDatasetTest(parameterized.TestCase):

  def test_get_test_dataset(self):

    with self.subTest('test_split'):
      dataset = test_dataset.get_dataset(split='test')
      self.assertEqual(dataset.num_examples,
                       test_dataset.SPLIT_DEFAULTS['test']['num_examples'])
      ds = dataset.builder_fn(shuffle=True, start=3, end=9)
      ds = ds.batch(batch_size=2)
      batches = list(tfds.as_numpy(ds))
      self.assertLen(batches, 3)

    with self.subTest('train_split'):
      dataset = test_dataset.get_dataset(split='train')
      ds = dataset.builder_fn(shuffle=False)
      examples = list(tfds.as_numpy(ds))
      self.assertLen(examples,
                     test_dataset.SPLIT_DEFAULTS['train']['num_examples'])

    with self.subTest('val_sub_split'):
      dataset = test_dataset.get_dataset(split='val', start=3, end=10)
      self.assertEqual(dataset.num_examples, 10 - 3)
      elements = list(tfds.as_numpy(dataset.builder_fn(shuffle=False)))
      self.assertLen(elements, dataset.num_examples)

      ds = dataset.builder_fn(shuffle=True, start=1, end=3)
      examples = list(tfds.as_numpy(ds))
      self.assertLen(examples, 2)


if __name__ == '__main__':
  absltest.main()
