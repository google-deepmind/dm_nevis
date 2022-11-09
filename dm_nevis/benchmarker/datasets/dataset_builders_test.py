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

"""Tests for dm_nevis.benchmarker.datasets.dataset_builders."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_nevis.benchmarker.datasets import dataset_builders


class DatasetBuildersTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(
          outer_start=0,
          inner_start=0,
          outer_end=0,
          inner_end=0,
          expected=(0, 0)),
      dict(
          outer_start=None,
          inner_start=None,
          outer_end=0,
          inner_end=0,
          expected=(None, 0)),
      dict(
          outer_start=None,
          inner_start=None,
          outer_end=None,
          inner_end=None,
          expected=(None, None)),
      dict(
          outer_start=5,
          inner_start=5,
          outer_end=20,
          inner_end=10,
          expected=(10, 15)),
      dict(
          outer_start=3,
          inner_start=5,
          outer_end=30,
          inner_end=10,
          expected=(8, 13)),
      dict(
          outer_start=3,
          inner_start=5,
          outer_end=12,
          inner_end=10,
          expected=(8, 12)),
      dict(
          outer_start=3,
          inner_start=5,
          outer_end=6,
          inner_end=10,
          expected=(6, 6))
  ])
  def test_combine_indices(self, outer_start, inner_start, outer_end, inner_end,
                           expected):

    start, end = dataset_builders.combine_indices(
        outer_start=outer_start,
        inner_start=inner_start,
        outer_end=outer_end,
        inner_end=inner_end)

    self.assertEqual(start, expected[0])
    self.assertEqual(end, expected[1])


if __name__ == '__main__':
  absltest.main()
