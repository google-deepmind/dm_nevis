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

"""Tests for dm_nevis.datasets_storage.handlers.splits."""

import collections
from absl.testing import absltest
from absl.testing import parameterized
from dm_nevis.datasets_storage.handlers import splits

_TEST_CASES = [
    dict(
        num_examples=10000,
        splits_with_fractions=dict(train=0.7, valid=0.1, test=0.2)),
    dict(
        num_examples=10000,
        splits_with_fractions=dict(train=0.8, valid=0.1, test=0.1)),
    dict(
        num_examples=100000,
        splits_with_fractions=dict(train=0.4, valid=0.25, test=0.35)),
]


class SplitsTest(parameterized.TestCase):

  @parameterized.parameters(_TEST_CASES)
  def test_random_split_generator_into_splits_with_fractions(
      self, num_examples, splits_with_fractions):

    def make_gen_fn():
      yield from range(num_examples)

    per_split_gen = splits.random_split_generator_into_splits_with_fractions(
        make_gen_fn, splits_with_fractions)

    split_names = splits_with_fractions.keys()

    per_split_elems = collections.defaultdict(set)
    for split_name in split_names:
      split_gen = per_split_gen[split_name]
      for elem in split_gen:
        per_split_elems[split_name].add(elem)

      fraction = len(per_split_elems[split_name]) / num_examples

      # Check that the fractions are close to the initial ones.
      self.assertAlmostEqual(
          fraction, splits_with_fractions[split_name], places=2)

    # Check that sum of the elements is equal to num_examples
    self.assertEqual(num_examples,
                     sum([len(elems) for elems in per_split_elems.values()]))

    # Check that different elements are disjoint
    split_names = sorted(splits_with_fractions.keys())
    for split_name_a in split_names:
      for split_name_b in split_names:
        if split_name_a == split_name_b:
          continue
        self.assertEmpty(per_split_elems[split_name_a].intersection(
            per_split_elems[split_name_b]))

  @parameterized.parameters(_TEST_CASES)
  def test_random_split_generator_into_splits_with_fractions_and_merged(
      self, num_examples, splits_with_fractions):

    def make_gen_fn():
      yield from range(num_examples)

    per_split_gen = splits.random_split_generator_into_splits_with_fractions(
        make_gen_fn, splits_with_fractions,
        {'train_and_valid': ('train', 'valid')})
    per_split_elems = collections.defaultdict(set)
    for split_name, split_gen in per_split_gen.items():
      for elem in split_gen:
        per_split_elems[split_name].add(elem)

    self.assertSetEqual(
        per_split_elems['train_and_valid'],
        per_split_elems['train'].union(per_split_elems['valid']))

    self.assertLen(
        per_split_elems['train_and_valid'],
        len(per_split_elems['train']) + len(per_split_elems['valid']))


if __name__ == '__main__':
  absltest.main()
