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

"""Tests for dm_nevis.streams.nevis."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_nevis.benchmarker.datasets import datasets
from dm_nevis.streams import nevis
import numpy as np
import tensorflow as tf


class NevisTest(parameterized.TestCase):

  @parameterized.parameters([
      {
          'labels': [
              [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.],
              [0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.],
              [0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
              [0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
              [1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.],
              [0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.],
              [0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
              [0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.],
              [0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
              [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.],
              [0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
              [1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
              [0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
              [0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.],
              [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.],
              [0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
          ],
          'expected': [
              [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.],
              [0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.],
              [0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
              [0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
              [1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.],
              [0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.],
              [0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
              [0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.],
              [0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
              [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.],
              [0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
              [1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
              [0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
              [0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.],
              [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.],
              [0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
          ]
      },
      {
          'labels': [
              [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0.],
              [0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
              [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
          ],
          'expected': [
              [1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.],
              [0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
              [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.],
          ]
      },
  ])
  def test_patch_biwi_minibatch(self, labels, expected):

    for label, expected in zip(labels, expected):
      example = datasets.MiniBatch(
          multi_label_one_hot=tf.constant(label), image=None, label=None)

      result = nevis._patch_biwi_minibatch(example)
      np.testing.assert_allclose(result.multi_label_one_hot.numpy(),
                                 np.array(expected))


if __name__ == '__main__':
  absltest.main()
