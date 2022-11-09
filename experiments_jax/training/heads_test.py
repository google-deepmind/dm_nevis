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

"""Tests for experiments_jax.training.heads."""

from absl.testing import absltest
from absl.testing import parameterized
from experiments_jax.training import heads
import haiku as hk
import jax
import numpy as np


class HeadsTest(parameterized.TestCase):

  def test_metrics_multi_label(self):
    num_labels = 17
    num_examples = 29

    @hk.transform_with_state
    def f(inputs, targets):
      head = heads.MultiLabelHead(num_classes=num_labels)
      return head.loss_and_metrics(inputs, targets, is_training=False)

    gen = np.random.default_rng(0)
    inputs = gen.normal(size=(num_examples, num_labels))
    targets = np.ones((num_examples, num_labels))

    params, state = f.init(jax.random.PRNGKey(0), inputs, targets)
    (_, diagnostics), _ = f.apply(params, state, None, inputs, targets)
    error = diagnostics['error']

    self.assertLessEqual(np.max(error), 1.0)
    self.assertGreaterEqual(np.min(error), 0.0)

    # We have p=0.5 chance of getting each prediction correct, summed over
    # a number of iid trials.
    expected_value = 0.5
    self.assertAlmostEqual(expected_value, error.mean(), delta=0.1)


if __name__ == '__main__':
  absltest.main()
