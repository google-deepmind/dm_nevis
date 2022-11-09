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

"""Tests for experiments_torch.training.heads."""

from absl.testing import absltest
from absl.testing import parameterized
from experiments_torch.training import heads
import numpy as np
import torch


class HeadsTest(parameterized.TestCase):

  def test_metrics_multi_label(self):
    num_labels = 17
    num_examples = 100

    head = heads.MultiLabelHead(features_dim=num_labels, num_classes=num_labels)

    inputs = np.random.normal(size=(num_examples, num_labels))
    targets = np.ones((num_examples, num_labels))

    with torch.no_grad():
      _, diagnostics = head.loss_and_metrics(
          torch.tensor(inputs).float(),
          torch.tensor(targets).float(),
          is_training=False)
    error = diagnostics['error']

    self.assertLen(error.shape, 2)
    self.assertEqual(error.shape[0], num_examples)
    self.assertEqual(error.shape[1], num_labels)
    self.assertLessEqual(np.max(error), 1.0)
    self.assertGreaterEqual(np.min(error), 0.0)

    # We have p=0.5 chance of getting each prediction correct, summed over
    # a number of iid trials.
    expected_value = 0.5
    self.assertAlmostEqual(expected_value, error.mean(), delta=0.1)

    diagnostics = {k: np.mean(v) for k, v in diagnostics.items()}
    for v in diagnostics.values():
      self.assertGreaterEqual(v, 0.)


if __name__ == '__main__':
  absltest.main()
