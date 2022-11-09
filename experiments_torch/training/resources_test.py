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
"""Tests for experiments_torch.training.resources."""

from absl.testing import absltest
from absl.testing import parameterized
from experiments_torch.training import resources
import torch
from torch import nn


@absltest.skipThisClass('Need PyTorch >= 1.13')
class ResourcesTest(parameterized.TestCase):

  def test_linear(self):
    module = nn.Linear(10, 1)
    x = torch.ones((2, 10))

    def _function():
      loss = module(x).mean()
      loss.backward()

    flops = resources.estimate_flops(module, _function)

    self.assertEqual(flops, 80)


if __name__ == '__main__':
  absltest.main()
