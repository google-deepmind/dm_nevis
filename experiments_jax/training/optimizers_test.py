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

"""Tests for experiments_jax.training.optimizers."""

from absl.testing import absltest
from absl.testing import parameterized
from experiments_jax.training import optimizers
import haiku as hk
import jax
import jax.numpy as jnp


class OptimizersTest(parameterized.TestCase):

  def test_default_weight_decay_mask(self):

    @hk.transform_with_state
    def f(x, is_training):
      x = hk.Linear(10, name='layer_1')(x)
      x = hk.BatchNorm(
          create_scale=True,
          create_offset=True,
          decay_rate=1e-3,
      )(x, is_training=is_training)
      x = hk.Linear(10, name='layer_2')(x)
      return x

    params, _ = f.init(jax.random.PRNGKey(0), jnp.zeros([2, 10]), True)
    masked_params = optimizers.default_weight_decay_mask(params)

    self.assertEqual(
        masked_params,
        {
            'batch_norm': {
                'offset': False,
                'scale': False
            },
            'layer_1': {
                'b': False,
                'w': True
            },
            'layer_2': {
                'b': False,
                'w': True
            }
        },
    )


if __name__ == '__main__':
  absltest.main()
