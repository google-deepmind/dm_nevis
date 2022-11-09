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

"""Tests for experiments_jax.training.modules."""

from absl.testing import absltest
from absl.testing import parameterized
from experiments_jax.training import modules
import haiku as hk
import jax.numpy as jnp


class ModelsTest(parameterized.TestCase):

  def test_flatten_only(self):
    batch_size = 4
    height, width = 8, 8

    image = jnp.zeros([batch_size, height, width, 3])
    rng = hk.PRNGSequence(1)

    def forward_fn(image, is_training):
      model = modules.FlattenOnly()
      return model(image, is_training=is_training)

    forward_t = hk.transform_with_state(forward_fn)
    params, state = forward_t.init(next(rng), image, is_training=True)

    h_train, _ = forward_t.apply(
        params, state, next(rng), image, is_training=True)
    h_test, _ = forward_t.apply(
        params, state, next(rng), image, is_training=False)

    self.assertSequenceEqual(h_train.shape, [batch_size, height*width*3])
    self.assertSequenceEqual(h_test.shape, [batch_size, height*width*3])

  def test_mlp(self):
    batch_size = 4
    height, width = 8, 8

    image = jnp.zeros([batch_size, height, width, 3])
    rng = hk.PRNGSequence(1)

    def forward_fn(image, is_training):
      model = modules.MLP(output_sizes=[16, 16])
      return model(image, is_training=is_training)

    forward_t = hk.transform_with_state(forward_fn)
    params, state = forward_t.init(next(rng), image, is_training=True)

    h_train, _ = forward_t.apply(
        params, state, next(rng), image, is_training=True)
    h_test, _ = forward_t.apply(
        params, state, next(rng), image, is_training=False)

    self.assertSequenceEqual(h_train.shape, [batch_size, 16])
    self.assertSequenceEqual(h_test.shape, [batch_size, 16])

  def test_convnet(self):
    batch_size = 4
    height, width = 8, 8

    image = jnp.zeros([batch_size, height, width, 3])
    rng = hk.PRNGSequence(1)

    def forward_fn(image, is_training):
      model = modules.ConvNet()
      return model(image, is_training=is_training)

    forward_t = hk.transform_with_state(forward_fn)
    params, state = forward_t.init(next(rng), image, is_training=True)

    h_train, _ = forward_t.apply(
        params, state, next(rng), image, is_training=True)
    h_test, _ = forward_t.apply(
        params, state, next(rng), image, is_training=False)

    self.assertLen(h_train.shape, 2)
    self.assertLen(h_test.shape, 2)


if __name__ == '__main__':
  absltest.main()
