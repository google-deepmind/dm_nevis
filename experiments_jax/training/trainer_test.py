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

"""Tests for experiments_jax.training.trainer."""

import functools

from typing import Set
from absl.testing import absltest
from absl.testing import parameterized
from dm_nevis.benchmarker.datasets import datasets
from dm_nevis.benchmarker.datasets import tasks
from experiments_jax.training import models
from experiments_jax.training import modules
from experiments_jax.training import trainer
import haiku as hk
import jax.numpy as jnp
import optax


SUPPORTED_TASKS = [
    tasks.TaskKey("task1", tasks.TaskKind.CLASSIFICATION,
                  tasks.ClassificationMetadata(num_classes=10)),
    tasks.TaskKey("task2", tasks.TaskKind.CLASSIFICATION,
                  tasks.ClassificationMetadata(num_classes=20)),
]


def modules_in_params(params: hk.Params) -> Set[str]:
  return {m for m, _, _ in hk.data_structures.traverse(params)}


class TrainerTest(parameterized.TestCase):

  def test_no_frozen_parameters(self):
    batch_size = 2
    image_size = 4
    prng = hk.PRNGSequence(0)

    model = models.build_model(
        functools.partial(modules.MLP, output_sizes=[16]),
        supported_tasks=SUPPORTED_TASKS,
        image_resolution=image_size)
    optimizer = optax.sgd(0.1)
    train_state = trainer.init_train_state(next(prng), model, optimizer)

    expected_modules = {
        "backbone/~/mlp/~/linear_0",
        "task1_head/~/linear",
        "task2_head/~/linear"}
    self.assertSetEqual(
        modules_in_params(train_state.trainable_params), expected_modules)
    self.assertSetEqual(
        modules_in_params(train_state.frozen_params), set())

    update_fn = trainer.build_update_fn(SUPPORTED_TASKS[0], model, optimizer)

    # Fake Data
    batch = datasets.MiniBatch(
        image=jnp.zeros([batch_size, image_size, image_size, 3]),
        label=jnp.zeros([batch_size]), multi_label_one_hot=None)
    train_state, _ = update_fn(batch, train_state)

    self.assertSetEqual(
        modules_in_params(train_state.trainable_params), expected_modules)
    self.assertSetEqual(
        modules_in_params(train_state.frozen_params), set())

  def test_frozen_parameters(self):
    batch_size = 2
    image_size = 4
    prng = hk.PRNGSequence(0)

    model = models.build_model(
        functools.partial(modules.MLP, output_sizes=[16], name="mlp"),
        supported_tasks=SUPPORTED_TASKS,
        image_resolution=image_size)
    optimizer = optax.sgd(0.1)

    def load_params_fn(params, state):
      # Consider parameters in the backbone frozen, heads are traiable:
      train_params, frozen_params = hk.data_structures.partition(
          lambda module_name, _1, _2: not module_name.startswith("backbone"),
          params)
      return train_params, frozen_params, state

    train_state = trainer.init_train_state(
        next(prng), model, optimizer, load_params_fn)
    self.assertSetEqual(
        modules_in_params(train_state.trainable_params),
        {"task1_head/~/linear", "task2_head/~/linear"})
    self.assertSetEqual(
        modules_in_params(train_state.frozen_params),
        {"backbone/~/mlp/~/linear_0"})

    update_fn = trainer.build_update_fn(SUPPORTED_TASKS[0], model, optimizer)

    # Fake Data
    batch = datasets.MiniBatch(
        image=jnp.zeros([batch_size, image_size, image_size, 3]),
        label=jnp.zeros([batch_size]), multi_label_one_hot=None)
    train_state, _ = update_fn(batch, train_state)
    self.assertSetEqual(
        modules_in_params(train_state.trainable_params),
        {"task1_head/~/linear", "task2_head/~/linear"})
    self.assertSetEqual(
        modules_in_params(train_state.frozen_params),
        {"backbone/~/mlp/~/linear_0"})


if __name__ == "__main__":
  absltest.main()
