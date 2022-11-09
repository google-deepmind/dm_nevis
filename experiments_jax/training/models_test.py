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

"""Tests for experiments_jax.training.models."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
from dm_nevis.benchmarker.datasets import tasks
from experiments_jax.training import models
from experiments_jax.training import modules
import haiku as hk
import jax.numpy as jnp

SUPPORTED_TASKS = frozenset({
    tasks.TaskKey("task1", tasks.TaskKind.CLASSIFICATION,
                  tasks.ClassificationMetadata(num_classes=10)),
    tasks.TaskKey("task2", tasks.TaskKind.CLASSIFICATION,
                  tasks.ClassificationMetadata(num_classes=20)),
})


class ModelsTest(parameterized.TestCase):

  def test_modelbuilder(self):
    batch_size = 2
    image_size = 4
    model = models.build_model(
        functools.partial(modules.MLP, output_sizes=[16]),
        supported_tasks=SUPPORTED_TASKS,
        image_resolution=image_size)

    prng = hk.PRNGSequence(0)
    params, state = model.init(next(prng))

    num_expected_modules = 1 + len(SUPPORTED_TASKS)

    self.assertLen(params, num_expected_modules)

    images = jnp.zeros([batch_size, image_size, image_size, 3])
    labels = jnp.zeros([batch_size], dtype=jnp.int32)
    for task in SUPPORTED_TASKS:
      _, _ = model.loss_and_metrics[task](params, state, next(prng), images,
                                          labels, True)

      _ = model.predict[task](params, state, next(prng), images, True)


if __name__ == "__main__":
  absltest.main()
