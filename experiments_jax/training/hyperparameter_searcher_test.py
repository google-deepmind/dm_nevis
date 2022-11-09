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

"""Tests for hyperparameter_searcher."""

from concurrent import futures
from absl.testing import absltest
from absl.testing import parameterized
from dm_nevis.benchmarker.learners import learner_interface
from experiments_jax.training import hyperparameter_searcher

DEFAULT_NUM_WORKERS = 3


class HyperparameterSearcherTest(parameterized.TestCase):

  def test_hyperparameter_searcher(self):

    def cost_function_builder():

      def f(state, overrides, **kwargs):
        del state, overrides, kwargs
        return 0, None, learner_interface.ResourceUsage()

      return f

    workers = hyperparameter_searcher.build_local_executor_workers(
        cost_function_builder,
        num_workers=DEFAULT_NUM_WORKERS,
        executor=futures.ThreadPoolExecutor)

    _test_searcher(workers)


def _test_searcher(workers):
  searcher = hyperparameter_searcher.HyperparameterTuner(workers)

  # All config is passed through the search space. Fixed config is set
  # using a search space with a single value.
  search_space = _product([
      _sweep('task_key', [0]),
      _sweep('learning_rate', [1, 2, 3]),
  ])

  result = searcher.minimize(None, search_space)
  assert result.cost == 0

  workers.queue.close()


def _product(sweeps):
  dcts, *sweeps = sweeps

  for sweep_dcts in sweeps:
    new_dcts = []
    for sweep_dct in sweep_dcts:
      new_dcts.extend({**dct, **sweep_dct} for dct in dcts)
    dcts = new_dcts

  return dcts


def _sweep(key, values):
  return [{key: value} for value in values]


if __name__ == '__main__':
  absltest.main()
