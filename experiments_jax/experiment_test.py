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

"""Tests for experiment."""

from absl.testing import absltest
from absl.testing import parameterized
from experiments_jax import experiment
from experiments_jax.configs import finetuning_dknn
from experiments_jax.configs import finetuning_prev


CONFIGS_TO_TEST = [
    {
        'testcase_name': 'finetuning_learner',
        'config': finetuning_prev.get_test_config(),
    },
    {
        'testcase_name': 'finetuning_dknn_learner',
        'config': finetuning_dknn.get_test_config(),
    },
]


class BaselineTest(parameterized.TestCase):

  @parameterized.named_parameters(CONFIGS_TO_TEST)
  def test_experiment_runs_and_finishes(self, config):
    experiment_config = experiment.config_from_config_dict(config.experiment)
    experiment.run_program(experiment_config)


if __name__ == '__main__':
  absltest.main()
