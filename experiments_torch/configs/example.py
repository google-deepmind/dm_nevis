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
"""Example learner config.

This is for use with `learners/example`, and is intended to show how to
implement a bare-bones learner.
"""
import functools

from dm_nevis.benchmarker.environment import logger_utils
from experiments_torch.learners.example import example_learner
from dm_nevis.streams import nevis_stream
import ml_collections


def get_config() -> ml_collections.ConfigDict:
  """The learner config, satisfying the `experiments.LearnerConfig` interface.
  """

  tensorboard_log_root = logger_utils.generate_tensorboard_log_root()
  metrics_logger_fn = functools.partial(logger_utils.get_metrics_writer,
                                        tensorboard_log_root)

  return ml_collections.ConfigDict({
      'experiment': {
          'resume_from_checkpoint_path': None,
          'stream': {
              'ctor': nevis_stream.NevisStream,
              'kwargs': {
                  'stream_variant': nevis_stream.NevisStreamVariant.DEBUG,
              }
          },
          'learner': {
              'learner_builder': example_learner.learner_builder,
              'config': {
                  'get_metrics_writer': metrics_logger_fn
              }
          },
      },
  })
