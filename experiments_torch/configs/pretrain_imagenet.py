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
"""Pretrain learner on only imagenet using a ResNet18 (cheap architecture).

The resulting checkpoint can then be taken as initialization for all subsequent
experiments using the pre-trained baseline (PT).
"""

import functools
import os

from dm_nevis.benchmarker.datasets import streams
from dm_nevis.benchmarker.datasets import tasks
from dm_nevis.benchmarker.environment import logger_utils
from experiments_torch.learners.finetuning import finetuning_learner
from experiments_torch.training import augmentations
from experiments_torch.training import resnet
from dm_nevis.streams import nevis_stream
import ml_collections
from torch import optim

IMAGE_SIZE = 64
DEFAULT_MAX_STEPS = 50_000
DEFAULT_WARMUP_EPOCHS = 7
DEFAULT_EARLY_STOPPING_GRACE = 50_000
DEFAULT_CHECKPOINT_DIR = os.environ.get('NEVIS_CHECKPOINT_DIR',
                                        '/tmp/nevis_checkpoint_dir')


def get_config() -> ml_collections.ConfigDict:
  """The learner config, satisfying the `experiments.LearnerConfig` interface.
  """
  tensorboard_log_root = logger_utils.generate_tensorboard_log_root()
  metrics_logger_fn = functools.partial(logger_utils.get_metrics_writer,
                                        tensorboard_log_root)
  config = ml_collections.ConfigDict({
      'experiment': {
          'resume_from_checkpoint_path': None,
          'stream': {
              'ctor': streams.FilteredStream,
              'kwargs': {
                  'stream_ctor':
                      nevis_stream.NevisStream,
                  'supported_task_kinds': [tasks.TaskKind.CLASSIFICATION],
                  'stream_variant':
                      nevis_stream.NevisStreamVariant.IMAGENET_ONLY,
                  'predict_event_splits': (
                      nevis_stream.Split.DEV,
                      nevis_stream.Split.DEV_TEST,
                      nevis_stream.Split.TEST,
                  ),
              },
          },
          'learner': {
              'learner_builder': finetuning_learner.learner_builder,
              'config': {
                  'train_states_checkpoint_path': DEFAULT_CHECKPOINT_DIR,
                  'finetuning': {
                      # The strategy for initializing train state for each task.
                      'strategy':
                          finetuning_learner.FinetuningStrategy.INDEPENDENT,
                  },
                  'augmentations': {
                      'eval': {
                          'ctor': augmentations.chain,
                          'kwargs': {
                              'augmentation_ctors_with_kwargs': [
                                  (augmentations
                                   .central_crop_via_cropped_window_and_resize,
                                   {
                                       'size': (IMAGE_SIZE, IMAGE_SIZE)
                                   }),
                                  (augmentations.normalize, {}),
                              ],
                          },
                      },
                      'train': {
                          'ctor': augmentations.chain,
                          'kwargs': {
                              'augmentation_ctors_with_kwargs': [
                                  (augmentations
                                   .random_crop_via_cropped_window_and_resize, {
                                       'size': (IMAGE_SIZE, IMAGE_SIZE)
                                   }),
                                  (augmentations.random_flip, {}),
                                  (augmentations.normalize, {}),
                              ],
                          },
                      },
                  },
                  'model': {
                      'ctor': resnet.resnet34,
                      'kwargs': {},
                  },
                  # Optionally load and/or freeze pretrained parameters.
                  'load_params_fn': None,
                  'optimization': {
                      # Optimizer, must not have `learning_rate` argument as it
                      # overridden by `learning_rate_schedule``.
                      # If `learning_rate_schedule` is off, then `learning_rate`
                      # can be used.
                      'optimizer': {
                          'ctor': optim.SGD,
                          'kwargs': {
                              # Overridden by the per-task hyper-optimization.
                              # Learning rate is specified by the learning rate
                              # schedule.
                              'momentum': 0.9,
                              'nesterov': True,
                              'weight_decay': 1e-4,
                          },
                      },
                      # Learning rate schedule.
                      'learning_rate_schedule': {
                          'name': 'warmup_cosine_decay',
                          'init_learning_rate':
                              0.1,  # Can be overridden by the learner.
                          'kwargs': {
                              'warmup_epochs': DEFAULT_WARMUP_EPOCHS,
                              'final_learning_rate': 1e-8,
                          },
                      },
                  },
                  'validation_metric': 'error',
                  # Early-stopping configuration
                  'max_steps': DEFAULT_MAX_STEPS,
                  'early_stopping_grace': DEFAULT_MAX_STEPS,
                  'run_validation_every_n_steps': 1_000,
                  'image_resolution': IMAGE_SIZE,
                  'label_smoothing': 0.0,
                  'prng_seed': 1,
                  'batch_size': 256,
                  'get_metrics_writer': metrics_logger_fn,
              },
          },
      }
  })
  return config
