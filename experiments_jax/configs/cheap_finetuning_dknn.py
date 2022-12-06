"""Cheaper config, uses standard ResNet18 and smaller hyper-param search space.
"""
import concurrent
import functools
import os
from typing import Any, Optional

from dm_nevis.benchmarker.environment import logger_utils
from experiments_jax.learners.finetuning_dknn import finetuning_dknn_learner as finetuning_learner
from experiments_jax.training import augmentations
from experiments_jax.training import hype
from experiments_jax.training import hyperparameter_searcher
from experiments_jax.training import modules
from experiments_jax.training import optimizers
from experiments_jax.training import resnet
from dm_nevis.streams import example_stream
import ml_collections

IMAGE_SIZE = 64
DEFAULT_MAX_STEPS = 25_000  # Reduced number of gradient steps.
DEFAULT_WARMUP_EPOCHS = 7
DEFAULT_EARLY_STOPPING_GRACE = 10
DEFAULT_CHECKPOINT_DIR = os.environ.get('NEVIS_CHECKPOINT_DIR',
                                        '/tmp/nevis_checkpoint_dir')


def get_config(sweep_name: Optional[str] = None) -> ml_collections.ConfigDict:
  """The learner config, satisfying the `experiments.LearnerConfig` interface.
  """

  experiment_name = 'Cheap Finetuning'
  if sweep_name:
    experiment_name += f' ({sweep_name})'

  # Search over four values of learning rate only, fixed label smoothing.
  def search_space_creator(seed, num_trials):
    del seed, num_trials
    return hype.zipit([
        hype.sweep('optimization.learning_rate_schedule.init_learning_rate',
                   [1e-4, 1e-3, 1e-2, 1e-1]),
        hype.sweep('label_smoothing', [0.15, 0.15, 0.15, 0.15]),
    ])

  tensorboard_log_root = logger_utils.generate_tensorboard_log_root()
  metrics_logger_fn = functools.partial(logger_utils.get_metrics_writer,
                                        tensorboard_log_root)

  config = ml_collections.ConfigDict({
      'sweep': _get_sweep(sweep_name),
      'experiment': {
          'resume_from_checkpoint_path': None,
          'stream': {
              'ctor': example_stream.ExampleStream,
              'kwargs': {},
          },
          'learner': {
              'learner_builder': finetuning_learner.learner_builder,
              'config': {
                  'train_states_checkpoint_path':
                      DEFAULT_CHECKPOINT_DIR,
                  'finetuning': {
                      # The strategy for initializing train state for each task.
                      'strategy':
                          finetuning_learner.FinetuningStrategy.INDEPENDENT,
                      'transfer_matrix_file':
                          None,
                      'batch_size_embed_for_knn':
                          128,
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
                      'ctor': resnet.ResNet18,  # Smaller network.
                      'kwargs': {},
                  },
                  # Optionally load and/or freeze pretrained parameters.
                  'load_params_fn':
                      None,
                  'optimization': {
                      # Optimizer, must not have `learning_rate` argument as it
                      # overridden by `learning_rate_schedule``.
                      # If `learning_rate_schedule` is off, then `learning_rate`
                      # can be used.
                      'optimizer': {
                          'ctor': optimizers.sgdw,
                          'kwargs': {
                              # Overridden by the per-task hype-optimization.
                              # Learning rate is specified by the learning rate
                              # schedule.
                              'momentum': 0.9,
                              'nesterov': True,
                              'weight_decay': 1e-4,
                              'mask': optimizers.default_weight_decay_mask,
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
                  # Learner search space.
                  'search_space_creator':
                      search_space_creator,
                  # The hyperparameter searcher configuration
                  'distributed_worker_builder':
                      functools.partial(
                          hyperparameter_searcher.build_local_executor_workers,
                          executor=concurrent.futures.ThreadPoolExecutor),
                  'num_workers':  # Set this to the number of available devices.
                      1,
                  # The parameter search-space is currently hard-coded.
                  'trials_per_task':
                      4,
                  'validation_metric':
                      'error',
                  # Early-stopping configuration
                  'max_steps':
                      DEFAULT_MAX_STEPS,
                  'early_stopping_grace':
                      DEFAULT_MAX_STEPS,
                  'run_validation_every_n_steps':
                      1_000,
                  'image_resolution':
                      IMAGE_SIZE,
                  'label_smoothing':
                      0.15,
                  'prng_seed':
                      1,
                  'batch': {
                      'type': finetuning_learner.BatchSizeStrategy.ADAPTIVE,
                      'kwargs': {
                          'size_proportion': 0.0025,
                          'batch_size': 256,
                          'max_batch_size': 512,
                      },
                  },
                  'get_metrics_writer':
                      metrics_logger_fn
              },
          },
      }
  })

  return config


def _get_sweep(sweep_name: Optional[str]) -> Any:
  """Returns a sweep by name."""

  if sweep_name is None:
    return hype.product([])

  sweeps_to_include = set(sweep_name.split(','))
  hyperparameter_iterators = []

  if 'max_1000_steps' in sweeps_to_include:
    hyperparameter_iterators.append(_max_1000_steps_sweep())

  if 'max_10_000_steps' in sweeps_to_include:
    hyperparameter_iterators.append(_max_10000_steps_sweep())

  if 'run_validation_every_n_steps_ablation' in sweeps_to_include:
    hyperparameter_iterators.append(
        _run_validation_every_n_steps_ablation_sweep())

  if 'number_of_steps' in sweeps_to_include or 'warmup_epochs' in sweeps_to_include:
    hyperparameter_iterators.append(number_of_steps_sweep())

  if 'seeds' in sweeps_to_include:
    hyperparameter_iterators.append(_seed_sweep())

  if 'models' in sweeps_to_include:
    hyperparameter_iterators.append(
        hype.sweep('experiment.learner.config.model.ctor',
                   [modules.VGG, resnet.CifarResNet34]))

  strategy_sweep = _finetuning_strategy_sweep(sweeps_to_include)
  if strategy_sweep is not None:
    hyperparameter_iterators.append(strategy_sweep)

  if not hyperparameter_iterators:
    raise ValueError('Unrecognized sweep name.')

  return hype.product(hyperparameter_iterators)


def number_of_steps_sweep():
  return hype.zipit([
      hype.sweep(
          'experiment.learner.config.max_steps',
          [DEFAULT_MAX_STEPS // 2, DEFAULT_MAX_STEPS, DEFAULT_MAX_STEPS * 2]),
      hype.sweep(
          'experiment.learner.config.early_stopping_grace',
          [DEFAULT_MAX_STEPS // 2, DEFAULT_MAX_STEPS, DEFAULT_MAX_STEPS * 2]),
      hype.sweep(
          'experiment.learner.config.optimization.learning_rate_schedule.kwargs.warmup_epochs',
          [
              DEFAULT_WARMUP_EPOCHS // 2, DEFAULT_WARMUP_EPOCHS,
              DEFAULT_WARMUP_EPOCHS * 2
          ]),
  ])


def _seed_sweep():
  return hype.sweep('experiment.learner.config.prng_seed', [1, 2, 3, 4, 5])


def _max_1000_steps_sweep():
  return hype.sweep('experiment.learner.config.max_steps', [1_000])


def _max_10000_steps_sweep():
  return hype.sweep('experiment.learner.config.max_steps', [10_000])


def _run_validation_every_n_steps_ablation_sweep():
  return hype.sweep('experiment.learner.config.run_validation_every_n_steps', [
      10,
      100,
      1_000,
      2_000,
  ])


def _finetuning_strategy_sweep(sweeps_to_include):
  """Constructs a sweep over the named finetuning strategies."""
  strategies = []
  paths = []

  if 'independent' in sweeps_to_include:
    strategies.append(finetuning_learner.FinetuningStrategy.INDEPENDENT)
    paths.append('')

  if 'previous' in sweeps_to_include:
    strategies.append(finetuning_learner.FinetuningStrategy.PREVIOUS)
    paths.append('')

  if 'dynamic_knn_transfer_matrix' in sweeps_to_include:
    strategies.append(
        finetuning_learner.FinetuningStrategy.DYNAMIC_KNN_TRANSFER_MATRIX)
    paths.append('')

  if not strategies:
    return None

  return hype.zipit([
      hype.sweep(
          'experiment.learner.config.finetuning.strategy',
          strategies,
      ),
      hype.sweep(
          'experiment.learner.config.finetuning.transfer_matrix_file',
          paths,
      ),
  ])
