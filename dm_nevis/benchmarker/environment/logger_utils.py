"""Utils for creating logger."""
import datetime
import os
from typing import Optional, Mapping, Any

from dm_nevis.benchmarker.environment import tensorboard_writer


def generate_tensorboard_log_root() -> str:
  """Generates log root for tensorboard."""
  log_dir = os.environ.get('TENSORBOARD_LOG_DIR', '/tmp/tensorboard')
  folder_name = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
  return os.path.join(log_dir, folder_name)


def get_metrics_writer(
    tensorboard_log_root: str,
    logger_name: str,
    index_of_training_event: Optional[int] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> tensorboard_writer.TensorBoardWriter:
  """Gets metrics writer by name."""
  if logger_name == 'benchmarker':
    return tensorboard_writer.TensorBoardWriter(
        logdir=os.path.join(tensorboard_log_root, 'benchmark_metrics'),
        prefix='benchmark_metrics',
        prefix_fields=['data_split'],
        step_field='index_of_most_recent_train_event',
    )
  elif logger_name in ['learner_train', 'learner_eval']:
    metric_prefix = f'train_event_{index_of_training_event}'
    logdir = os.path.join(tensorboard_log_root, metric_prefix)
    if overrides is not None:
      overrides_str = ','.join([f'{k}={v}' for k, v in overrides.items()])
      logdir = os.path.join(logdir, overrides_str)
    logdir = os.path.join(logdir, logger_name.split('_')[1])
    return tensorboard_writer.TensorBoardWriter(
        logdir=logdir,
        prefix=metric_prefix,
    )
  elif logger_name == 'finetuning':
    return tensorboard_writer.TensorBoardWriter(
        logdir=os.path.join(tensorboard_log_root, 'finetuning'),
        prefix='finetuning',
        step_field='index_of_train_event',
    )
  else:
    raise NotImplementedError(f'Unknown logger_name {logger_name}.')
