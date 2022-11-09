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
"""A writer that logs metrics to tensorboard."""
import numbers
from typing import Any, Sequence

from absl import logging
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
try:
  import torch  # pylint: disable=g-import-not-at-top
except ModuleNotFoundError:
  torch = None


class TensorBoardWriter:
  """A writer that logs metrics to tensorboard."""

  def __init__(
      self,
      logdir: str,
      prefix: str = "",
      prefix_fields: Sequence[str] = (),
      step_field: str = "step",
  ):
    """Constructs a writer that logs metrics to tensorboard.

    Args:
      logdir: Logging directory used to instantiate `tf.summary.SummaryWriter`.
      prefix: Harcoded prefix to add to each metric.
      prefix_fields: A sequence of metric names that will be used as prefix of
        metric. They appear after the hardcoded prefix.
      step_field: The name of the metric used as `step` argument to logging with
        tensorboard. It will be the x-axis on tensorboard dashboard.
    """
    self._prefix = prefix
    logging.info("Created TensorBoardWriter with logdir=%s", logdir)
    self._file_writer = tf.summary.create_file_writer(logdir)
    self._prefix_fields = prefix_fields
    self._step_field = step_field

  def write(self, metrics_data: dict[str, Any]) -> None:
    """Write metrics data on stdout.

    Args:
      metrics_data: A mapping of metrics name to metrics value to log.
    """
    # Extract logging step from metrics.
    if self._step_field not in metrics_data:
      raise ValueError("metrics_data doesn't contain the field that is"
                       f" used as step, metrics_data={metrics_data},"
                       f" step_field={self._step_field}")
    step = metrics_data.pop(self._step_field)

    # Construct prefix for metric.
    prefix = self._construct_prefix(metrics_data)

    for metric_name, value in metrics_data.items():
      self._log_one_metric(prefix, metric_name, value, step)

  def _construct_prefix(self, metrics_data: dict[str, Any]) -> str:
    """Constructs prefix for each metric name."""
    prefixes = [self._prefix] if self._prefix else []
    for field in self._prefix_fields:
      val = metrics_data.pop(field, None)
      if val is not None:
        prefixes.append(val)
    prefix = "/".join(prefixes)
    return prefix

  def _log_one_metric(
      self,
      prefix: str,
      metric_name: str,
      value: Any,
      step: int,
  ) -> None:
    """Logs one metric value."""
    tf_metric_name = f"{prefix}/{metric_name}"
    if torch is not None and isinstance(value, torch.Tensor):
      if torch.numel(value) == 1:
        value = value.item()
      else:
        logging.warning(
            "%sTrying to log %s with shape %s: %s,"
            " while only scalar is supported.", "*" * 50, type(value),
            value.shape, value)
        return
    if isinstance(value, jnp.ndarray) or isinstance(value, np.ndarray):
      if value.size == 1:
        value = value.item()
      else:
        logging.warning(
            "%sTrying to log %s with shape %s: %s,"
            " while only scalar is supported.", "*" * 50, type(value),
            value.shape, value)
        return
    with self._file_writer.as_default():
      if isinstance(value, numbers.Number):
        tf.summary.scalar(tf_metric_name, value, step=step)
      elif isinstance(value, str):
        tf.summary.text(tf_metric_name, value, step=step)
      else:
        logging.warning(
            "%sCan't handle metric '%s' which has type %s: %s",
            "*" * 50 + "\n",
            metric_name,
            type(value),
            value,
        )

  def flush(self) -> None:
    """Flushes the buffer and ensure data is actually written."""
    logging.info("flush tensorboard metrics")
    self._file_writer.flush()
    logging.flush()

  def close(self) -> None:
    """Closes logging writer."""
    logging.info("close tensorboard writer")
    self._file_writer.close()
