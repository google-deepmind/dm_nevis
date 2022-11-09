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

"""A datawriter logging on stdout."""
from typing import Any, Mapping

from absl import logging


class LoggingWriter:
  """A datawriter logging on stdout."""

  def __init__(self, prefix: str = ""):
    self.prefix = f"{prefix}: "

  def write(self, metrics_data: Mapping[str, Any]) -> None:
    """Writes metrics data on stdout.

    Args:
      metrics_data: A mapping of metrics name to metrics value to log.
    """
    message = self.prefix + "\n".join(
        [f"{k}: {v}" for k, v in metrics_data.items()])
    logging.info(message)

  def flush(self) -> None:
    """Flushes the buffer and ensure data is actually written."""
    logging.flush()

  def close(self) -> None:
    """Closes logging writer."""
