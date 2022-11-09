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

"""Defines the interface for a data writer."""
from typing import TypeVar, Protocol


MetricsData = TypeVar("MetricsData")


class DataWriter(Protocol[MetricsData]):
  """The interface that checkpoints must satisfy to work with the benchmarker.

  The checkpointer must support reading and writing checkpointable state.
  """

  def write(self, metrics_data: MetricsData) -> None:
    """Writes metrics to persistent state."""

  def flush(self) -> None:
    """Flushes the buffer and ensure data is actually written."""

  def close(self) -> None:
    """Closes metrics writer and free whatever was allocated."""
