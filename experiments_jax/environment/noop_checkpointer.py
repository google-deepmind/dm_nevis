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

"""A dummy checkpointer doing nothing."""
from typing import Any, Optional

from absl import logging


class NoOpCheckpointer:
  """A No-Operation checkpointer doing nothing."""

  def __init__(self,
               *,
               namespace: Optional[str] = None,
               base_path: Optional[str] = None,
               restore_path: Optional[str] = None):
    """Create a no-op checkpointer.

    Args:
      namespace: Appended to the base_path, so that checkpoints written with
        this writer are independent.
      base_path: if set, checkpoints will be written here.
      restore_path: path to restore state from.
    Returns: A checkpointer.
    """
    del namespace, base_path, restore_path

  def write(self, state: Any) -> None:
    """Writes a checkpoint.

    Args:
      state: Arbitrary checkpointable state
    """
    del state
    logging.warning(
        "Received checkpoint write request (ignoring it - no checkpoint will be written)."
    )

  def restore(self, *, age: int = 0) -> Optional[Any]:
    """Restores the most recent checkpointed state.

    Args:
      age: if present, the age of the checkpoint to restore.

    Returns:
      The most recent checkpoint that was successfully written using write,
      or None if no checkpoint state is available.
    """
    del age
    logging.warning(
        "Received checkpoint restore request (ignoring it - no checkpoint will be restored)."
    )
    return None
