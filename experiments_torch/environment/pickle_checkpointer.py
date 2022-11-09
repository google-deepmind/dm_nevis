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
"""A checkpointer that saves with pickle."""
import os
import pickle
from typing import Any, Optional

from absl import logging


class PickleCheckpointer:
  """A checkpointer that saves with pickle.

  The current checkpointer will always overwrite the most recent checkpoint
  in the bash path.
  """

  def __init__(self, base_path: str):
    """Creates a pickle checkpointer.

    Args:
      base_path: Path to write checkpoints to.
    Returns: A checkpointer.
    """
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    self.base_path = base_path

  def write(self, state: Any) -> None:
    """Writes a checkpoint to the base path.

    Args:
      state: Arbitrary checkpointable state
    """
    logging.info("Saving checkpoint to %s", self.base_path)
    partial_path = f"{self.base_path}.part"
    with open(partial_path, "wb") as f:
      pickle.dump(state, f)
    os.rename(partial_path, self.base_path)

  def restore(self) -> Optional[Any]:
    """Restores the most recent checkpointed state.

    Returns:
      The most recent checkpoint that was successfully written using write,
      or None if no checkpoint state is available.
    """
    if not os.path.exists(self.base_path):
      logging.warning("No checkpoint found at %s", self.base_path)
      return None

    logging.info("Restore checkpoint from %s", self.base_path)
    with open(self.base_path, "rb") as f:
      state = pickle.load(f)
    return state
