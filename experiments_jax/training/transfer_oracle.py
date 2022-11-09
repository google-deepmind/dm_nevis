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

"""Precomputed transfer matrix."""

import itertools
import json
from typing import Any, Dict, List, Sequence, Tuple

from dm_nevis.benchmarker.datasets import tasks
import numpy as np


class TransferMatrix:
  """Represents a transfer matrix between TaskKeys.

  The transfer matrix is represented as an R^(n, m) matrix, where n is the
  number of source tasks (tasks that can be transferred from) and m is the
  number of target tasks.

  Given the transfer matrix M, we take

  M[i,j] := e_j(p_i) - c(i, j).

  Where e_j(p_i) is an estimated error achieved when fitting a model to solve
  the target task j using parameters p_i, trained using source task i.
  c(i, j) is a computed normalizing offset.

  The normalizing c(i, j) offset may be zero, if no extra information is known,
  otherwise c(i, j) represents e_j(p_z), where p_z represents a randomly
  initialized set of parameters.

  This means that when selecting tasks to transfer, smaller values correspond
  to the best source tasks.

  TODO: unfortunately, given this convention, it appears to not be
  possible to distinguish the case where the value c(.,.) is known, and it is
  not advantageous to transfer from any existing task, from the case where no
  offsetting constants are known. We should find a way to improve this.
  """

  def __init__(self, source_tasks: Sequence[tasks.TaskKey],
               target_tasks: Sequence[tasks.TaskKey], matrix: np.ndarray):
    """Represents a transfer matrix from a source-tasks to target tasks."""
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (len(source_tasks), len(target_tasks))

    self.source_tasks = source_tasks
    self.target_tasks = target_tasks
    self.transfer_matrix = matrix

  @classmethod
  def from_file(cls, filename: str) -> "TransferMatrix":
    """Loads a transfer matrix from the given file.

    The file has to be a .json-file with 3 fields: "source_tasks" and
    "target_tasks" and lists with TaskKeys for the rows and columns
    of the transfer matrix respectively.
    "matrix" contains a numpy array. The rows correspond to source-tasks and
    the columns to target tasks. The upper triangular matrix describes causally
    admissable pairs when sequentially ingesting tasks.
    See colabs/transfer_matrix.ipynb to recreate a transfer matrix.

    Args:
      filename: The filename for a transfer-matrix pickle file. If the supplied
        filename comes without directory component, it is assumed to be raltive
        to the ./transfer_matrices/ resource directory.

    Returns:
      A TransferMatrix instance.
    """
    with open(filename, "rb") as f:
      return cls.from_dict(json.load(f))

  @classmethod
  def from_dict(cls, d: Dict[str, Any]) -> "TransferMatrix":
    return cls(
        source_tasks=[tasks.TaskKey.from_dict(e) for e in d["source_tasks"]],
        target_tasks=[tasks.TaskKey.from_dict(e) for e in d["target_tasks"]],
        matrix=np.asarray(d["matrix"]))

  def to_dict(self) -> Dict[str, Any]:
    return dict(
        source_tasks=[task.to_dict() for task in self.source_tasks],
        target_tasks=[task.to_dict() for task in self.target_tasks],
        matrix=self.transfer_matrix.tolist())

  def task_key_by_name(self, name: str) -> tasks.TaskKey:
    """Returns the task_key for the given task name."""
    for tk in itertools.chain(self.target_tasks, self.source_tasks):
      if tk.name == name:
        return tk
    raise ValueError(f"Unknown task_name '{name}'")

  def transfer_tasks(self,
                     task: tasks.TaskKey) -> List[Tuple[tasks.TaskKey, float]]:
    """Returns a list of source transfer tasks, ordered by their usefullness.

    Args:
      task: a target task to transfer to.

    Returns:
      a list of (task_key, transfer)-tuples (tasks with highest transfer first).
    """
    if task not in self.target_tasks:
      return []

    col_idx = self.target_tasks.index(task)
    transfer_column = self.transfer_matrix[:, col_idx]
    return [(self.source_tasks[i], transfer_column[i])
            for i in transfer_column.argsort()]
