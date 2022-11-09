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

"""A module to define the available tasks."""
import enum

from typing import Any, Dict, NamedTuple, Union


class TaskKind(enum.Enum):
  CLASSIFICATION = 0
  MULTI_LABEL_CLASSIFICATION = 1


class ClassificationMetadata(NamedTuple):
  num_classes: int


class MultiLabelClassificationMetadata(NamedTuple):
  num_classes: int

TaskMetadata = Union[ClassificationMetadata, MultiLabelClassificationMetadata]


class TaskKey(NamedTuple):
  """A hashable key to uniquely identify a task.

  Task keys must uniquely define a task, and provide the minimal information
  required to initialize a prediction head.

  Attributes:
    name: The (unique) name of the task, which may be shared across multiple
      datasets if they define matching tasks.
    kind: The kind of task (such as classification).
    metadata: The metadata of the task.
  """
  name: str
  kind: TaskKind
  metadata: TaskMetadata

  @classmethod
  def from_dict(cls, d: Dict[str, Any]) -> "TaskKey":
    """Deserializes a dict into a TaskKey."""
    if d["kind"] == "Classification":
      return cls(
          name=d["name"],
          kind=TaskKind.CLASSIFICATION,
          metadata=ClassificationMetadata(num_classes=d["num_classes"]))
    elif d["kind"] == "MultiLabelClassification":
      return cls(
          name=d["name"],
          kind=TaskKind.MULTI_LABEL_CLASSIFICATION,
          metadata=MultiLabelClassificationMetadata(
              num_classes=d["num_classes"]))
    else:
      raise ValueError("Deserialization failed")

  def to_dict(self) -> Dict[str, Any]:
    """Serializes a TaskKey into a dictionary."""
    if self.kind == TaskKind.CLASSIFICATION:
      return {
          "kind": "Classification",
          "name": self.name,
          "num_classes": self.metadata.num_classes,
      }
    elif self.kind == TaskKind.MULTI_LABEL_CLASSIFICATION:
      return {
          "kind": "MultiLabelClassification",
          "name": self.name,
          "num_classes": self.metadata.num_classes,
      }
    else:
      raise ValueError("Unknown TaskKind")
