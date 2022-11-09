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

"""Tests for dm_nevis.benchmarker.datasets.tasks."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_nevis.benchmarker.datasets import tasks


class TasksTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(task_key=tasks.TaskKey(
          name="task1", kind=tasks.TaskKind.CLASSIFICATION,
          metadata=tasks.ClassificationMetadata(num_classes=10))),
      dict(task_key=tasks.TaskKey(
          name="task2", kind=tasks.TaskKind.MULTI_LABEL_CLASSIFICATION,
          metadata=tasks.MultiLabelClassificationMetadata(num_classes=10))),
  ])
  def test_serialization_roundtrip(self, task_key):
    d = task_key.to_dict()
    task_key_restored = tasks.TaskKey.from_dict(d)
    self.assertEqual(task_key, task_key_restored)


if __name__ == "__main__":
  absltest.main()
