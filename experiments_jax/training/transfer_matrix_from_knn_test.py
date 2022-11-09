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

"""Tests for transfer_matrix_from_knn."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_nevis.benchmarker.datasets import tasks
from dm_nevis.benchmarker.datasets.builders import test_dataset as test_dataset_builder
from experiments_jax.training import transfer_matrix_from_knn
import numpy as np


class TransferMatrixFromKnnTest(parameterized.TestCase):

  def test_give_me_a_name(self):
    train_dataset = test_dataset_builder.get_dataset('train', start=0, end=10)
    test_dataset = test_dataset_builder.get_dataset('test', start=0, end=10)

    def embedding_fn(state, batch):
      del state
      return np.zeros([batch.image.shape[0], 32])

    t1 = tasks.TaskKey(
        'task_1',
        tasks.TaskKind.CLASSIFICATION,
        tasks.ClassificationMetadata(2),
    )

    t2 = tasks.TaskKey(
        'task_2',
        tasks.TaskKind.CLASSIFICATION,
        tasks.ClassificationMetadata(2),
    )

    tasks_and_train_states = [(t1, 0), (t2, 1)]
    m = transfer_matrix_from_knn.compute_transfer_matrix_using_knn_classifier(
        embedding_fn,
        tasks_and_train_states,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=2,
        preprocessing_fn=lambda x: x,
    )

    self.assertEqual(m.source_tasks, [t1, t2])
    self.assertEqual(m.target_tasks, [train_dataset.task_key])


if __name__ == '__main__':
  absltest.main()
