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

"""Tests for evaluate_embeddings."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_nevis.benchmarker.datasets import tasks
from dm_nevis.benchmarker.datasets.builders import test_dataset as test_dataset_builder
from experiments_jax.training import evaluate_embeddings
import numpy as np


EMBEDDING_DIMS = 2
IMAGE_SIZE = 8


class EvaluateEmbeddingsTest(parameterized.TestCase):

  @parameterized.parameters([
      tasks.TaskKind.CLASSIFICATION,
      tasks.TaskKind.MULTI_LABEL_CLASSIFICATION,
  ])
  def test_evaluate_embeddings_classification(self, task_kind):

    train_dataset = test_dataset_builder.get_dataset(
        split='train',
        image_size=IMAGE_SIZE,
        task_kind=task_kind,
    )

    test_dataset = test_dataset_builder.get_dataset(
        split='test',
        image_size=IMAGE_SIZE,
        task_kind=task_kind,
    )

    def random_projection_embed(state, batch):
      rng = np.random.default_rng(seed=state)
      weights = rng.uniform(size=[IMAGE_SIZE * IMAGE_SIZE * 3, EMBEDDING_DIMS])
      feats = np.reshape(batch.image, (batch.image.shape[0], -1))
      return feats @ weights

    # The states are used as seeds for the random projection embedding.
    states = [1, 2, 3]

    results = evaluate_embeddings.evaluate(
        random_projection_embed,
        states,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=10,
    )

    # The test task is easy to solve, and the KNN classifier should have less
    # than 0.1 error rate.
    max_weight = max(result.weight for result in results)
    self.assertGreater(max_weight, 0.9)

  def test_multilabel_knn_classifier(self):
    """Test the edge case that one of the labels is all the same value."""
    classifier = evaluate_embeddings._MultiLabelKNNClassifier()

    x = np.array([
        [0],
        [1],
        [2],
        [3],
        [4],
    ])

    # Note that the first column is all 0, and the final column is all 1 - this
    # exercises the edge case in sklearn that attempts to guess the number of
    # classes from the number of distinct values, where we know there are two.
    y = np.array(
        [
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
        ],
        dtype=np.int32,
    )

    classifier.fit(x, y)
    result = classifier.predict([[0], [1], [2]])
    expected = np.array([
        [0., 0.6, 0.4, 1.],
        [0., 0.6, 0.4, 1.],
        [0., 0.6, 0.4, 1.],
    ])
    np.testing.assert_allclose(expected, result)


if __name__ == '__main__':
  absltest.main()
