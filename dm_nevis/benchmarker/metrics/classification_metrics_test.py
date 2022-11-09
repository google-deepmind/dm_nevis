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

"""Tests for dm_nevis.benchmarker.metrics.classification_metrics."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_nevis.benchmarker.datasets import datasets
from dm_nevis.benchmarker.learners import learner_interface
from dm_nevis.benchmarker.metrics import classification_metrics
import numpy as np


def _prediction(label, logits):
  """Creates a prediction fixture for testing."""
  return learner_interface.Predictions(
      batch=datasets.MiniBatch(
          image=None,
          label=np.array(label, dtype=np.int32),
          multi_label_one_hot=None,
      ),
      output=[np.array(logits, dtype=np.float32)],
  )


class ImageClassificationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'uneven_batches_all_predictions_incorrect',
          'predictions': [
              _prediction([0, 1], [[0.1, 0.9], [0.9, 0.1]]),
              _prediction([0, 1], [[0.4, 0.6], [0.9, 0.1]]),
              _prediction([1], [[1000, 0]]),
          ],
          'expected_top_one_accuracy': 0.0,
          'expected_top_five_accuracy': 1.0,
          'expected_cross_entropy': 200.86228814,
          'expected_top_one_correct': 0,
          'expected_top_five_correct': 5,
          'expected_num_examples': 5,
      },
      {
          'testcase_name': 'correct_prediction_not_on_array_boundary',
          'predictions': [_prediction([2], [[0, 1, 5, 3]])],
          'expected_top_one_accuracy': 1.0,
          'expected_top_five_accuracy': 1.0,
          # verified: -log(exp(5) / (1 + exp(1) + exp(5) + exp(3)))
          'expected_cross_entropy': 0.14875513,
          'expected_top_one_correct': 1,
          'expected_top_five_correct': 1,
          'expected_num_examples': 1,
      },
      {
          'testcase_name': 'mixed_reults_within_a_single_batch',
          'predictions': [
              _prediction([2, 3], [[0, -1, 5, -3], [0, 1, 5, 3]]),
              _prediction([3], [[0, 0, 0, 1]]),
          ],
          'expected_top_one_accuracy': 2 / 3,
          'expected_top_five_accuracy': 1.0,
          'expected_cross_entropy': 0.96731003,
          'expected_top_one_correct': 2,
          'expected_top_five_correct': 3,
          'expected_num_examples': 3,
      },
      {
          'testcase_name': 'top_five_edge_case_1',
          'predictions': [_prediction([0], [[0, 5, 1, 3, 2, 4]])],
          'expected_top_one_accuracy': 0.0,
          'expected_top_five_accuracy': 0.0,
          'expected_cross_entropy': 5.45619345,
          'expected_top_one_correct': 0,
          'expected_top_five_correct': 0,
          'expected_num_examples': 1,
      },
      {
          'testcase_name': 'top_five_edge_case_2',
          'predictions': [_prediction([2], [[0, 5, 1, 3, 2, 4]])],
          'expected_top_one_accuracy': 0.0,
          'expected_top_five_accuracy': 1.0,
          'expected_cross_entropy': 4.45619345,
          'expected_top_one_correct': 0,
          'expected_top_five_correct': 1,
          'expected_num_examples': 1,
      },
      {
          'testcase_name': 'no_predictions',
          'predictions': [],
          'expected_top_one_accuracy': np.nan,
          'expected_top_five_accuracy': np.nan,
          'expected_cross_entropy': np.nan,
          'expected_top_one_correct': 0,
          'expected_top_five_correct': 0,
          'expected_num_examples': 0,
      },
  )
  def test_compute_metrics(self, predictions, expected_top_one_accuracy,
                           expected_top_five_accuracy, expected_cross_entropy,
                           expected_top_one_correct, expected_top_five_correct,
                           expected_num_examples):
    m = classification_metrics.compute_metrics(predictions)
    np.testing.assert_allclose(m.top_one_accuracy, expected_top_one_accuracy)
    np.testing.assert_allclose(m.top_five_accuracy, expected_top_five_accuracy)
    np.testing.assert_allclose(m.cross_entropy, expected_cross_entropy)
    np.testing.assert_allclose(m.num_examples, expected_num_examples)
    np.testing.assert_allclose(m.top_one_correct, expected_top_one_correct)
    np.testing.assert_allclose(m.top_five_correct, expected_top_five_correct)


if __name__ == '__main__':
  absltest.main()
