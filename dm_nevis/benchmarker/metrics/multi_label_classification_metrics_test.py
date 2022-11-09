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

"""Tests for dm_nevis.benchmarker.metrics.multi_label_classification_metrics."""

import math
from absl.testing import absltest
from absl.testing import parameterized
from dm_nevis.benchmarker.datasets import datasets
from dm_nevis.benchmarker.datasets import tasks
from dm_nevis.benchmarker.learners import learner_interface
from dm_nevis.benchmarker.metrics import multi_label_classification_metrics
import numpy as np


TASK = tasks.TaskKey('task', tasks.TaskKind.MULTI_LABEL_CLASSIFICATION,
                     tasks.MultiLabelClassificationMetadata(num_classes=4))


class MultiLabelClassificationMetrics(parameterized.TestCase):

  def test_mean_average_precision(self):
    gen = np.random.default_rng(0)
    num_classes = 2

    batches = [
        datasets.MiniBatch(
            image=gen.standard_normal(size=(1000, 4, 4, 3)),
            multi_label_one_hot=gen.integers(
                size=(1000, 4), low=0, high=num_classes),
            label=None,
        ),
        datasets.MiniBatch(
            image=gen.standard_normal(size=(300, 4, 4, 3)),
            multi_label_one_hot=gen.integers(
                size=(300, 4), low=0, high=num_classes),
            label=None,
        ),
    ]

    predictions = []
    for batch in batches:
      output = [
          np.random.uniform(size=(batch.image.shape[0],))
          for _ in range(num_classes)
      ]
      predictions.append(
          learner_interface.Predictions(batch=batch, output=output))

    metrics = multi_label_classification_metrics.compute_metrics(predictions)

    self.assertEqual(metrics.num_examples, 1300)
    # The mAP should be approximately 0.5 for randomly sampled data.
    self.assertAlmostEqual(metrics.mean_average_precision, 0.5, delta=0.05)

  def test_empty_predictions(self):
    metrics = multi_label_classification_metrics.compute_metrics([])
    self.assertEqual(metrics.num_examples, 0)
    self.assertTrue(math.isnan(metrics.mean_average_precision))


if __name__ == '__main__':
  absltest.main()
