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

"""A module implementing standard metrics for multi-label tasks."""

from typing import Iterable, NamedTuple

from dm_nevis.benchmarker.learners import learner_interface
import numpy as np
import sklearn.metrics


class MultiLabelClassificationMetrics(NamedTuple):
  """Common classification metrics.

  Attributes:
    num_examples: The total number of examples used to compute the metrics.
    mean_average_precision: Mean average precision.
  """
  num_examples: int
  mean_average_precision: float


def compute_metrics(
    predictions: Iterable[learner_interface.Predictions]
) -> MultiLabelClassificationMetrics:
  """Computes standard multi-label classification metrics over predictions.

  Args:
    predictions: Predictions constist of the input batch and the learner's
      output on the given input. The input batch must contain labels confirming
      holding single integer labels corresponding to multi-label classification
      task. The outputs are expected to contain unnormalized logits, such as the
      output from a linear layer with no activations.

  Returns:
    A dataclass of classification metrics for multi-label binary classification
  tasks.
  """
  all_probs = []
  all_targets = []

  for prediction in predictions:
    (multi_label_one_hot, probs) = (prediction.batch.multi_label_one_hot,
                                    prediction.output)
    probs = np.stack(probs, axis=1)
    if np.amin(probs) < 0.0 or np.amax(probs) > 1.0:
      raise ValueError('Probabilities must be in the range [0, 1].')

    all_probs.append(probs)
    all_targets.append(multi_label_one_hot)

  if not all_probs:
    return MultiLabelClassificationMetrics(
        num_examples=0,
        mean_average_precision=np.nan,
    )

  all_probs = np.concatenate(all_probs, axis=0)
  all_targets = np.concatenate(all_targets, axis=0)

  mean_average_precision = sklearn.metrics.average_precision_score(
      all_targets,
      all_probs,
  )

  return MultiLabelClassificationMetrics(
      num_examples=all_probs.shape[0],
      mean_average_precision=mean_average_precision,
  )
