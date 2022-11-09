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

"""A module implementing standard metrics for unary classification tasks."""

from typing import Iterable, NamedTuple

import chex
from dm_nevis.benchmarker.learners import learner_interface
import jax
import numpy as np
import optax


class ClassificationMetrics(NamedTuple):
  """Common classification metrics.

  Attributes:
    num_examples: The total number of examples used to compute the metrics.
    cross_entropy: The computed categorical cross entropy.
    top_one_accuracy: The number of examples that predicted the correct class
      first, normalized to [0, 1] by the number of examples.
    top_five_accuracy: The number of examples that predicted the correct class
      in the top five most likely classes, normalized to [0, 1] by the number of
      examples.
    top_one_correct: The integer number of predictions where the correct class
      was predicted first.
    top_five_correct: The integer number of predictions where the correct class
      was predicted in the top five predictions.
  """
  num_examples: np.ndarray
  cross_entropy: np.ndarray
  top_one_accuracy: np.ndarray
  top_five_accuracy: np.ndarray
  top_one_correct: np.ndarray
  top_five_correct: np.ndarray


def compute_metrics(
    predictions: Iterable[learner_interface.Predictions]
) -> ClassificationMetrics:
  """Computes standard unary classification metrics over predictions.

  Args:
    predictions: Predictions constist of the input batch and the learner's
      output on the given input. The input batch must contain labels confirming
      holding single integer labels corresponding to a single label multinomial
      classification task. The outputs are expected to contain unnormalized
      logits, such as the output from a linear layer with no activations.

  Returns:
    A dataclass of classification metrics for single label multinomial
  classification tasks.
  """

  top_one_correct, top_five_correct, cross_entropy = 0.0, 0.0, 0.0
  num_examples = 0

  for prediction in predictions:
    label, logits = prediction.batch.label, prediction.output[0]
    chex.assert_rank(label, 1)
    chex.assert_rank(logits, 2)

    num_examples += logits.shape[0]
    cross_entropy += _softmax_cross_entropy(logits, label).sum()
    top_one_correct += _top_n_correct(logits, label, n=1)
    top_five_correct += _top_n_correct(logits, label, n=5)

  if num_examples:
    top_one_accuracy = top_one_correct / num_examples
    top_five_accuracy = top_five_correct / num_examples
    cross_entropy = cross_entropy / num_examples
  else:
    top_one_accuracy = np.nan
    top_five_accuracy = np.nan
    cross_entropy = np.nan

  return ClassificationMetrics(
      num_examples=np.array(num_examples, dtype=int),
      cross_entropy=np.array(cross_entropy),
      top_one_accuracy=np.array(top_one_accuracy),
      top_five_accuracy=np.array(top_five_accuracy),
      top_one_correct=np.array(top_one_correct),
      top_five_correct=np.array(top_five_correct),
  )


def _top_n_correct(logits: np.ndarray, targets: np.ndarray, *,
                   n: int) -> np.ndarray:
  """Returns the number of predictions that predict the correct class in top n.

  Args:
    logits: Unnormalized logits of shape (<batch size>, <num classes>).
    targets: Unary class labels of shape (<batch size>), of integer type.
    n: The maximum index of the correct prediction in the sorted logits. if n is
      greater than or equal to the number of classes, then this function will
      return the batch size.

  Returns:
    The number of correct predictions (between 0 and <batch size>). A correct
    prediction is when the correct prediction falls within the top n largest
    values over the logits (by magnitude).
  """
  if n < 1:
    raise ValueError(f"n must be larger than 0, got {n}")

  targets = targets.reshape((*targets.shape, 1))
  top_n_predictions = np.argsort(logits, axis=-1)[:, -n:]
  return np.sum(targets == top_n_predictions)


def _softmax_cross_entropy(logits: np.ndarray,
                           targets: np.ndarray) -> np.ndarray:
  """Computes the softmax cross entropy for unnormalized logits.

  Note: This function uses jax internally, and will thus use hardware
  accleration, if one is available.

  Args:
    logits: Unnormalized logits of shape (<batch size>, <num classes>). These
      are interpreted as log probabilities, and could for example come from the
      output of a linear layer with no activations.
    targets: Unary class labels of shape (<batch size>), of integer type.

  Returns:
    The cross entropy computed between the softmax over the logits and the
    one-hot targets.
  """
  chex.assert_rank(targets, 1)
  batch_size, num_classes = logits.shape

  targets_one_hot = jax.nn.one_hot(targets, num_classes)
  cross_entropy = optax.softmax_cross_entropy(logits, targets_one_hot)

  chex.assert_shape(cross_entropy, (batch_size,))
  return np.array(cross_entropy)
