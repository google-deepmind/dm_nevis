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

"""Prediction heads."""

from typing import Dict, List, Optional, Set, Tuple, Protocol

import chex
import distrax
from dm_nevis.benchmarker.datasets import tasks
import haiku as hk
import jax
import jax.numpy as jnp


Loss = chex.Array
Metrics = Dict[str, chex.Array]
TaskKey = tasks.TaskKey


class Head(Protocol):
  """A Prediction head.

  Heads combine a prediction layer to map some representation to a prediction
  together with a loss function and other diagnostics appropriate for the
  kind of prediction.
  """

  def predict(
      self,
      inputs: chex.Array,
      is_training: bool,
  ) -> List[distrax.Distribution]:
    """Generates a prediction given the representation `h`.

    Args:
      inputs: representation to derive predictions from.
      is_training: bool

    Returns:
      A list over distribution objects representing the predictions (one for
    each label).
    """

  def loss_and_metrics(
      self,
      inputs: chex.Array,
      targets: chex.Array,
      is_training: bool = False,
  ) -> Tuple[Loss, Metrics]:
    """Evaluates the predictions given representations and ground-truth targets.

    Args:
      inputs: representation to derive predictions from.
      targets: ground-truth to evaluate against.
      is_training: bool

    Returns:
      A dictionary with per-example metrics and a scalar "loss".
    """


class CategoricalHead(hk.Module):
  """A categorical prediction head.

  Encapsulates a linear layer to predict logits given a representation
  and computes relevant metrics such as xent, error, expected-calibration-error
  given ground truth labels.
  """

  def __init__(self,
               num_classes: int,
               label_smoothing: float = 0.,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._num_classes = num_classes
    self._label_smoothing = label_smoothing
    self._logit_layer = hk.Linear(num_classes)

  def predict(
      self,
      inputs: chex.Array,
      is_training: bool = False,
  ) -> List[distrax.Categorical]:
    """Computes class probabilities given representations."""
    del is_training
    return [distrax.Categorical(logits=self._logit_layer(inputs))]

  def loss_and_metrics(
      self,
      inputs: chex.Array,
      targets: chex.Array,
      is_training: bool = False,
  ) -> Tuple[Loss, Metrics]:
    """Computes loss and metrics given representations and target labels."""
    chex.assert_rank(targets, 1)  # [batch_size]

    # Categorical distribuion
    predictive_distribution = self.predict(inputs, is_training=is_training)[0]

    if self._label_smoothing != 0 and is_training:
      one_hot_targets = jax.nn.one_hot(targets, self._num_classes)
      smoothed_targets = (one_hot_targets * (1 - self._label_smoothing) +
                          self._label_smoothing / self._num_classes)

      neg_log_probs = -jax.nn.log_softmax(predictive_distribution.logits)
      chex.assert_rank(neg_log_probs, 2)  # [batch_size, num_classes]
      xent = jnp.sum(smoothed_targets * neg_log_probs, axis=1)
    else:
      xent = -predictive_distribution.log_prob(targets)

    predicted_labels = predictive_distribution.mode()
    error = jnp.not_equal(predicted_labels, targets).astype(jnp.float32)
    loss = jnp.mean(xent)

    return (loss, {"loss": loss, "xent": xent, "error": error})


class MultiLabelHead(hk.Module):
  """A binary multi-label prediction head.

  Encapsulates a linear layer to predict logits given a representation
  and computes relevant metrics such as cross entropy, error,
  expected-calibration-error given ground truth labels.
  """

  def __init__(self,
               num_classes: int,
               label_smoothing: float = 0.,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._num_classes = num_classes
    self._label_smoothing = label_smoothing
    self._logit_layer = hk.Linear(num_classes)

  def predict(self,
              inputs: chex.Array,
              is_training: bool = False) -> List[distrax.Bernoulli]:
    """Computes class logits given representations."""
    del is_training

    logits = self._logit_layer(inputs)

    output_distributions = []

    for i in range(self._num_classes):
      output_distributions.append(distrax.Bernoulli(logits=logits[:, i]))
    return output_distributions

  def loss_and_metrics(
      self,
      inputs: chex.Array,
      targets: chex.Array,
      is_training: bool = False,
  ) -> Tuple[Loss, Metrics]:
    """Computes loss and metrics given representations and target labels."""
    chex.assert_rank(targets, 2)  # [batch_size, num_classes]

    # Product of independent Bernoulli.
    predictive_distributions = self.predict(inputs, is_training=is_training)
    cross_entropies = []
    predicted_labels = []
    errors = []
    for i, predictive_distribution in enumerate(predictive_distributions):
      expected_label = targets[:, i]
      if self._label_smoothing != 0 and is_training:
        smoothed_targets = (expected_label * (1 - self._label_smoothing) +
                            self._label_smoothing / 2)
        cross_entropies.append(
            -predictive_distribution.log_prob(smoothed_targets))
      else:
        cross_entropies.append(
            -predictive_distribution.log_prob(expected_label))
      predicted_label = predictive_distribution.mode()
      predicted_labels.append(predicted_label)
      error = jnp.not_equal(predicted_label, expected_label).astype(jnp.float32)
      errors.append(error)

    cross_entropies = jnp.stack(cross_entropies, axis=-1)
    error = jnp.mean(jnp.stack(errors, axis=-1), axis=-1)
    loss = jnp.mean(cross_entropies)

    return (loss, {"loss": loss, "xent": cross_entropies, "error": error})


def build_head(task_keys: Set[TaskKey], **kwargs) -> Head:
  """Builds an appropriate head for the given task."""
  assert len(task_keys) == 1

  task_key = list(task_keys)[0]

  task_kind = task_key.kind

  if task_kind == tasks.TaskKind.CLASSIFICATION:
    return CategoricalHead(
        num_classes=task_key.metadata.num_classes, name=f"{task_key.name}_head",
        **kwargs)
  elif task_kind == tasks.TaskKind.MULTI_LABEL_CLASSIFICATION:
    return MultiLabelHead(
        num_classes=task_key.metadata.num_classes, name=f"{task_key.name}_head",
        **kwargs)
  else:
    raise ValueError(f"Unsupported task kind: {task_kind}")
