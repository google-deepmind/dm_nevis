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

from typing import Dict, List, Optional, Set, Tuple, Union, Protocol

from dm_nevis.benchmarker.datasets import tasks
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


Loss = torch.Tensor
Metrics = Dict[str, Union[np.ndarray, float]]
TaskKey = tasks.TaskKey


class Head(Protocol):
  """A Prediction head.

  Heads combine a prediction layer to map some representation to a prediction
  together with a loss function and other diagnostics appropriate for the
  kind of prediction.
  """

  def predict(self, inputs: torch.Tensor, is_training: bool,
              as_probs: bool) -> List[torch.Tensor]:
    """Generates a prediction given the representation `h`.

    Args:
      inputs: representation to derive predictions from.
      is_training: bool
      as_probs: bool

    Returns:
      A list over distribution objects representing the predictions (one for
    each label).
    """

  def loss_and_metrics(
      self,
      inputs: torch.Tensor,
      targets: torch.Tensor,
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


class CategoricalHead(nn.Module):
  """A categorical prediction head.

  Encapsulates a linear layer to predict logits given a representation
  and computes relevant metrics such as xent, error, expected-calibration-error
  given ground truth labels.
  """

  def __init__(self,
               features_dim: int,
               num_classes: int,
               label_smoothing: float = 0.,
               name: Optional[str] = None):
    super().__init__()
    self._num_classes = num_classes
    self._label_smoothing = label_smoothing
    self._logit_layer = nn.Linear(features_dim, num_classes)

  def forward(self, x):
    return self._logit_layer(x)

  def predict(
      self,
      inputs: torch.Tensor,
      is_training: bool = False,
      as_probs: bool = False,
  ) -> List[torch.Tensor]:
    """Computes class probabilities given representations."""
    del is_training
    logits = self.forward(inputs)
    if as_probs:
      return [F.softmax(logits, -1)]
    return [logits]

  def loss_and_metrics(
      self,
      inputs: torch.Tensor,
      targets: torch.Tensor,
      is_training: bool = False,
  ) -> Tuple[Loss, Metrics]:
    """Computes loss and metrics given representations and target labels."""
    assert len(targets.shape) == 1  # [batch_size]

    # Categorical distribuion
    logits = self.predict(inputs, is_training=is_training)[0]
    log_probs = F.log_softmax(logits, dim=-1)

    if self._label_smoothing != 0 and is_training:
      one_hot_targets = F.one_hot(targets, self._num_classes)  # pytype: disable=module-attr
      smoothed_targets = (
          one_hot_targets * (1 - self._label_smoothing) +
          self._label_smoothing / self._num_classes)

      neg_log_probs = -log_probs
      assert len(neg_log_probs.shape) == 2  # [batch_size, num_classes]
      xent = torch.sum(smoothed_targets * neg_log_probs, dim=1)
    else:
      xent = F.cross_entropy(logits, targets, reduce=False)  # pytype: disable=wrong-keyword-args

    predicted_labels = logits.argmax(dim=-1)
    error = torch.ne(predicted_labels, targets).float()
    loss = torch.mean(xent)

    return (loss, {
        "loss": float(loss.item()),
        "xent": xent.detach().cpu().numpy(),
        "error": error.detach().cpu().numpy()
    })


class MultiLabelHead(nn.Module):
  """A binary multi-label prediction head.

  Encapsulates a linear layer to predict logits given a representation
  and computes relevant metrics such as cross entropy, error,
  expected-calibration-error given ground truth labels.
  """

  def __init__(self,
               features_dim: int,
               num_classes: int,
               label_smoothing: float = 0.,
               name: Optional[str] = None):
    super().__init__()
    self._num_classes = num_classes
    self._label_smoothing = label_smoothing
    self._logit_layer = nn.Linear(features_dim, num_classes)

  def forward(self, x):
    return self._logit_layer(x)

  def predict(self,
              inputs: torch.Tensor,
              is_training: bool = False,
              as_probs=False) -> List[torch.Tensor]:
    """Computes class logits given representations."""
    del is_training

    logits = self.forward(inputs)

    output_distributions = []

    for i in range(self._num_classes):
      if as_probs:
        output_distributions.append(F.sigmoid(logits[:, i]))
      else:
        output_distributions.append(logits[:, i])
    return output_distributions

  def loss_and_metrics(self,
                       inputs: torch.Tensor,
                       targets: torch.Tensor,
                       is_training: bool = False) -> Tuple[Loss, Metrics]:
    """Computes loss and metrics given representations and target labels."""
    assert len(targets.shape) == 2  # [batch_size, num_classes]

    # Product of independent Bernoulli.
    predictive_distributions = self.predict(inputs, is_training=is_training)
    cross_entropies = []
    predicted_labels = []
    errors = []
    for i, predictive_distribution in enumerate(predictive_distributions):
      expected_label = targets[:, i]
      if self._label_smoothing != 0 and is_training:
        smoothed_targets = (
            expected_label * (1 - self._label_smoothing) +
            self._label_smoothing / 2)
        cross_entropies.append(
            F.binary_cross_entropy_with_logits(predictive_distribution,
                                               smoothed_targets))
      else:
        cross_entropies.append(
            F.binary_cross_entropy_with_logits(predictive_distribution,
                                               expected_label))
      predicted_label = (F.sigmoid(predictive_distribution) > 0.5).long()
      predicted_labels.append(predicted_label)
      error = torch.ne(predicted_label, expected_label.long())
      errors.append(error.float())

    cross_entropies = torch.stack(cross_entropies, dim=-1)
    error = torch.stack(errors, dim=-1)
    loss = torch.mean(cross_entropies)

    return (loss, {
        "loss": float(loss.item()),
        "xent": cross_entropies.detach().cpu().numpy(),
        "error": error.detach().cpu().numpy()
    })


def build_head(features_dim: int, task_keys: Set[TaskKey], **kwargs) -> Head:
  """Builds an appropriate head for the given task."""
  assert len(task_keys) == 1

  task_key = list(task_keys)[0]

  task_kind = task_key.kind

  if task_kind == tasks.TaskKind.CLASSIFICATION:
    return CategoricalHead(
        features_dim=features_dim,
        num_classes=task_key.metadata.num_classes,
        name=f"{task_key.name}_head",
        **kwargs)
  elif task_kind == tasks.TaskKind.MULTI_LABEL_CLASSIFICATION:
    return MultiLabelHead(
        features_dim=features_dim,
        num_classes=task_key.metadata.num_classes,
        name=f"{task_key.name}_head",
        **kwargs)
  else:
    raise ValueError(f"Unsupported task kind: {task_kind}")
