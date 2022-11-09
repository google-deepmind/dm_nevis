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

"""Multihead models."""
import contextlib
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

from dm_nevis.benchmarker.datasets import tasks
from experiments_torch.training import heads

import torch
from torch import nn

PredictFn = Callable[[torch.Tensor, tasks.TaskKey], torch.Tensor]
LossAndMetricsFn = Callable[[torch.Tensor, torch.Tensor, bool, tasks.TaskKey],
                            Tuple[torch.Tensor, Mapping[str, float]]]


class Model(nn.Module):
  """PyTorch model.

  Attributes:
    backbone: a features extractor (CNN, MLP...) producing a flat embedding per
      image.
    heads_map: A mapping of task key to head classifier.
  """

  def __init__(self, backbone: nn.Module, heads_map: Mapping[str, heads.Head]):
    super().__init__()
    self.backbone = backbone
    self.heads_map = nn.ParameterDict(heads_map)  # pytype: disable=module-attr

  def forward(self, images: torch.Tensor, labels: torch.Tensor,
              is_training: bool, task_key: tasks.TaskKey):
    context_manager = contextlib.nullcontext if is_training else torch.no_grad
    training_mode = self.training
    self.train(is_training)
    with context_manager():
      embeddings = self.backbone(images)
      loss_and_metrics = self.heads_map[task_key.name].loss_and_metrics(
          embeddings, labels, is_training)
    self.train(training_mode)
    return loss_and_metrics

  def loss_and_metrics(self, images: torch.Tensor, labels: torch.Tensor,
                       is_training: bool, task_key: tasks.TaskKey):
    return self.forward(images, labels, is_training, task_key)

  def predict(self, images: torch.Tensor, task_key: tasks.TaskKey):
    training_mode = self.training
    self.eval()
    with torch.no_grad():
      embeddings = self.backbone(images)
      outputs = self.heads_map[task_key.name].predict(
          embeddings, is_training=False, as_probs=True)
    self.train(training_mode)
    return [o.cpu().numpy() for o in outputs]


def build_model(model_ctor: Callable[..., Any],
                supported_tasks: Iterable[tasks.TaskKey],
                head_kwargs: Optional[Dict[str, Any]] = None) -> Model:
  """Constructs a model with a backbone and multiple task heads.

  Args:
    model_ctor: Constructor for the backbone.
    supported_tasks: The tasks that the returned model supports training on.
    head_kwargs: kwargs for head constructor.

  Returns:
    A model implementing the independent baseline strategy.
  """
  head_kwargs = head_kwargs or {}
  backbone = model_ctor(name="backbone")
  heads_map = {}
  for task_key in supported_tasks:
    heads_map[task_key.name] = heads.build_head(backbone.features_dim,
                                                {task_key}, **head_kwargs)

  return Model(backbone, heads_map)


def size_summary(model: nn.Module) -> str:
  """Return a string summarizing the size of the `model` parameters."""
  num_params = sum(p.numel() for p in model.parameters())
  byte_size = num_params * 4  # 1 float32 == 4 bytes
  return f"{num_params} params ({byte_size / 1e6:.2f}MB)"


def param_summary(model: nn.Module) -> str:
  """Return a string with a detailed parameter breakdown."""
  return "\n".join([
      f"  {name}: {param.data.shape} [{param.data.dtype}]"
      for name, param in model.named_parameters()
  ])
