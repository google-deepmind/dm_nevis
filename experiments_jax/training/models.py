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

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

import chex
import distrax
from dm_nevis.benchmarker.datasets import tasks
from experiments_jax.training import heads
import haiku as hk
import jax
import numpy as np

PredictFn = Callable[[hk.Params, hk.State, chex.PRNGKey, chex.ArrayTree, bool],
                     Tuple[chex.ArrayTree, hk.State]]
LossAndMetricsFn = Callable[
    [hk.Params, hk.State, chex.PRNGKey, chex.ArrayTree, chex.ArrayTree, bool],
    Tuple[Tuple[chex.ArrayTree, chex.ArrayTree], hk.State]]
InitFn = Callable[[chex.PRNGKey], Tuple[hk.Params, hk.State]]


@chex.dataclass
class Model:
  init: InitFn
  predict: Mapping[tasks.TaskKey, PredictFn]
  loss_and_metrics: Mapping[tasks.TaskKey, LossAndMetricsFn]


def build_model(model_ctor: Callable[..., Any],
                supported_tasks: Iterable[tasks.TaskKey],
                image_resolution: int,
                head_kwargs: Optional[Dict[str, Any]] = None) -> Model:
  """Constructs a model with a backbone and multiple task heads.

  Args:
    model_ctor: Constructor for the backbone.
    supported_tasks: The tasks that the returned model supports training on.
    image_resolution: The suppored image resolution of the returned model.
    head_kwargs: kwargs for head constructor.

  Returns:
    A model implementing the independent baseline strategy.
  """
  head_kwargs = head_kwargs or {}
  init_fns = {}
  predict_fns = {}
  loss_and_metrics_fns = {}
  for task in set(supported_tasks):

    @hk.transform_with_state
    def predict(img, is_training, task=task):
      backbone = model_ctor(name="backbone")
      head = heads.build_head({task}, **head_kwargs)

      feats = backbone(img, is_training)
      distributions = head.predict(feats, is_training)
      probs = []
      for distribution in distributions:
        assert (isinstance(distribution, distrax.Categorical) or
                isinstance(distribution, distrax.Bernoulli))
        probs.append(distribution.probs)
      return probs

    @hk.transform_with_state
    def loss_and_metrics(img, labels, is_training, task=task):
      backbone = model_ctor(name="backbone")
      head = heads.build_head({task}, **head_kwargs)

      feats = backbone(img, is_training)
      return head.loss_and_metrics(feats, labels, is_training)

    init_fns[task] = loss_and_metrics.init
    predict_fns[task] = jax.jit(predict.apply, static_argnums=[4])
    loss_and_metrics_fns[task] = jax.jit(
        loss_and_metrics.apply, static_argnums=[5])

  def init(rng_key):
    images = np.zeros([1, image_resolution, image_resolution, 3])
    all_params = None
    all_states = None
    for task_key, fn in init_fns.items():
      if task_key.kind == tasks.TaskKind.MULTI_LABEL_CLASSIFICATION:
        num_classes = task_key.metadata.num_classes
        labels = np.zeros([1, num_classes], dtype=np.int32)
      else:
        labels = np.zeros([1], dtype=np.int32)
      params, state = fn(rng_key, images, labels, True)

      if all_params is None:
        all_params = params
        all_states = state
      else:
        all_params = hk.data_structures.merge(all_params, params)
        all_states = hk.data_structures.merge(all_states, state)
    return all_params, all_states

  return Model(
      init=init,
      predict=predict_fns,
      loss_and_metrics=loss_and_metrics_fns,
  )


def size_summary(params: chex.ArrayTree) -> str:
  """Returns a string summarizing the size of `params`."""
  num_params = hk.data_structures.tree_size(params)
  byte_size = hk.data_structures.tree_bytes(params)
  return f"{num_params} params ({byte_size / 1e6:.2f}MB)"


def param_summary(params: chex.ArrayTree) -> str:
  """Returns a string with a detailed parameter breakdown."""
  return "\n".join([
      f"  {m}/{n}: {v.shape} [{v.dtype}]"
      for m, n, v in hk.data_structures.traverse(params)
  ])
