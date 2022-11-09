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

"""A module for custom optimizers and masks."""

from typing import Any, FrozenSet, Optional

import chex
import optax
import tree

DEFAULT_NAMES_TO_DECAY = frozenset(['w'])


def sgdw(learning_rate: Any,
         *,
         weight_decay: float,
         nesterov: Optional[float] = None,
         momentum: Optional[float] = None,
         mask: Optional[Any] = None) -> optax.GradientTransformation:
  """SGD with l2 weight decay."""

  # Decay is applied before scaling by l.r., so it is close to the classical
  # L2-regularized loss optimization.
  return optax.chain(
      optax.add_decayed_weights(weight_decay, mask),
      optax.sgd(
          learning_rate=learning_rate,
          nesterov=nesterov,
          momentum=momentum,
      ),
  )


def default_weight_decay_mask(
    updates: chex.ArrayTree,
    *,
    names_to_decay: FrozenSet[str] = DEFAULT_NAMES_TO_DECAY) -> chex.ArrayTree:
  """Masks all updates in the tree that don't have a name in the given list.

  Args:
    updates: The updates to mask.
    names_to_decay: The names of the parameters to apply weight decay to.

  Returns:
    A tree of the same shape as updates, with the value of 1 for all input
    tensors that have a name in the given list.
  """

  def mask(path, _):
    return path[-1] in names_to_decay

  return tree.map_structure_with_path(mask, updates)
