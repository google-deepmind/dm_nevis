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

"""Learning rate schedules for Nevis benchmarker."""

from typing import Any, Dict, NamedTuple, Sequence

from absl import logging
import optax


class ProgressAndScale(NamedTuple):
  """Combines a progress and scale.

  Attributes:
    progress: the progress (in the range [0, 1] through the full train loop).
    scale: The learning rate scaling to apply once the given progress has been
      completed.
  """
  progress: float
  scale: float


def piecewise_constant_progress_aligned_learning_rate_schedule(
    init_value: float,
    max_steps: int,
    learning_progress_boundaries_and_scales: Sequence[ProgressAndScale],
) -> optax.Schedule:
  """Piece-wise constant learning rate depending on learning progress.

  Args:
    init_value: Initial value of the learning rate.
    max_steps: Maximum number of training steps (batched weight updates).
    learning_progress_boundaries_and_scales: A sequence of tuples `(progress,
      scale)`, where `progress` indicates a the portion (in [0,1] range) of
      `max_steps` at which learning rate is scaled by `scale`.

  Returns:
    Learning rate schedule function.
  """
  boundaries_and_scales = {}
  for (progress, scale) in learning_progress_boundaries_and_scales:
    step = int(progress * max_steps)
    boundaries_and_scales[step] = scale

  logging.info('Using piecewise linear.\n Boundaries: \n%s',
               boundaries_and_scales)

  return optax.piecewise_constant_schedule(
      init_value, boundaries_and_scales=boundaries_and_scales)


def constant_learning_rate_schedule(init_value: float) -> optax.Schedule:
  """Constant learning rate schedule."""
  return lambda s: init_value


def warmup_cosine_decay_learning_rate_schedule(
    initial_learning_rate: float, steps_per_epoch: int, max_steps: int,
    warmup_epochs: int, final_learning_rate: float) -> optax.Schedule:
  """Warmup cosine learning rate schedule."""

  # The warmup steps must be strictly less than the number of overall steps.
  warmup_steps = min(max_steps - 1, warmup_epochs * steps_per_epoch)

  logging.info(
      'Cosine decay schedule: warmup: %d, max steps: %d',
      warmup_steps,
      max_steps,
  )

  return optax.warmup_cosine_decay_schedule(
      init_value=0.0,
      peak_value=initial_learning_rate,
      warmup_steps=warmup_steps,
      end_value=final_learning_rate,
      decay_steps=max_steps)


def build_learning_rate_schedule(
    learning_rate_schedule_name: str, initial_learning_rate: float,
    steps_per_epoch: int, max_steps: int,
    learning_rate_schedule_kwargs: Dict[str, Any]) -> optax.Schedule:
  """Creates a learning_rate_schedule function for given arguments.

  This function assumes that `steps_per_epoch` and `max_steps` are not contained
  in `learning_rate_schedule_kwargs`. The reason for this constraint is due to
  the fact that these arguments could be dynamically recomputed on the learner
  side depending on which dataset is used.

  Args:
    learning_rate_schedule_name: A name of a learning rate schedule.
    initial_learning_rate: An initial value of the learning rate.
    steps_per_epoch: Number of batched weight updates per epoch.
    max_steps: Maximum number of batched weight updates for the training run.
    learning_rate_schedule_kwargs: Dictionary containing additional arguments
      for a given learning rate schedule.

  Returns:
    Learning rate schedule.


  """
  if 'steps_per_epoch' in learning_rate_schedule_kwargs:
    raise ValueError(
        '`steps_per_epoch` must not be in `learning_rate_schedule_kwargs`.')
  if 'max_steps' in learning_rate_schedule_kwargs:
    raise ValueError(
        '`max_steps` must not be in `learning_rate_schedule_kwargs`.')

  if learning_rate_schedule_name == 'constant':
    return constant_learning_rate_schedule(initial_learning_rate)
  elif learning_rate_schedule_name == 'piecewise_constant_progress_aligned':
    return piecewise_constant_progress_aligned_learning_rate_schedule(
        initial_learning_rate, max_steps, **learning_rate_schedule_kwargs)
  elif learning_rate_schedule_name == 'warmup_cosine_decay':
    return warmup_cosine_decay_learning_rate_schedule(
        initial_learning_rate, steps_per_epoch, max_steps,
        **learning_rate_schedule_kwargs)
  else:
    raise ValueError(
        f'Unsupported `learning_rate_schedule_name` = `{learning_rate_schedule_name}`'
    )
