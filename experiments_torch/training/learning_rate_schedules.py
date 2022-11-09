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
# TODO: scheduler should be multiplier or setter? wbt frozen?
# pylint: disable=protected-access
# pytype: disable=attribute-error
from typing import Any, Dict, Mapping, NamedTuple, Optional, Sequence

from absl import logging
from torch import optim
from torch.optim import lr_scheduler


class ProgressAndScale(NamedTuple):
  """Combines a progress and scale.

  Attributes:
    progress: the progress (in the range [0, 1] through the full train loop).
    scale: The learning rate scaling to apply once the given progress has been
      completed.
  """
  progress: float
  scale: float


class MultiplicativeLR(lr_scheduler._LRScheduler):
  """Piece-wise constant learning rate depending on learning progress.

  Attributes:
    optimizer: Wrapped optimizer.
    initial_learning_rate: float
    max_steps: Maximum number of training steps (batched weight updates).
    boundaries_and_scale: A sequence of tuples `(progress, scale)`, where
      `progress` indicates a step of `max_steps` at which learning rate is
      scaled by `scale`.

  Returns:
    Learning rate schedule function.
  """

  def __init__(self, optimizer: optim.Optimizer, initial_learning_rate: float,
               boundaries_and_scales: Mapping[float, float]):
    self.initial_learning_rate = initial_learning_rate
    self.boundaries_and_scales = boundaries_and_scales
    super().__init__(optimizer)

  def get_lr(self):
    prev_scale, scale = 1.0, 1.0
    for boundary, scale in self.boundaries_and_scales.items():
      if self._step_count < boundary:
        break
    prev_scale = scale

    return [
        self.initial_learning_rate * prev_scale if group['lr'] != 0. else 0.
        for group in self.optmizer.param_groups
    ]


def piecewise_constant_progress_aligned_learning_rate_schedule(
    optimizer: optim.Optimizer,
    initial_learning_rate: float,
    max_steps: int,
    learning_progress_boundaries_and_scales: Sequence[ProgressAndScale],
) -> lr_scheduler._LRScheduler:
  """Piece-wise constant learning rate depending on learning progress.

  Args:
    optimizer: Wrapped optimizer.
    initial_learning_rate: float
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

  return MultiplicativeLR(optimizer, initial_learning_rate,
                          boundaries_and_scales)


class ConstantLR(lr_scheduler._LRScheduler):
  """Constant learning rate schedule.

  # Args:
    optimizer: wrapper optimizer.
    init_value: constant learning rate value.
  """

  def __init__(self, optimizer: optim.Optimizer, init_value: float):
    super().__init__(optimizer)
    self.init_value = init_value

  def get_lr(self):
    return [self.init_value for _ in self.optimizer.param_groups]


def constant_learning_rate_schedule(
    optimizer: optim.Optimizer, init_value: float) -> lr_scheduler._LRScheduler:
  """Constant learning rate schedule."""
  return ConstantLR(optimizer, init_value)


class GradualWarmupScheduler(lr_scheduler._LRScheduler):
  """Warmup scheduler followed by any existing scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        end_warmup_lr: final learning rate after warmup.
        warmup_steps: total number of warmup steps
        after_scheduler: after warmup_steps, use this scheduler (eg.
          ReduceLROnPlateau)
  """

  def __init__(self,
               optimizer: optim.Optimizer,
               end_warmup_lr: float,
               warmup_steps: int,
               after_scheduler: Optional[lr_scheduler._LRScheduler] = None):
    self.end_warmup_lr = end_warmup_lr
    self.warmup_steps = warmup_steps
    self.after_scheduler = after_scheduler
    super().__init__(optimizer)

  def get_lr(self):
    if self._step_count < self.warmup_steps:
      return [
          self._step_count *
          (self.end_warmup_lr / self.warmup_steps) if group['lr'] != 0. else 0.
          for group in self.optimizer.param_groups
      ]
    return self.after_scheduler.get_lr()

  def step(self, epoch=None):
    if self._step_count < self.warmup_steps:
      super(GradualWarmupScheduler, self).step(epoch)
    else:
      self.after_scheduler.step(epoch)


def warmup_cosine_decay_learning_rate_schedule(
    optimizer: optim.Optimizer, initial_learning_rate: float,
    steps_per_epoch: int, max_steps: int, warmup_epochs: int,
    final_learning_rate: float) -> lr_scheduler._LRScheduler:
  """Warmup cosine learning rate schedule."""

  # The warmup steps must be strictly less than the number of overall steps.
  warmup_steps = min(max_steps - 1, warmup_epochs * steps_per_epoch)

  logging.info(
      'Cosine decay schedule: warmup: %d, max steps: %d',
      warmup_steps,
      max_steps,
  )
  return GradualWarmupScheduler(
      optimizer,
      end_warmup_lr=initial_learning_rate,
      warmup_steps=warmup_steps,
      after_scheduler=lr_scheduler.CosineAnnealingLR(
          optimizer, T_max=max_steps, eta_min=final_learning_rate))


def build_learning_rate_schedule(
    optimizer: optim.Optimizer, learning_rate_schedule_name: str,
    initial_learning_rate: float, steps_per_epoch: int, max_steps: int,
    learning_rate_schedule_kwargs: Dict[str, Any]) -> lr_scheduler._LRScheduler:
  """Creates a learning_rate_schedule function for given arguments.

  This function assumes that `steps_per_epoch` and `max_steps` are not contained
  in `learning_rate_schedule_kwargs`. The reason for this constraint is due to
  the fact that these arguments could be dynamically recomputed on the learner
  side depending on which dataset is used.

  Args:
    optimizer: Wrapped optimizer.
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
    return constant_learning_rate_schedule(optimizer, initial_learning_rate)
  elif learning_rate_schedule_name == 'piecewise_constant_progress_aligned':
    return piecewise_constant_progress_aligned_learning_rate_schedule(
        optimizer, max_steps, **learning_rate_schedule_kwargs)
  elif learning_rate_schedule_name == 'warmup_cosine_decay':
    return warmup_cosine_decay_learning_rate_schedule(
        optimizer, initial_learning_rate, steps_per_epoch, max_steps,
        **learning_rate_schedule_kwargs)
  else:
    raise ValueError(
        f'Unsupported `learning_rate_schedule_name` = `{learning_rate_schedule_name}`'
    )
