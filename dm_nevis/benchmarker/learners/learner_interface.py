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

"""Interface for learners."""

import dataclasses
from typing import Any, Callable, Iterator, NamedTuple, Optional, Tuple, Protocol

from dm_nevis.benchmarker.datasets import datasets
from dm_nevis.benchmarker.datasets import streams


LearnerState = Any
Checkpoint = Any
CheckpointFn = Callable[[Checkpoint], None]


@dataclasses.dataclass
class ResourceUsage:
  """Estimated resources used by a learner.

  All attributes are optional and default to None, since they may be either
  inappropriate or unavailable depending on the specific implementation details
  of the learner.

  Floating point values are used to measure flops since these may overflow
  integers. Since these are typically estimates, the inherent inprecision
  may be ignored.

  Attributes:
    floating_point_operations: An estimate of the number of floating point
      operations used, of any precision. Sometimes referred to as FLOPs. We
      avoid this acronym due to the ambiguity with FLOPS (floating point
      operations per second).
    peak_parameter_count: The peak number of parameters used by the learner.
    peak_parameter_size_bytes: The peak number of bytes used to store the
      learner's parameters.
  """
  floating_point_operations: Optional[float] = None
  peak_parameter_count: Optional[int] = None
  peak_parameter_size_bytes: Optional[int] = None

  def combine(self, other: 'ResourceUsage') -> 'ResourceUsage':
    """Combines with other resource usage dataclasses and return a new one..

    Args:
      other: Resources to be combined with `self`.

    Returns:
      Accumulated resource usage. If any values are None in the inputs, then the
      return values will be set to None.
    """

    def add_or_none(x, y):
      if x is None or y is None:
        return None
      return x + y

    def max_or_none(x, y):
      if x is None or y is None:
        return None
      return max(x, y)

    return ResourceUsage(
        floating_point_operations=add_or_none(self.floating_point_operations,
                                              other.floating_point_operations),
        peak_parameter_count=max_or_none(self.peak_parameter_count,
                                         other.peak_parameter_count),
        peak_parameter_size_bytes=max_or_none(self.peak_parameter_size_bytes,
                                              other.peak_parameter_size_bytes),
    )


class Predictions(NamedTuple):
  """Input batch and resulting learner predictions.

  TODO: Implement a specific type for the returned output.

  In all cases, the batch and output are assumed to have a single batch
  dimension that is identical between the abtch and output attributes.

  Attributes:
    batch: The verbatim input batch used to compute the predictions.
    output: The outputs resulting from running prediction on the batch.
  """
  batch: datasets.MiniBatch
  output: Any


class InitFn(Protocol):

  def __call__(self) -> LearnerState:
    """Initializes a learner's state."""


class TrainFn(Protocol):

  def __call__(
      self,
      event: streams.TrainingEvent,
      state: LearnerState,
      write_checkpoint: CheckpointFn,
      *,
      checkpoint_to_resume: Optional[Checkpoint] = None,
  ) -> Tuple[LearnerState, ResourceUsage]:
    """Trains a learner with the given state, and returns updated state."""


class PredictFn(Protocol):

  def __call__(
      self,
      event: streams.PredictionEvent,
      state: LearnerState,
  ) -> Iterator[Predictions]:
    """Computes predictions."""


class Learner(NamedTuple):
  init: InitFn
  train: TrainFn
  predict: PredictFn
