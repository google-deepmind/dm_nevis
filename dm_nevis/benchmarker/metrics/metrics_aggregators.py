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

"""Aggregation of metrics in the benchmarker.

The ultimate goal of running the benchmarker is to publish metrics computed
from the prediction events from a task stream. This package provides a uniform
interface for defining the metrics that may be computed, and also provides
standard implementations for common metrics that may be used.

Metrics in the benchmarker are manipulated using a stateless dataclass of
pure functions called the MetricsAggregator. Each time a prediction event
is encountered by the environment, the predictions are fed to the metrics
aggregator, along with the prior state stored by the metrics aggregator.
This allows the metrics aggregator to keep track of all statistics over all
prediction tasks that the environment has encountered.

At the end of a benchmark run, the metrics aggregator's "compute_results"
function will be called with the state that the metrics aggregator has
accumulated up to the current point. It is in the compute_results function
that the metrics aggregator may compute a "final" statistical summary over every
prediction event that occurred in the whole task stream may be summarized, and
returned. For full generality, the result is allowed to be any pytree.

The environment commits to running aggregate exactly once for each prediction
event that is encountered, and the metrics aggregator is allowed to log any
intermediate metrics that it wishes, to allow "online" debugging of model
progresss, and intermediate results to be visualized before a full task stream
run has been completed.
"""

import dataclasses
from typing import Callable, Iterator

import chex
from dm_nevis.benchmarker.datasets import streams
from dm_nevis.benchmarker.learners import learner_interface


State = chex.ArrayTree
Results = chex.ArrayTree
InitFn = Callable[[], State]
AggregateTrainEventFn = Callable[
    [State, streams.TrainingEvent, learner_interface.ResourceUsage], State]
AggregatePredictEventFn = Callable[
    [State, streams.PredictionEvent, Iterator[learner_interface.Predictions]],
    State]
ComputeResultsFn = Callable[[State], Results]


@dataclasses.dataclass
class MetricsAggregator:
  """Metrics class collecting together pure functions for manipulating metrics.

  Similarly to other JAX libraries, this class does not contain state
  internally, but rather provides pure functions that expclitly manipulate the
  state. The state may be initialized by calling the init() function.

  Attributes:
    init: A function to initialize the metrics state.
    aggregate_train_event: A function to combine together an existing state and
      new predictions, for a given training event.
    aggregate_predict_event: A function to combine together existing state and
      predictions.
    compute_results: A function to compute the results given the state observed
      up to this point is assumed to be called once all data has been added to
      the state.
  """
  init: InitFn
  aggregate_train_event: AggregateTrainEventFn
  aggregate_predict_event: AggregatePredictEventFn
  compute_results: ComputeResultsFn


def noop_metrics_aggregator() -> MetricsAggregator:
  """Creates a metrics aggregator that does nothing."""

  def init():
    return None

  def aggregate_train_event(state, event, resources_used):
    del event, resources_used
    return state

  def aggregate_predict_event(state, event, predictions):
    del event

    for _ in predictions:
      continue

    return state

  def compute_results(state):
    del state
    return {}

  return MetricsAggregator(init, aggregate_train_event, aggregate_predict_event,
                           compute_results)
