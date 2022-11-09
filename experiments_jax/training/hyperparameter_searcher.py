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

"""Hyperparameter tuner library.

This library is intended for executing multiple long running cost functions
within workers. Each time a cost function completes, the resulting cost is
compared to the values that have been computed so far, and ultimately the lowest
cost solution is returned.

This library contains a minimum interface require to distribute computation,
and currently uses concurrent.Executor instances to manage tasks. To generalize
to distribute compute, a failure-robust executor with the ability to schedule
on remote machines would be needed.

The cost function being executed by each worker supports checkpointing. These
checkpoints are written to a checkpoint store that is independent from the main
thread maintaining the minimum cost results found so far. We can safely do this
due to the idempotency of the workers - calls to the cost function are
stateless, and so the fact that they start from scratch or resume from an
checkpoint can be totally hidden from the code that is scheduling the job. From
the perspective of the top-level search, tasks are scheduled and eventually
complete. To ensure that the tasks and their checkpoints are all kept in sync
with the learner, each search has a unique `search_id` that is stored along with
all checkpoints (both in the top level search process, and in the workers).
"""

from concurrent import futures
import dataclasses
import json
import os
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Set, Protocol
import uuid

from absl import logging
from dm_nevis.benchmarker.environment import checkpointer_interface
from dm_nevis.benchmarker.learners import learner_interface


DEFAULT_TOPIC_NAME = 'default'
DEEPMIND_A100_DEVICES_PER_NODE = 16
DEFAULT_RETRIES_ON_FAILURE = 3

Context = Any
Outputs = Any
Cost = Any
Checkpoint = Any
CheckpointFn = Callable[[Checkpoint], None]
Overrides = Dict[str, Any]

# Note(rhemsley): Checkpointing can take several minutes, and blocks
# receiving of new tasks, we thus disable checkpointing of responses by default,
# since it can delay the running of the searcher by a very long time.
# It may make sense to enable this for the case that jobs take very different
# amounts of time.
SHOULD_CHECKPOINT_RESPONSES = False


class CostFn(Protocol):

  def __call__(self,
               context: Context,
               overrides: Overrides,
               *,
               write_checkpoint: CheckpointFn = lambda _: None,
               checkpoint_to_resume: Optional[Checkpoint] = None) -> Cost:
    ...


CostFunctionBuilder = Callable[[], CostFn]
CheckpointerBuilder = Callable[[str], checkpointer_interface.Checkpointer]


@dataclasses.dataclass
class SearchResult:
  cost: Cost
  outputs: Outputs
  overrides: Overrides
  resources_used: learner_interface.ResourceUsage


class HyperParameterTunerWorkers(NamedTuple):
  queue: Any


class JobKey(NamedTuple):
  """A key to uniquely identify a job in a hyperparameter search."""
  search_id: str
  job_index: int

  def to_str(self) -> str:
    return json.dumps({
        'search_id': self.search_id,
        'job_index': self.job_index,
    })

  @classmethod
  def from_str(cls, v: str) -> 'JobKey':
    dct = json.loads(v)
    return cls(**dct)


@dataclasses.dataclass(frozen=True)
class Task:
  context: Context
  overrides: Overrides


@dataclasses.dataclass(frozen=True)
class HyperParameterTunerState:
  search_id: str
  min_cost: Optional[float]
  min_cost_overrides: Optional[Overrides]
  min_cost_outputs: Optional[Outputs]
  completed_job_keys: Set[JobKey]
  search_space: Sequence[Overrides]
  total_resources_used: learner_interface.ResourceUsage


class HyperparameterTuner:
  """A hyperparameter tuner using a pool of workers."""

  def __init__(self, workers: HyperParameterTunerWorkers):
    self._queue = workers.queue

  def minimize(self,
               context: Context,
               search_space: Sequence[Overrides],
               *,
               checkpoint_to_resume: Optional[Checkpoint] = None,
               write_checkpoint: CheckpointFn = lambda _: None) -> SearchResult:
    """Minimizes the cost function over the search space using workers.

    The tuner supports checkpointing and resuming the cost function tasks
    using the 1) the write_checkpoint() callback - which allows the running
    hyper parameter tuner to store checkpointed state, and 2) The
    checkpoint_to_resume value, which can be set with the most recently written
    value that was written to write_checkpoint before failure occurred.

    Args:
      context: An arbitrary (picklable) state object to pass to the cost
        function. This is for passing values to the cost function that do not
        change in the search space.
      search_space: The space over which to search. All configuration to be
        passed to thecost function must be passed through here. The search space
        must be deterministic, otherwise preemptions would cause all progress
        made running training jobs to be lost.
      checkpoint_to_resume: Resume a previously checkpointed hyperparameter
        search.
      write_checkpoint: A callable to checkpoint the progress of this search.

    Returns:
      The minimum cost obtained over the results. Note that the result is
      non-deterministic.
    """

    state = HyperParameterTunerState(
        search_id=uuid.uuid4().hex,
        min_cost=None,
        min_cost_overrides=None,
        min_cost_outputs=None,
        completed_job_keys=set(),
        search_space=tuple(search_space),
        total_resources_used=learner_interface.ResourceUsage(
            floating_point_operations=0,
            peak_parameter_count=0,
            peak_parameter_size_bytes=0),
    )

    if checkpoint_to_resume:
      state = checkpoint_to_resume
      logging.info('Resuming checkpointed search `%s`...', state.search_id)
      assert isinstance(state, HyperParameterTunerState)
    else:
      # We checkpoint the initial state, to ensure a deterministic search space.
      logging.info('Starting new search `%s`...', state.search_id)
      write_checkpoint(state)

    futures_to_job_key = {}
    for i, overrides in enumerate(search_space):
      job_key = _make_job_key(state.search_id, i)

      if job_key in state.completed_job_keys:
        logging.info('Skipping previously completed job: %s', job_key)
        continue

      logging.info('Enqueueing job: %s, overrides: %s', job_key, overrides)
      task = Task(context, overrides)
      future = self._queue.enqueue_task(job_key, task)
      futures_to_job_key[future] = job_key

    for future in futures.as_completed(futures_to_job_key):
      job_key = futures_to_job_key[future]
      result: SearchResult = future.result()

      logging.info(
          'Received result for task: %s (%d/%d), cost: %s',
          job_key,
          len(state.completed_job_keys) + 1,
          len(futures_to_job_key),
          result.cost,
      )

      state = dataclasses.replace(
          state,
          total_resources_used=state.total_resources_used.combine(
              result.resources_used),
          completed_job_keys=state.completed_job_keys | set([job_key]),
      )

      if state.min_cost is None or result.cost < state.min_cost:
        state = dataclasses.replace(
            state,
            min_cost=result.cost,
            min_cost_outputs=result.outputs,
            min_cost_overrides=result.overrides,
        )

      if SHOULD_CHECKPOINT_RESPONSES:
        write_checkpoint(state)

    assert isinstance(state, HyperParameterTunerState)

    logging.info('Minimum of %s achieved with %s', state.min_cost,
                 state.min_cost_overrides)

    return SearchResult(
        cost=state.min_cost,
        outputs=state.min_cost_outputs,
        overrides=state.min_cost_overrides,
        resources_used=state.total_resources_used,
    )


def default_checkpointer_builder(
    job_namespace: str) -> checkpointer_interface.Checkpointer:
  """Builds a checkpointer that does nothing."""

  class NoOpCheckpointer:

    def write(self, state):
      del state

    def restore(self):
      return None

  del job_namespace
  return NoOpCheckpointer()


def build_local_executor_workers(
    cost_function_builder: CostFunctionBuilder,
    *,
    executor: Callable[..., futures.Executor],
    num_workers: int,
    checkpointer_builder: CheckpointerBuilder = default_checkpointer_builder,
) -> HyperParameterTunerWorkers:
  """Constructs workers that operate locally in a pool of processes."""

  class LocalProcessPoolQueue:
    """Wrap a process pool to give a local task queue."""

    def __init__(self, max_workers: int,
                 cost_function_builder: CostFunctionBuilder,
                 checkpointer_builder: CheckpointerBuilder):

      self._pool = executor(max_workers=max_workers)
      self._cost_function_builder = cost_function_builder
      self._checkpointer_builder = checkpointer_builder

    def close(self):
      self._pool.shutdown()

    def enqueue_task(self, job_key, task):
      return self._pool.submit(_worker, job_key, task,
                               self._cost_function_builder,
                               self._checkpointer_builder)

  return HyperParameterTunerWorkers(
      LocalProcessPoolQueue(
          max_workers=num_workers,
          cost_function_builder=cost_function_builder,
          checkpointer_builder=checkpointer_builder,
      ))


def _worker(
    job_key: JobKey,
    task: Task,
    cost_function_builder: CostFunctionBuilder,
    checkpointer_builder: CheckpointerBuilder,
) -> SearchResult:
  """Worker function to be executed on worker nodes."""

  cost_function = cost_function_builder()

  logging.info('[Process %s] Received task %s: %s', os.getpid(), job_key,
               task.overrides)
  job_namespace = _job_key_to_namespace(job_key)
  checkpointer = checkpointer_builder(job_namespace)

  cost, outputs, resources_used = cost_function(
      task.context,
      task.overrides,
      checkpoint_to_resume=checkpointer.restore(),
      write_checkpoint=checkpointer.write)

  logging.info('Completed task %s: cost: %s', job_key, cost)
  return SearchResult(cost, outputs, task.overrides, resources_used)


def _make_job_key(search_id: str, job_index: int) -> JobKey:
  return JobKey(
      search_id=search_id,
      job_index=job_index,
  )


def _job_key_to_namespace(job_key: JobKey) -> str:
  return f'hyperparameter_search_{job_key.search_id}_job_{job_key.job_index}'
