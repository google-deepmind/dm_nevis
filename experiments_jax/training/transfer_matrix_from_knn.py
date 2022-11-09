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

"""A module to compute a transfer matrix using a KNN classifier."""

import datetime
import time
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple

from absl import logging
from dm_nevis.benchmarker.datasets import datasets
from dm_nevis.benchmarker.datasets import tasks
from dm_nevis.benchmarker.environment import datawriter_interface
from experiments_jax.training import evaluate_embeddings
from experiments_jax.training import transfer_oracle
import numpy as np


def compute_transfer_matrix_using_knn_classifier(
    embedding_fn: evaluate_embeddings.EmbeddingFn,
    tasks_and_train_states: Sequence[Tuple[
        tasks.TaskKey, evaluate_embeddings.EmbeddingFnState]],
    train_dataset: datasets.Dataset,
    test_dataset: datasets.Dataset,
    *,
    batch_size: int,
    preprocessing_fn: Callable[[datasets.MiniBatch], datasets.MiniBatch],
) -> transfer_oracle.TransferMatrix:
  """Computes a single-column transfer matrix using an embedding function.

  Args:
    embedding_fn: A callable that computes embeddings for examples given a
      state.
    tasks_and_train_states: The viable tasks and train states that can be
      transferred from.
      train_dataset: The dataset to use for training a model using the
        embedding.
      test_dataset: The dataset to use to test the embeddings.
      batch_size: The batch size to use when computing the embeddings.
      preprocessing_fn: A function to map to the datasets (e.g. image resize.)
        before calling embed.

  Returns:
    A transfer matrix estimated from the input train states. Note that the
    transfer matrix only has a single column, corresponding to task associated
    with the train and test dataset.
  """

  task_keys = [t[0] for t in tasks_and_train_states]
  states = [t[1] for t in tasks_and_train_states]

  logging.info('Evaluating transfer using KNN for %d states', len(task_keys))

  start_time = time.monotonic()
  results = evaluate_embeddings.evaluate(
      embedding_fn,
      states=states,
      train_dataset=train_dataset,
      test_dataset=test_dataset,
      batch_size=batch_size,
      preprocessing_fn=preprocessing_fn,
  )
  elapsed = datetime.timedelta(seconds=(time.monotonic() - start_time))
  _log_summary(elapsed, train_dataset.task_key, results, task_keys)

  # Note(rhemsley): The transfer matrix uses smaller values to indicate better
  # transfer, so subtract all weights from 1 here.
  matrix = np.array([[1 - r.weight] for r in results])

  return transfer_oracle.TransferMatrix(
      source_tasks=task_keys,
      target_tasks=[train_dataset.task_key],
      matrix=matrix,
  )


def publish_transfer_matrix(
    metrics: datawriter_interface.DataWriter,
    matrix: transfer_oracle.TransferMatrix,
    *,
    extra_context: Optional[Mapping[str, Any]],
) -> None:
  """Publishes transfer values to a metrics writer."""

  logging.info('Publishing computed transfer values to metrics writer...')

  target_task, *other_target_tasks = matrix.target_tasks
  if other_target_tasks:
    raise ValueError(
        f'Expected only a single target task, got {matrix.target_tasks}')

  extra_context = extra_context or {}

  for i, (source_task, weight) in enumerate(matrix.transfer_tasks(target_task)):
    metrics.write({
        'target_task_name': target_task.name,
        'source_task_name': source_task.name,
        'source_task_column_index': i,
        'weight': weight,
        **extra_context,
    })

  metrics.flush()


def _log_summary(elapsed: datetime.timedelta, target_task: tasks.TaskKey,
                 results: Sequence[evaluate_embeddings.EvaluationResult],
                 task_keys: Sequence[tasks.TaskKey]) -> None:
  """Logs the results of the KNN transfer weightings."""

  sorted_tasks = sorted((r.weight, task, i)
                        for i, (r, task) in enumerate(zip(results, task_keys)))

  headers = ['Weight (higher is better)', 'Source Task', 'Original Index']
  logging.info(
      'Computed transfer matrix in %s for task %s\n%s\n',
      elapsed,
      target_task,
      _tabulate(reversed(sorted_tasks), headers),
  )


def _tabulate(elements: Iterable[Sequence[Any]], headers: Iterable[Any]) -> str:
  """Builds an ASCII table of results."""

  # TODO: Solve why we can't easily depend on tabulate, due to
  # visibility issues and global tap failures.
  headers = list(headers)
  elements = list(elements)

  max_length_by_col = [len(str(h)) for h in headers]

  for row in elements:
    for i, r in enumerate(row):
      max_length_by_col[i] = max(max_length_by_col[i], len(str(r)))

  def format_line(values, max_length_by_col):
    line = '|'
    for v, n in zip(values, max_length_by_col):
      line += f' {v} '.ljust(n + 3) + '|'
    return line

  length = (sum(max_length_by_col) + 1 + 4 * len(max_length_by_col))

  lines = []
  lines.append('-' * length)
  lines.append(format_line(headers, max_length_by_col))
  lines.append('-' * length)
  for row in elements:
    lines.append(format_line(row, max_length_by_col))
  lines.append('-' * length)

  return '\n'.join(lines)
