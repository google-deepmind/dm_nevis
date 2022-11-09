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

"""Metrics for the LSCL project."""
# TODO: add back the test file

import collections
import dataclasses
import io
import os
from typing import Callable, Iterator, Mapping, Optional, Sequence, Union

from absl import logging
from dm_nevis.benchmarker.datasets import datasets
from dm_nevis.benchmarker.datasets import streams
from dm_nevis.benchmarker.datasets import tasks
from dm_nevis.benchmarker.environment import datawriter_interface
from dm_nevis.benchmarker.learners import learner_interface
from dm_nevis.benchmarker.metrics import classification_metrics
from dm_nevis.benchmarker.metrics import metrics_aggregators
from dm_nevis.benchmarker.metrics import multi_label_classification_metrics
import numpy as np
from tensorflow.io import gfile


KNOWN_SPLIT_SUFFICES = frozenset([
    "train",
    "dev",
    "train_and_dev",
    "dev_test",
    "test",
])
UNKNOWN_SPLIT_NAME = "unknown_split"

DEFAULT_OUTPUT_DIR = os.environ.get("NEVIS_OUTPUT_DIR", "/tmp/nevis_output_dir")


@dataclasses.dataclass(frozen=True)
class PredictionMetrics:
  event: streams.PredictionEvent
  task: tasks.TaskKey
  metrics: Union[
      classification_metrics.ClassificationMetrics,
      multi_label_classification_metrics.MultiLabelClassificationMetrics]


@dataclasses.dataclass(frozen=True)
class TrainMetrics:
  event: streams.TrainingEvent
  task: tasks.TaskKey
  resources_used: learner_interface.ResourceUsage


@dataclasses.dataclass(frozen=True)
class LSCLMetricsState:
  """The metrics state for this aggregator.

  We maintain a 1-to-1 relationship between events in the stream and entries
  in the (ordered) sequence of metrics objects.
  """
  predictions_dir: str
  metrics: Sequence[Union[PredictionMetrics, TrainMetrics]]


def lscl_metrics(
    dataset_lookup: Callable[[streams.DatasetKey], datasets.Dataset],
    metrics_writer: datawriter_interface.DataWriter
) -> metrics_aggregators.MetricsAggregator:
  """Returns a metrics aggregator for the LSCL stream.

  This aggregator computes common classification metrics for every
  prediction event in the stream. Once the stream has finished, the aggregator
  will fetch the final computed metrics for each task, and then compute an
  overall normalized accuracy for each of these final computed metrics,
  normalized by the total number of examples.

  Args:
    dataset_lookup: A callable to retrieve datasets given the dataset key.
    metrics_writer: A pipe to write debug metrics to. This will be written each
      time aggregate is called.

  Returns:
    A metrics aggregator for use in the LSCL stream and with the benchmarker.
  """

  def init() -> LSCLMetricsState:
    logging.info("Initializing metrics")
    predictions_dir = _create_output_dir()
    logging.info("Writing raw predictions to %s", predictions_dir)

    return LSCLMetricsState(
        predictions_dir=predictions_dir,
        metrics=[],
    )

  def aggregate_train_event(
      state: LSCLMetricsState,
      event: streams.TrainingEvent,
      resources_used: learner_interface.ResourceUsage,
  ) -> LSCLMetricsState:
    task_key = dataset_lookup(event.dev_dataset_key).task_key
    return dataclasses.replace(
        state,
        metrics=[
            *state.metrics,
            TrainMetrics(event, task_key, resources_used),
        ],
    )

  def aggregate_predict_event(
      state: LSCLMetricsState,
      event: streams.PredictionEvent,
      predictions: Iterator[learner_interface.Predictions],
  ) -> LSCLMetricsState:

    resources_used = _combined_train_resources_used(state)
    dataset = dataset_lookup(event.dataset_key)
    task = dataset.task_key
    task_kind = task.kind

    outdir = os.path.join(
        state.predictions_dir,
        f"event_{len(state.metrics)}",
    )

    if not gfile.exists(outdir):
      gfile.makedirs(outdir)

    path = os.path.join(outdir, "raw_predictions.npz")

    with WrappedPredictionsWriter(predictions, path=path, task=task) as wrapped:
      if task_kind == tasks.TaskKind.CLASSIFICATION:
        metrics = classification_metrics.compute_metrics(wrapped)
      elif task_kind == tasks.TaskKind.MULTI_LABEL_CLASSIFICATION:
        metrics = multi_label_classification_metrics.compute_metrics(wrapped)
      else:
        raise NotImplementedError(f"Unsupported task kind: {task_kind}.")

    payload = {
        "raw_predictions_and_targets_path": path,
        "stream_index": len(state.metrics),
        "index_of_most_recent_train_event": _num_train_events(state) - 1,
        "task_name": task.name,
        "task_kind": str(task.kind),
        "dataset_key": str(event.dataset_key),
        "data_split": _try_to_extract_split(event.dataset_key),
        "cumulative_train_flops_used": resources_used.floating_point_operations,
        "peak_parameter_count": resources_used.peak_parameter_count,
        "peak_parameter_size_bytes": resources_used.peak_parameter_size_bytes,
        **metrics._asdict(),
    }

    logging.info("Metrics for task %s: %s", task.name, payload)
    metrics_writer.write(payload)
    metrics_writer.flush()

    return dataclasses.replace(
        state,
        metrics=[*state.metrics,
                 PredictionMetrics(event, task, metrics)],
    )

  return metrics_aggregators.MetricsAggregator(init, aggregate_train_event,
                                               aggregate_predict_event,
                                               _compute_results)


def _compute_results(state: LSCLMetricsState) -> metrics_aggregators.Results:
  """Compute statistics over the stream."""

  prediction_metrics_by_split = _extract_prediction_metrics_by_split(state)
  results = {}

  for split, metrics in prediction_metrics_by_split.items():
    single_label_results = _compute_single_label_results(metrics)
    multi_label_results = _compute_multi_label_results(metrics)

    for key, value in single_label_results.items():
      results[f"{split}_{key}"] = value

    for key, value in multi_label_results.items():
      results[f"{split}_{key}"] = value

  return results


def _extract_prediction_metrics_by_split(
    state: LSCLMetricsState) -> Mapping[str, Sequence[PredictionMetrics]]:
  """Separates out the predict metrics by dataset split name."""
  predict_metrics_by_split = collections.defaultdict(list)

  for m in state.metrics:
    if not isinstance(m, PredictionMetrics):
      continue

    split = _try_to_extract_split(m.event.dataset_key) or UNKNOWN_SPLIT_NAME
    predict_metrics_by_split[split].append(m)

  return dict(predict_metrics_by_split)


def _compute_single_label_results(
    metrics: Sequence[PredictionMetrics]) -> metrics_aggregators.Results:
  """Compute results for single class case."""

  num_events = 0
  top_one_correct = 0
  num_examples = 0
  total_accuracy = 0.0

  for m in metrics:
    if not isinstance(m.metrics, classification_metrics.ClassificationMetrics):
      continue

    num_events += 1
    total_accuracy += m.metrics.top_one_accuracy
    top_one_correct += m.metrics.top_one_correct
    num_examples += m.metrics.num_examples

  if num_examples == 0:
    weighted_accuracy = float("nan")
    accuracy = float("nan")
  else:
    weighted_accuracy = top_one_correct / num_examples
    accuracy = total_accuracy / num_events

  return {
      "weighted_average_single_label_accuracy": weighted_accuracy,
      "average_single_label_accuracy": accuracy,
  }


def _compute_multi_label_results(
    metrics: Sequence[PredictionMetrics]) -> metrics_aggregators.Results:
  """Compute results for multi label case."""

  num_events = 0
  total_mean_average_precision = 0.0

  for m in metrics:
    if not isinstance(
        m.metrics,
        multi_label_classification_metrics.MultiLabelClassificationMetrics):
      continue

    num_events += 1
    total_mean_average_precision += m.metrics.mean_average_precision

  if num_events == 0:
    mean_mean_average_precision = float("nan")
  else:
    mean_mean_average_precision = total_mean_average_precision / num_events

  # TODO: Find a better way to combine mAP.
  return {
      "average_multi_label_mean_average_precision": mean_mean_average_precision
  }


def _combined_train_resources_used(
    state: LSCLMetricsState) -> learner_interface.ResourceUsage:
  """Computes total train resources used so far."""
  result = None

  for m in state.metrics:
    if not isinstance(m, TrainMetrics):
      continue

    if result is None:
      result = m.resources_used
    else:
      result = result.combine(m.resources_used)

  if result is None:
    return learner_interface.ResourceUsage()

  return result


def _num_train_events(state: LSCLMetricsState) -> int:
  return sum(1 for m in state.metrics if isinstance(m, TrainMetrics))


def _try_to_extract_split(dataset_key: str) -> Optional[str]:
  """Attempts to compute the split from the dataset key.

  For the LSCL stream, the dataset splits are stored at the end of the dataset
  key, as `<dataset_name>_<split>`.

  Args:
    dataset_key: The key to try and compute the split for.

  Returns:
    The split name, or None if no match was found.
  """
  suffices_by_length = sorted(KNOWN_SPLIT_SUFFICES, key=lambda x: -len(x))

  for suffix in suffices_by_length:
    if dataset_key.endswith("_" + suffix):
      return suffix

  return None


def _create_output_dir() -> str:
  result = os.path.join(DEFAULT_OUTPUT_DIR, "predictions")
  if not gfile.exists(result):
    gfile.makedirs(result)
  return result


class WrappedPredictionsWriter:
  """A writer for storing raw predictions to an output file.

  This writer wraps a prediction iterator and copies the raw outputs
  and targets in memory. When the context managed by this object is closed,
  the raw data is concatenated together into a single numpy array, and then
  written into a multipart numpy array to the output path.
  """

  def __init__(self, predictions: Iterator[learner_interface.Predictions], *,
               path: str, task: tasks.TaskKey):

    if task.kind not in {
        tasks.TaskKind.CLASSIFICATION,
        tasks.TaskKind.MULTI_LABEL_CLASSIFICATION,
    }:
      raise ValueError("Cannot save predictions for unsupported task: {task}")

    self._task = task
    self._path = path
    self._iter = predictions
    self._raw_targets = []
    self._raw_outputs = []

  def __enter__(self):
    return self

  def __exit__(self, *args, **kwargs):
    del args, kwargs

    if not self._raw_outputs:
      logging.warning("Skipping writing empty predictions...")
      return

    logging.info("Writing targets and outputs to local files...")
    targets = np.concatenate(self._raw_targets, axis=0)
    outputs = np.concatenate(self._raw_outputs, axis=0)

    # https://github.com/tensorflow/tensorflow/issues/32090#issuecomment-986135710
    io_buffer = io.BytesIO()
    np.savez(io_buffer, targets=targets, outputs=outputs)
    with gfile.GFile(self._path, "wb") as outfile:
      logging.info("Writing raw targets and outputs to %s", self._path)
      outfile.write(io_buffer.getvalue())
      logging.info("Finished writing raw targets and outputs.")

  def __iter__(self):
    return self

  def __next__(self):
    prediction = next(self._iter)

    if self._task.kind is tasks.TaskKind.CLASSIFICATION:
      targets, outputs = prediction.batch.label, prediction.output
      self._raw_targets.append(targets)
      self._raw_outputs.append(outputs[0])
    elif self._task.kind is tasks.TaskKind.MULTI_LABEL_CLASSIFICATION:
      targets, outputs = prediction.batch.multi_label_one_hot, prediction.output
      self._raw_targets.append(targets)
      self._raw_outputs.append(np.stack(outputs, axis=1))
    else:
      raise ValueError(f"Unsupported task: {self._task.kind}")

    return prediction
