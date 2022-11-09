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

"""A module for evaluating embeddings using a KNN classifier."""

import dataclasses
from typing import Any, Callable, Sequence, Tuple

from absl import logging
import chex
from dm_nevis.benchmarker.datasets import datasets
from dm_nevis.benchmarker.datasets import tasks
import numpy as np
import sklearn.neighbors
import tensorflow as tf
import tensorflow_datasets as tfds

EmbeddingFnState = Any
EmbeddingFn = Callable[[EmbeddingFnState, datasets.MiniBatch], np.ndarray]

DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_TRAIN_DATASET_SIZE = 10_000
DEFAULT_MAX_TEST_DATASET_SIZE = 5_000


@dataclasses.dataclass(frozen=True)
class EvaluationResult:
  weight: float
  state: EmbeddingFnState


def evaluate(
    embedding_fn: EmbeddingFn,
    states: Sequence[EmbeddingFnState],
    *,
    train_dataset: datasets.Dataset,
    test_dataset: datasets.Dataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_train_size: int = DEFAULT_MAX_TRAIN_DATASET_SIZE,
    max_test_size: int = DEFAULT_MAX_TEST_DATASET_SIZE,
    preprocessing_fn: Callable[[datasets.MiniBatch],
                               datasets.MiniBatch] = lambda x: x,
) -> Sequence[EvaluationResult]:
  """Given an emebdding function, computes the embedding function quality.

  Args:
    embedding_fn: A parameterized embedding function, parameterized by a "state"
    states: The states to use for the embedding function. The evaluation will be
      computed over all of the given states. This makes it possible to compare
      the embedding quality for multiple parameter sets.
    train_dataset: The dataset to train a KNN classifier with.
    test_dataset: A dataset of test examples to evaluate against.
    batch_size: The maximum batch size to use when computing embeddings.
    max_train_size: Limit the train dataset to this number of examples.
    max_test_size: Limit the test dataset to this number of examples.
    preprocessing_fn: A function to apply to the minibatches before embeddings
      are computed. This will be called in tensorflow datasets, and so must be a
      valid tensorflow function that can operate in graph mode.

  Returns:
    A sequence of results, in the same order as the input states, each of which
    corresponds to one of the states in the `states` argument. Each result has a
    weight (larger weights are better), and the associated state that achieved
    it. For classification tasks, the weight represents the accuracy on the test
    set. For multi-label classification tasks, the weight represents the mAP
    computed on the test dataset.
  """

  task = train_dataset.task_key
  train_ds, test_ds = _create_datasets(train_dataset, test_dataset, batch_size,
                                       preprocessing_fn, max_train_size,
                                       max_test_size)
  result = []

  for state in states:

    if task.kind is tasks.TaskKind.MULTI_LABEL_CLASSIFICATION:
      weight = _evaluate_multilabel_classification_embedding(
          embedding_fn,
          state,
          train_ds,
          test_ds,
      )
    elif task.kind is tasks.TaskKind.CLASSIFICATION:
      weight = _evaluate_classification_embedding(
          embedding_fn,
          state,
          train_ds,
          test_ds,
      )
    else:
      raise ValueError(f'Unsupported task kind: {task}')

    result.append(EvaluationResult(weight, state))

  return result


def _evaluate_multilabel_classification_embedding(
    embedding_fn: EmbeddingFn,
    state: EmbeddingFnState,
    train_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
) -> float:
  """Evaluates multilabel classifiction tasks."""

  train_embeddings, train_labels = _compute_multilabel_classification_embeddings(
      embedding_fn,
      state,
      train_ds,
  )

  test_embeddings, test_labels = _compute_multilabel_classification_embeddings(
      embedding_fn,
      state,
      test_ds,
  )

  classifier = _MultiLabelKNNClassifier()
  classifier.fit(train_embeddings, train_labels)
  predictions = classifier.predict(test_embeddings)
  chex.assert_equal_shape([predictions, test_labels])

  mean_average_precision = sklearn.metrics.average_precision_score(
      test_labels,
      predictions,
  )

  return mean_average_precision


def _evaluate_classification_embedding(
    embedding_fn: EmbeddingFn,
    state: EmbeddingFnState,
    train_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
) -> float:
  """Evaluates an embedding function."""

  train_embeddings, train_labels = _compute_classification_embeddings(
      embedding_fn,
      state,
      train_ds,
  )

  test_embeddings, test_labels = _compute_classification_embeddings(
      embedding_fn,
      state,
      test_ds,
  )

  classifier = sklearn.neighbors.KNeighborsClassifier()
  classifier.fit(train_embeddings, train_labels)

  predictions = classifier.predict(test_embeddings)
  chex.assert_equal_shape([predictions, test_labels])
  accuracy = (predictions == test_labels).mean()

  return accuracy


def _compute_multilabel_classification_embeddings(
    embedding_fn: EmbeddingFn,
    state: EmbeddingFnState,
    ds: tf.data.Dataset,
) -> Tuple[np.ndarray, np.ndarray]:
  """Computes embeddings for multilabel classifiction tasks."""

  embeddings = []
  labels = []
  total_examples_seen = 0

  for batch in tfds.as_numpy(ds):
    logging.log_every_n_seconds(logging.INFO, 'Completed %d embeddings...',
                                10, total_examples_seen)
    embeddings.append(embedding_fn(state, batch))
    labels.append(batch.multi_label_one_hot.astype(np.int32))
    total_examples_seen += batch.image.shape[0]

  logging.info('Completed %d embeddings [done].', total_examples_seen)

  embeddings = np.concatenate(embeddings, axis=0)
  labels = np.concatenate(labels, axis=0)

  chex.assert_rank(embeddings, 2)
  chex.assert_rank(labels, 2)

  return embeddings, labels


def _compute_classification_embeddings(
    embedding_fn: EmbeddingFn,
    state: EmbeddingFnState,
    ds: tf.data.Dataset,
) -> Tuple[np.ndarray, np.ndarray]:
  """Computes emebddings for classification tasks."""

  embeddings = []
  labels = []
  total_examples_seen = 0

  for batch in tfds.as_numpy(ds):
    logging.log_every_n_seconds(logging.INFO,
                                'Completed %d embeddings...', 10,
                                total_examples_seen)
    embeddings.append(embedding_fn(state, batch))
    labels.append(batch.label)
    total_examples_seen += batch.image.shape[0]

  logging.info('Completed %d embeddings [done].', total_examples_seen)

  embeddings = np.concatenate(embeddings, axis=0)
  labels = np.concatenate(labels, axis=0)

  chex.assert_rank(embeddings, 2)
  chex.assert_rank(labels, 1)

  return embeddings, labels


def _create_datasets(
    train_dataset: datasets.Dataset,
    test_dataset: datasets.Dataset,
    batch_size: int,
    preprocessing_fn: Callable[[datasets.MiniBatch], datasets.MiniBatch],
    max_train_size: int,
    max_test_size: int,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  """Creates the datasets for training and testing."""

  train_ds = train_dataset.builder_fn(shuffle=True)
  train_ds = train_ds.take(max_train_size)
  train_ds = train_ds.map(preprocessing_fn)
  train_ds = train_ds.batch(batch_size).cache()

  test_ds = test_dataset.builder_fn(shuffle=True)
  test_ds = test_ds.take(max_test_size)
  test_ds = test_ds.map(preprocessing_fn)
  test_ds = test_ds.batch(batch_size).cache()

  return train_ds, test_ds


class _MultiLabelKNNClassifier:
  """A multi-label classifier for multi-label binary classification tasks."""

  def __init__(self):
    self._classifier = sklearn.neighbors.KNeighborsClassifier()

  def fit(self, x: np.ndarray, y: np.ndarray) -> None:
    """Fits a knn classifier for features x and labels y.

    It is assumed that y is a binary vector with final dimension num_labels.

    Args:
      x: The features with shape (n_examples, n_features).
      y: The labels with shape (n_examples, n_labels).
    """
    if y.dtype != np.int32:
      raise ValueError(f'y should have type np.int32, got {y.dtype}')

    if np.amax(y) > 1 or np.amin(y) < 0:
      raise ValueError(
          f'y must contain only 0s and 1s, got {np.amax(y), np.amin(y)}')

    self._y0 = y[0]
    self._classifier.fit(x, y)

  def predict(self, x: np.ndarray) -> np.ndarray:
    """Computes predictions for features x.

    Args:
      x: The features to compute predictions for.

    Returns:
      An array of shape (n_examples, n_labels), where result[i, j] represents
      probability that the jth label is present for the ith example.

    """
    predictions = self._classifier.predict_proba(x)
    result = []

    # This little dance is required since sklearn "automatically" computes the
    # number of classes for each label. If any label is all 1 or all 0, sklearn
    # will presume that this label only has a single class. We thus have handle
    # this case explicitly.
    for i, prediction in enumerate(predictions):
      if prediction.shape[-1] == 2:
        result.append(prediction)
      elif prediction.shape[-1] == 1:
        predicted_class = int(self._y0[i])
        prediction = np.zeros((prediction.shape[0], 2))
        prediction[:, predicted_class] = 1
        result.append(prediction)
      else:
        raise ValueError(f'Unexpected num classes: {prediction.shape[-1]}')

    # The result returned by sklearn is a list of length n_labels, each of
    # which contains an array of shape (n_examples, n_classes).
    # This is because sklearn supports the case that each label has a variable
    # number of classes. In our case, we know they are all binary.
    result = np.stack(result, axis=0)
    result = np.transpose(result, (1, 0, 2))
    result = result[:, :, 1]
    return result
