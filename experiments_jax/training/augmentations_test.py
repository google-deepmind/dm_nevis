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

"""Tests for experiments_jax.training.augmentations."""

import functools

from typing import Any, Mapping, Sequence, Tuple
from absl.testing import absltest
from absl.testing import parameterized
from dm_nevis.benchmarker.datasets import datasets
from experiments_jax.training import augmentations
import numpy as np
import tensorflow as tf


class AugmentationsTest(parameterized.TestCase):

  def test_chain(self):
    ds = _fixture_dataset([((30, 60, 3), 0), ((30, 60, 1), 0)])
    augmentation_fn = functools.partial(
        augmentations.chain,
        augmentation_ctors_with_kwargs=[
            (augmentations.normalize, {}),
            (augmentations.resize, {
                'size': (30, 30)
            }),
            (augmentations.random_crop, {
                'size': (20, 20)
            }),
        ])
    ds = ds.map(augmentation_fn)

    items = list(ds)
    self.assertLen(items, 2)
    for item in items:
      # Grayscale images should be converted to color.
      self.assertEqual(3, item.image.shape[-1])
      self.assertEqual((20, 20, 3), item.image.shape)

  def test_normalize(self):
    ds = _fixture_dataset([((30, 60, 3), 0), ((30, 60, 1), 0)])
    ds = ds.map(augmentations.normalize)

    items = list(ds)
    self.assertLen(items, 2)
    for item in items:
      # Grayscale images should be converted to color.
      self.assertEqual(3, item.image.shape[-1])

  def test_standardize_per_image(self):
    ds = _fixture_dataset([((30, 60, 3), 0), ((30, 60, 1), 0)])
    ds = ds.map(augmentations.standardize_per_image)

    items = list(ds)
    # We only test whether this does compile.
    self.assertLen(items, 2)

  def test_random_flip(self):
    ds = _fixture_dataset([((30, 60, 3), 0), ((30, 60, 1), 0)])
    ds = ds.map(augmentations.random_flip)

    items = list(ds)
    # We only test whether this does compile.
    self.assertLen(items, 2)

  def test_resize(self):
    ds = _fixture_dataset([((30, 30, 3), 0)])
    ds = ds.map(functools.partial(augmentations.resize, size=(20, 20)))

    items = list(ds)
    self.assertLen(items, 1)

    for item in items:
      self.assertEqual((20, 20, 3), item.image.shape)

  def test_central_crop_via_cropped_window_and_resize(self):
    ds = _fixture_dataset([((30, 30, 3), 0)])
    ds = ds.map(
        functools.partial(
            augmentations.central_crop_via_cropped_window_and_resize,
            size=(20, 20)))

    items = list(ds)
    self.assertLen(items, 1)

    for item in items:
      self.assertEqual((20, 20, 3), item.image.shape)

  def test_random_crop_via_cropped_window_and_resize(self):
    ds = _fixture_dataset([((30, 30, 3), 0)])
    ds = ds.map(
        functools.partial(
            augmentations.random_crop_via_cropped_window_and_resize,
            size=(20, 20)))

    items = list(ds)
    self.assertLen(items, 1)

    for item in items:
      self.assertEqual((20, 20, 3), item.image.shape)

  def test_central_crop_via_cropped_window_and_resize_small_image(self):
    ds = _fixture_dataset([((3, 3, 3), 0)])
    ds = ds.map(
        functools.partial(
            augmentations.central_crop_via_cropped_window_and_resize,
            size=(2, 2)))

    items = list(ds)
    self.assertLen(items, 1)

    for item in items:
      self.assertEqual((2, 2, 3), item.image.shape)

  def test_random_crop_via_cropped_window_and_resize_small_image(self):
    ds = _fixture_dataset([((3, 3, 3), 0)])
    ds = ds.map(
        functools.partial(
            augmentations.random_crop_via_cropped_window_and_resize,
            size=(2, 2)))

    items = list(ds)
    self.assertLen(items, 1)

    for item in items:
      self.assertEqual((2, 2, 3), item.image.shape)

  def test_central_crop(self):
    ds = _fixture_dataset([((30, 30, 3), 0)])

    ds = ds.map(functools.partial(augmentations.central_crop, size=(20, 20)))

    items = list(ds)
    self.assertLen(items, 1)

    for item in items:
      self.assertEqual((20, 20, 3), item.image.shape)

  def test_random_crop(self):
    ds = _fixture_dataset([((30, 30, 3), 0)])
    ds = ds.map(functools.partial(augmentations.random_crop, size=(20, 20)))

    items = list(ds)
    self.assertLen(items, 1)

    for item in items:
      self.assertEqual((20, 20, 3), item.image.shape)

  @parameterized.parameters([
      dict(image_shape=(224, 300), padding=0, expected=(0, 38, 224, 224)),
      dict(image_shape=(300, 224), padding=0, expected=(38, 0, 224, 224)),
      dict(image_shape=(224, 300), padding=16, expected=(16, 54, 192, 192)),
      dict(image_shape=(300, 224), padding=16, expected=(54, 16, 192, 192)),
      dict(image_shape=(32 + 1, 32 + 1), padding=16, expected=(16, 16, 1, 1)),
  ])
  def test_central_crop_window(self, image_shape, padding, expected):
    image_shape = tf.constant(image_shape, dtype=tf.int32)
    bbox = augmentations.central_crop_window(image_shape, padding)
    np.testing.assert_allclose(expected, bbox)

  @parameterized.parameters([
      dict(image_shape=(224, 300, 3)),
      dict(image_shape=(224, 224, 3)),
      dict(image_shape=(100, 10, 3)),
  ])
  def test_random_sample_crop_window(self, image_shape):

    windows = []

    for i in range(100):
      crop_window = augmentations.sample_random_crop_window(
          tf.constant(image_shape), seed=i)
      windows.append(tuple(w.numpy() for w in crop_window))

    # Test that we see plenty of variety in the samples.
    different_samples = set(windows)
    assert len(different_samples) > 50

    image_area = image_shape[0] * image_shape[1]

    (min_area, max_area) = augmentations.AREA_RANGE
    (min_aspect_ratio, max_aspect_ratio) = augmentations.ASPECT_RATIO_RANGE

    sampled_min_area = min(w[2] * w[3] for w in windows)
    sampled_max_area = max(w[2] * w[3] for w in windows)
    sampled_min_aspect_ratio = min(w[3] / w[2] for w in windows)
    sampled_max_aspect_ratio = min(w[3] / w[2] for w in windows)

    self.assertLess(sampled_max_area / image_area, max_area + 1e-4)
    self.assertGreater(sampled_min_area / image_area, min_area - 1e-4)
    self.assertLess(sampled_max_aspect_ratio, max_aspect_ratio + 1e-4)
    self.assertGreater(sampled_min_aspect_ratio, min_aspect_ratio - 1e-4)


def _fixture_dataset(
    shapes_and_labels: Sequence[Tuple[Tuple[int, int, int], int]]
) -> tf.data.Dataset:
  """Constructs a fixture containing minibatches.

  We round-trip the data via pngs, since this will result in shape tensors
  that are not determined at graph compile time. This ensures that the tested
  mappable functions work in graph mode, which is enforced by
  tf.data.Dataset.map(...).

  Args:
    shapes_and_labels: The image shapes and label values to use for the
      fixtures.

  Returns:
    A tensorflow dataset.
  """

  def gen():
    for shape, label in shapes_and_labels:
      yield _encode_example(image=np.zeros(shape, dtype=np.uint8), label=label)

  ds = tf.data.Dataset.from_generator(
      gen,
      output_signature={
          'image': tf.TensorSpec(shape=(), dtype=tf.string),
          'label': tf.TensorSpec(shape=(), dtype=tf.int32),
      })

  def to_minibatch(example) -> datasets.MiniBatch:
    return datasets.MiniBatch(
        image=tf.io.decode_png(example['image']),
        label=example['label'],
        multi_label_one_hot=None,
    )

  return ds.map(to_minibatch, deterministic=True)


def _encode_example(image: np.ndarray, label: int) -> Mapping[str, Any]:
  """Create a tf example using named fields."""

  return {
      'image': _encoded_png_feature(image),
      'label': label,
  }


def _encoded_png_feature(image: np.ndarray) -> bytes:
  return tf.io.encode_png(image).numpy()


if __name__ == '__main__':
  absltest.main()
