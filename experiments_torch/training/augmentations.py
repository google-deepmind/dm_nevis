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

"""Data augmentations.

Augmentations are intended to be used in the context of a map function in a
tf.data.Dataset. This means that the functions must be applyable in tensorflow
graph mode [1]. To achieve this, any run-time variable shape must be managed
strictly using tensorflow functions (such as tf.cond(...)).

This can be tested using the test fixutres in the tests for this module.

[1]: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map
"""

import dataclasses
import functools
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
from dm_nevis.benchmarker.datasets import datasets
import tensorflow as tf


AugmentFn = Callable[[datasets.MiniBatch], datasets.MiniBatch]
Kwargs = Dict[str, Any]
DEFAULT_PADDING = 0
# offset_y, offset_x, crop_height, crop_width
CropWindow = Tuple[int, int, int, int]
AREA_RANGE = (0.08, 1.0)
MAX_ATTEMPTS = 10
ASPECT_RATIO_RANGE = (3 / 4, 4 / 3)
MIN_OBJECT_COVERED = 0.1


def chain(
    example: datasets.MiniBatch,
    augmentation_ctors_with_kwargs: Sequence[Tuple[AugmentFn, Kwargs]]
) -> datasets.MiniBatch:
  """Applies data augmentations to example sequentially."""
  for (ctor, kwargs) in augmentation_ctors_with_kwargs:
    augmentation_fn = functools.partial(ctor, **kwargs)
    example = augmentation_fn(example)
  return example


def standardize_per_image(example: datasets.MiniBatch) -> datasets.MiniBatch:
  """Standartizes each image."""
  image = tf.image.per_image_standardization(example.image)
  return dataclasses.replace(example, image=image)


def random_flip(example: datasets.MiniBatch) -> datasets.MiniBatch:
  """Randomly flips each image."""
  image = tf.image.random_flip_left_right(example.image)
  return dataclasses.replace(example, image=image)


def normalize(example: datasets.MiniBatch) -> datasets.MiniBatch:
  """Ensures the images have 3 channels and are in range -1..1."""
  # Images from nevis datasets are 0..255, however stored as int64.
  # This confuses the other image-preprocessing functions => cast to uint8.
  image = example.image

  def true_fn():
    # no-op for grayscale, results in correct shape for RGB
    image_sliced = image[:, :, :1]
    return tf.image.grayscale_to_rgb(image_sliced)

  is_grayscale = tf.equal(tf.shape(image)[-1], 1)
  image = tf.cond(
      pred=is_grayscale,
      true_fn=true_fn,
      false_fn=lambda: image)

  # Convert to range -1..1
  image = tf.cast(image, dtype=tf.uint8)
  image = 2 * tf.image.convert_image_dtype(image, dtype=tf.float32) - 1
  return dataclasses.replace(example, image=image)


def _distorted_bounding_box_crop(
    image_shape: tf.Tensor,
    *,
    bbox: tf.Tensor,
    min_object_covered: float,
    aspect_ratio_range: Tuple[float, float],
    area_range: Tuple[float, float],
    max_attempts: int,
    seed: Optional[int] = None,
) -> CropWindow:
  """Generates cropped_image using one of the bboxes randomly distorted."""
  bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
      image_shape,
      bounding_boxes=bbox,
      min_object_covered=min_object_covered,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=max_attempts,
      use_image_if_no_bounding_boxes=True,
      seed=seed)

  # Crop the image to the specified bounding box.
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  return (offset_y, offset_x, target_height, target_width)


def sample_random_crop_window(image_shape: tf.Tensor,
                              seed: Optional[int] = None) -> CropWindow:
  """Randomly sample a crop window, given an image size and config.

  It may be that the random sampler is unable to satisfy the constraints given
  (within an acceptable number of iterations). In this case, the sampler
  falls back to returning the result of `pad_and_center_crop_window`, with the
  default padding set.

  Args:
    image_shape: A tensor containing [image_height, image_width, channels].
    seed: If specified, a random seed for sampling cropping window.

  Returns:
    A crop window [min y, min x, height, width] in image coordinates.
  """

  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  crop_window = _distorted_bounding_box_crop(
      image_shape,
      bbox=bbox,
      min_object_covered=MIN_OBJECT_COVERED,
      aspect_ratio_range=ASPECT_RATIO_RANGE,
      area_range=AREA_RANGE,
      max_attempts=MAX_ATTEMPTS,
      seed=seed)

  # If the random crop failed, return the center crop.
  if tf.reduce_all(tf.equal(image_shape[:2], crop_window[2:])):
    crop_window = central_crop_window(image_shape)
  return crop_window


def central_crop_window(image_shape: tf.Tensor,
                        padding: int = DEFAULT_PADDING) -> CropWindow:
  """Compute a crop window for a padded center crop of the given image shape.

  Args:
    image_shape: The shape of the jpeg [height, width, channels], or [height,
      width].
    padding: The padding between the input image and the resulting image. The
      padding represents the distance between the input image and the output
      image at each edge (so that the total number of pixels removed from the
      smallest edge is 2 X the padding value.

  Returns:
    A crop window [y, x, image_size, image_size],
    where image_size = min(height, width) - 2 * padding, and y and x are
    chosen so that the resutling crop falls in the center of the input image.
  """
  # Scrub the channels size, if it was provided.
  image_shape = image_shape[:2]

  min_image_side = tf.math.reduce_min(image_shape)
  image_height = image_shape[0]
  image_width = image_shape[1]

  # If the padding is larger than the image, no pixels will be returned.
  tf.debugging.assert_greater(min_image_side, 2 * padding)

  offset_y = tf.cast((image_height - min_image_side) / 2, dtype=tf.int32)
  offset_x = tf.cast((image_width - min_image_side) / 2, dtype=tf.int32)

  image_size = tf.cast(min_image_side - 2 * padding, tf.int32)
  return (offset_y + padding, offset_x + padding, image_size, image_size)


def central_crop_via_cropped_window_and_resize(
    example: datasets.MiniBatch, size: Tuple[int, int]) -> datasets.MiniBatch:
  """Extract the central region of the image and resize to the given size."""

  crop_window = central_crop_window(tf.shape(example.image))
  cropped_image = tf.image.crop_to_bounding_box(example.image, *crop_window)

  cropped_image = tf.image.resize(cropped_image, size=size)

  return dataclasses.replace(example, image=cropped_image)


def random_crop_via_cropped_window_and_resize(
    example: datasets.MiniBatch, size: Tuple[int, int]) -> datasets.MiniBatch:
  """Randomly sample a crop from the image and resize to the given size."""

  crop_window = sample_random_crop_window(tf.shape(example.image))
  cropped_image = tf.image.crop_to_bounding_box(example.image, *crop_window)

  cropped_image = tf.image.resize(cropped_image, size=size)

  return dataclasses.replace(example, image=cropped_image)


def central_crop(example: datasets.MiniBatch,
                 size: Tuple[int, int]) -> datasets.MiniBatch:
  """Extract the central region of the image."""
  height = tf.shape(example.image)[0]
  width = tf.shape(example.image)[1]
  tf.debugging.assert_equal(height, width)

  fraction = size[0] / height
  image = tf.image.central_crop(example.image, fraction)
  return dataclasses.replace(example, image=image)


def random_crop(example: datasets.MiniBatch,
                size: Tuple[int, int]) -> datasets.MiniBatch:
  """Randomly sample crops with `size`."""
  height, width = size
  n_channels = tf.shape(example.image)[-1]
  image = tf.image.random_crop(example.image, (height, width, n_channels))
  return dataclasses.replace(example, image=image)


def resize(example: datasets.MiniBatch, size: Tuple[int,
                                                    int]) -> datasets.MiniBatch:
  """Resizes the image to the given size."""
  image = tf.image.resize(example.image, size)
  return dataclasses.replace(example, image=image)
