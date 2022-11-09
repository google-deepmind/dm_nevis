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

"""Encoding and decoding of dataset examples.

Examples are represented as instaces of the `Example` dataclass.
These dataclasses are serialized to tf.train.Example protobuf containers, which
can be serialized to the wire. When loading datasets, the dataset_loader
deserializes data into tf.train.Example protobuff and parses it given the
features description provided by the dataset metadata.
"""

import io
import os
import queue
import threading
import types
from typing import Any, Callable, Optional, Type
from dm_nevis.datasets_storage.handlers import types as handlers_types
import PIL.Image as pil_image
import tensorflow as tf
import tensorflow_datasets as tfds

DEFAULT_IMAGE_FEATURE_NAME = 'png_encoded_image'
DEFAULT_LABEL_FEATURE_NAME = 'label'
DEFAULT_MULTI_LABEL_FEATURE_NAME = 'multi_label'

SUPPORTED_FEATURES = frozenset([
    DEFAULT_IMAGE_FEATURE_NAME, DEFAULT_LABEL_FEATURE_NAME,
    DEFAULT_MULTI_LABEL_FEATURE_NAME
])

RecordWriter = Callable[[str], Any]

_DEFAULT_MAX_WRITER_QUEUE_SIZE = 100


def get_default_features(num_channels: int,
                         num_classes: int) -> tfds.features.FeaturesDict:
  """Creates a default single class features specification."""
  return tfds.features.FeaturesDict({
      DEFAULT_IMAGE_FEATURE_NAME:
          tfds.features.Image(shape=(None, None, num_channels)),
      DEFAULT_LABEL_FEATURE_NAME:
          tfds.features.ClassLabel(num_classes=num_classes)
  })


def build_example_encoder(
    features: tfds.features.FeaturesDict
) -> Callable[[handlers_types.Example], tf.train.Example]:
  """Returns a function to create tf.train.Example."""

  if set(features.keys()) - SUPPORTED_FEATURES:
    raise ValueError('features contains features which are not supported.')

  example_serializer = tfds.core.example_serializer.ExampleSerializer(
      features.get_serialized_info())

  def encode(example: handlers_types.Example) -> tf.train.Example:
    """Creates a tf.train.Example. Ignored fields not in `features`."""
    example_to_serialize = {}

    for key, value in example._asdict().items():
      if key == 'image':
        example_to_serialize[DEFAULT_IMAGE_FEATURE_NAME] = _encoded_png_feature(
            value)
      elif key in features:
        example_to_serialize[key] = value

    return example_serializer.get_tf_example(example_to_serialize)

  return encode


def build_example_decoder(
    features: tfds.features.FeaturesDict
) -> Callable[[tf.Tensor], tf.train.Example]:
  """Returns feature decoder."""

  example_parser = tfds.core.example_parser.ExampleParser(
      features.get_serialized_info())

  def decoder(encoded_tf_example: tf.Tensor) -> tf.train.Example:
    """Decodes the example into tf.train.Example proto."""
    return features.decode_example(
        example_parser.parse_example(encoded_tf_example))

  return decoder


class ThreadpoolExampleWriter:
  """A class for encoding and writing examples to shards."""

  _STOP_QUEUE = object()

  def __init__(self,
               num_shards: int,
               example_encoder: Callable[[handlers_types.Example],
                                         tf.train.Example],
               *,
               output_dir: str = '',
               record_writer: RecordWriter = tf.io.TFRecordWriter) -> None:
    """Creates the writer class, with the given number of write shards.

    Each shard is assigned a worker thread, which handles encoding and protobuf
    serializing. This class is designed to be used as a context manager. Writing
    happens asynchronously and is only completed once the context manager exits
    successfully.

    Args:
      num_shards: The number of shards to write.
      example_encoder: Function which encodes examples into tf.train.Example
        protobuf.
      output_dir: The directory name to join to the shard path.
      record_writer: The class used to write records.
    """

    self._example_encoder = example_encoder

    self._queues = []
    self._threads = []

    def worker(q, path):
      writer = record_writer(path)
      while True:
        example = q.get()
        if example is self._STOP_QUEUE:
          break
        writer.write(self._example_encoder(example).SerializeToString())

    for i in range(num_shards):
      q = queue.Queue(maxsize=_DEFAULT_MAX_WRITER_QUEUE_SIZE)
      path = os.path.join(output_dir, f'data-{i:05d}-of-{num_shards:05d}')
      thread = threading.Thread(target=worker, args=(q, path), daemon=True)
      thread.start()
      self._queues.append(q)
      self._threads.append(thread)

  def __enter__(self):
    return self

  def __exit__(self, exc_type: Optional[Type[BaseException]],
               exc_val: Optional[BaseException],
               exc_tb: Optional[types.TracebackType]) -> None:
    """Blocks until writing the shards is complete."""
    del exc_type, exc_val, exc_tb

    for q in self._queues:
      q.put(self._STOP_QUEUE)

    for thread in self._threads:
      thread.join()

  def write(self, shard: int, example: handlers_types.Example) -> None:
    """Asynchronously writes the given example to the given shard index."""
    self._queues[shard].put(example)


def _encoded_png_feature(image: handlers_types.ImageLike) -> bytes:
  """Encodes an image to bytes."""

  if not isinstance(image, pil_image.Image):
    raise ValueError('Encoder only works with PIL images.')

  buffer = io.BytesIO()

  # We strip any ICC profile to avoid decoding warnings caused by invalid ICC
  # profiles in some datasets.
  image.save(buffer, format='PNG', icc_profile=b'')

  buffer = buffer.getvalue()
  return buffer
