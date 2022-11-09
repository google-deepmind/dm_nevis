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

"""Tests for dm_nevis.datasets_storage.encoding."""

import collections
from typing import Iterable
from absl.testing import absltest
from absl.testing import parameterized
from dm_nevis.datasets_storage import encoding
from dm_nevis.datasets_storage.handlers import types
import numpy as np
import PIL.Image as pil_image
import tensorflow as tf


class EncodingTest(parameterized.TestCase):

  @parameterized.parameters([
      {
          'num_examples': 10,
          'num_shards': 2
      },
      {
          'num_examples': 7,
          'num_shards': 1
      },
      {
          'num_examples': 7,
          'num_shards': 3
      },
      {
          'num_examples': 1,
          'num_shards': 5
      },
      {
          'num_examples': 0,
          'num_shards': 5
      },
  ])
  def test_threadpool_example_writer(self, num_examples, num_shards):
    examples = list(_example_fixtures(n=num_examples))
    features = encoding.get_default_features(num_channels=3, num_classes=10)
    written_data = collections.defaultdict(list)

    class MemoryRecordWriter:

      def __init__(self, path: str) -> None:
        self._path = path

      def write(self, buffer: bytes) -> None:
        written_data[self._path].append(buffer)

    with encoding.ThreadpoolExampleWriter(
        num_shards=num_shards,
        record_writer=MemoryRecordWriter,
        example_encoder=encoding.build_example_encoder(features)) as writer:

      for i, example in enumerate(examples):
        writer.write(i % num_shards, example)

    total_written = sum(len(v) for v in written_data.values())
    self.assertEqual(num_examples, total_written)

  def test_encode_example(self):
    example, *_ = _example_fixtures(n=1)
    features = encoding.get_default_features(num_channels=3, num_classes=10)
    encoder = encoding.build_example_encoder(features=features)
    encoded = encoder(example)
    decoder = encoding.build_example_decoder(features=features)
    result = decoder(encoded.SerializeToString())

    self.assertEqual(
        [30, 60, 3],
        result[encoding.DEFAULT_IMAGE_FEATURE_NAME].shape.as_list())
    self.assertIn(result[encoding.DEFAULT_LABEL_FEATURE_NAME].numpy(),
                  set(range(10)))
    self.assertIsInstance(result[encoding.DEFAULT_IMAGE_FEATURE_NAME],
                          tf.Tensor)
    self.assertEqual(result[encoding.DEFAULT_IMAGE_FEATURE_NAME].dtype,
                     tf.uint8)

    np.testing.assert_allclose(result[encoding.DEFAULT_IMAGE_FEATURE_NAME],
                               example.image)


def _example_fixtures(*, n: int) -> Iterable[types.Example]:
  gen = np.random.default_rng(seed=0)

  for _ in range(n):
    img = pil_image.fromarray(
        gen.integers(0, 255, size=(30, 60, 3), dtype=np.uint8))

    yield types.Example(
        image=img,
        label=gen.integers(0, 10),
        multi_label=None,
    )


if __name__ == '__main__':
  absltest.main()
