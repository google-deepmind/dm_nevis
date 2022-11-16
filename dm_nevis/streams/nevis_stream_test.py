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
"""Tests for dm_nevis.streams.nevis_stream."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_nevis.streams import nevis_stream


class NevisStreamTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("spaces", "some data key", "some_data_key"),
      ("everything", " /-~", "____"),
  )
  def test_canonicalize_name(self, name, expected):
    self.assertEqual(nevis_stream._canonicalize_name(name), expected)

  @parameterized.named_parameters(
      ("full_stream", nevis_stream.NevisStreamVariant.FULL),
      ("short_tream", nevis_stream.NevisStreamVariant.SHORT),
  )
  def test_datasets_in_stream(self, stream_variant):
    n1 = nevis_stream.datasets_in_stream(stream_variant, remove_duplicates=True)
    n2 = nevis_stream.datasets_in_stream(
        stream_variant, remove_duplicates=False)

    self.assertSameElements(n1, n2)


if __name__ == "__main__":
  absltest.main()
