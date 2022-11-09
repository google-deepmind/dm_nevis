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

"""Tests for dm_nevis.benchmarker.streams.streams."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_nevis.benchmarker.datasets import streams
from dm_nevis.benchmarker.datasets import tasks
from dm_nevis.benchmarker.datasets import test_stream


class StreamsTest(parameterized.TestCase):

  def test_filtered_stream(self):
    stream = streams.FilteredStream(
        test_stream.TestStream,
        supported_task_kinds=[tasks.TaskKind.CLASSIFICATION])
    self.assertLen(list(stream.events()), 4)

    stream = streams.FilteredStream(
        test_stream.TestStream,
        supported_task_kinds=[tasks.TaskKind.MULTI_LABEL_CLASSIFICATION])
    self.assertEmpty(list(stream.events()))


if __name__ == '__main__':
  absltest.main()
