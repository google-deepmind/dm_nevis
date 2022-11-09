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

"""Tests for dm_nevis.datasets_storage.download_util."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_nevis.datasets_storage import download_util as du


class DownloadUtilTest(parameterized.TestCase):

  def test_get_filename_from_url(self):
    url = """http://docs.python.org:80/3/library/foo.image.jpg?highlight=params#url-parsing"""
    expected = 'foo.image.jpg'
    self.assertEqual(du._get_filename_from_url(url), expected)

  @parameterized.parameters([
      (1000, 10),
      (10, 1000),
      (10000, 100),
  ])
  def test_shuffle_generator_with_buffer(self, num_elements,
                                         shuffle_buffer_size):

    def make_gen():
      for i in range(num_elements):
        yield i

    shuffled_gen = du.shuffle_generator_with_buffer(make_gen(),
                                                    shuffle_buffer_size)

    self.assertSameElements(list(shuffled_gen), list(make_gen()))

    self.assertNotEqual(list(shuffled_gen), list(make_gen()))


if __name__ == '__main__':
  absltest.main()
