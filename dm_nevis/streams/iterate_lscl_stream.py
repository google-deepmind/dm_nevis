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

"""A binary to iterate through the lscl stream and verify that all data can be read.

This binary is for testing that all of the datasets can successfully be
located and accessed.
"""

from typing import Sequence

from absl import app
from absl import flags
from absl import logging
from dm_nevis.benchmarker.datasets import streams
from dm_nevis.streams import lscl_stream
from tensorflow.io import gfile
import tensorflow_datasets as tfds

_OUTPATH = flags.DEFINE_string('outpath', '/tmp/output.txt',
                               'Destination to write to.')
_STREAM_VARIANT = flags.DEFINE_enum_class('stream_variant',
                                          lscl_stream.LSCLStreamVariant.FULL,
                                          lscl_stream.LSCLStreamVariant,
                                          'Stream to iterate')
_STREAM_STOP_YEAR = flags.DEFINE_integer(
    'stream_stop_year', 2022, 'The year when to stop the stream (exclusive)')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('General statistics about datasets for streams:')
  datasets = lscl_stream.DATASET_NAME_TO_SOURCE
  num_available_datasets = len(
      [k for k, v in datasets.items() if v is not None])
  num_tfds_datasets = len(
      [k for k, v in datasets.items() if isinstance(v, lscl_stream.TFDSSource)])
  num_nevis_datasets = len([
      k for k, v in datasets.items() if isinstance(v, lscl_stream.NevisSource)
  ])
  logging.info('Num known datasets %d', len(datasets))
  logging.info('Num available datasets %d', num_available_datasets)
  logging.info('Num tfds datasets %d', num_tfds_datasets)
  logging.info('Num nevis datasets %d', num_nevis_datasets)

  logging.info('Iterating LSCL %s stream up to year %d', _STREAM_VARIANT.value,
               _STREAM_STOP_YEAR.value)
  logging.info('Writing output to %s', _OUTPATH.value)

  total_test_images = 0
  total_train_images = 0
  total_dev_images = 0
  stream = lscl_stream.LSCLStream(
      stream_variant=_STREAM_VARIANT.value, stop_year=_STREAM_STOP_YEAR.value)
  with gfile.GFile(_OUTPATH.value, 'wt') as f:
    f.write('=== LSCL stream ===\n')

    num_train_datasets = 0

    for event in stream.events():
      logging.info('Reading datasets for %s', event)

      if isinstance(event, streams.PredictionEvent):
        test_dataset = stream.get_dataset_by_key(event.dataset_key)
        total_test_images += test_dataset.num_examples

      elif isinstance(event, streams.TrainingEvent):
        train_dataset = stream.get_dataset_by_key(event.train_dataset_key)
        dev_dataset = stream.get_dataset_by_key(event.dev_dataset_key)

        total_dev_images += dev_dataset.num_examples
        total_train_images += train_dataset.num_examples
        num_train_datasets += 1

      for key in streams.all_dataset_keys(event):
        dataset = stream.get_dataset_by_key(key)
        assert dataset.num_examples is not None

        f.write('---\n')
        f.write(f'  Key: {key}, num examples: {dataset.num_examples}\n')

        ds = dataset.builder_fn(shuffle=False).batch(1)
        for _ in range(10):
          batch = next(iter(tfds.as_numpy(ds)))
          f.write(f'  Example: {batch}\n')

    f.write('\n===\n')
    f.write(f'Num train images: {total_train_images}\n')
    f.write(f'Num test images: {total_test_images}\n')
    f.write(f'Total training datasets in stream: {num_train_datasets}\n')


if __name__ == '__main__':
  app.run(main)
