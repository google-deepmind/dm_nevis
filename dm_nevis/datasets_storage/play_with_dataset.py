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

r"""Script to download all the Nevis datasets.
"""

import os
from typing import Sequence
from absl import app
from absl import flags
from absl import logging
from dm_nevis.datasets_storage import dataset_loader
import tensorflow_datasets as tfds

_DATASET = flags.DEFINE_string('dataset', 'animal', 'Dataset name.')
_DATASET_VERSION = flags.DEFINE_string('dataset_version', 'temp',
                                       'Dataset version.')
_DATASET_ROOT_DIR = flags.DEFINE_string('dataset_root_dir', '',
                                        'Dataset version.')


def main(argv: Sequence[str]) -> None:
  del argv

  if _DATASET_ROOT_DIR.value:
    path = os.path.join(_DATASET_ROOT_DIR.value, _DATASET.value)
    logging.info('Reading dataset from %s', path)
  else:
    path = None

  if path:
    metadata = dataset_loader.get_metadata_from_path(path)
  else:
    metadata = dataset_loader.get_metadata(
        _DATASET.value, version=_DATASET_VERSION.value)

  splits = metadata.additional_metadata['splits']
  num_classes = metadata.num_classes
  image_shape = metadata.image_shape

  logging.info('metadata: %s', str(metadata))
  logging.info('splits: %s', str(splits))
  logging.info('num_classes: %d', num_classes)
  logging.info('image_shape: %s', str(image_shape))

  for split in splits:
    logging.info('Trying split `%s`.', split)

    if path:
      dataset = dataset_loader.load_dataset_from_path(path, split)
    else:
      dataset = dataset_loader.load_dataset(
          _DATASET.value, split, version=_DATASET_VERSION.value)

    ds = iter(tfds.as_numpy(dataset.builder_fn(shuffle=True)))
    elem = next(ds)

    logging.info(elem)

    logging.info('Checking the integrity of `%s`.', split)
    for elem in ds:
      pass
    logging.info('Checks for `%s` are passed.', split)


if __name__ == '__main__':
  app.run(main)
