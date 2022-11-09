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

"""MPEG7 handler."""

import os
from typing import Optional
import zipfile

from absl import logging

from dm_nevis.datasets_storage.handlers import extraction_utils as utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types

from tensorflow.io import gfile


def _label_from_filename(filename: str) -> Optional[str]:
  """Extracts a text label given a filename for the MPEG7 dataset.

  Args:
    filename: Name of the file, for example,
      '/path/to/dataset/original/lizzard-13.gif'

  Returns:
    label: A text label, for example, 'lizzard' in the filename above
  """
  # There are six extra files when the dataset is downloaded. Ignore those files
  label = os.path.split(filename)[1].split('-')[0]
  if 'confusion' in label or 'shapedata' in label:
    logging.info('skipping %s', label)
    return None
  return label


def _path_to_label_fn(path: str, label_to_id):
  label = _label_from_filename(path)
  if label:
    return label_to_id[label]
  else:
    return None


# pylint:disable=missing-function-docstring
def mpeg7_handler(dataset_path: str) -> types.HandlerOutput:
  files = gfile.listdir(dataset_path)
  labels = set()

  for file in files:
    with zipfile.ZipFile(os.path.join(dataset_path, file), 'r') as zf:
      for filename in zf.namelist():
        label = _label_from_filename(filename)
        if label is not None:
          labels.add(label)

  labels = sorted(labels)
  num_classes = len(labels)
  label_to_id = dict(((label, idx) for idx, label in enumerate(labels)))

  metadata = types.DatasetMetaData(
      num_classes=num_classes,
      num_channels=1,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(label_to_id=label_to_id))

  def make_gen_fn():
    return utils.generate_images_from_zip_files(
        dataset_path,
        files,
        path_to_label_fn=lambda path: _path_to_label_fn(path, label_to_id))

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)
  return metadata, per_split_gen


mpeg7_dataset = types.DownloadableDataset(
    name='mpeg7',
    download_urls=[
        types.DownloadableArtefact(
            url='https://dabi.temple.edu/external/shape/MPEG7/MPEG7dataset.zip',
            checksum='bedd54856c425dcc6e242515c4f67d75')
    ],
    website_url='https://dabi.temple.edu/external/shape/MPEG7',
    paper_title='Learning Context Sensitive Shape Similarity by Graph Transduction',
    authors=' Xiang Bai and Xingwei Yang and Longin Jan Latecki and Wenyu Liu and Zhuowen Tu.',
    year='2009',
    handler=mpeg7_handler,
)
