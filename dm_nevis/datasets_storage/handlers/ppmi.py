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

"""PPMI handler."""

import os
from typing import Dict
import zipfile

from dm_nevis.datasets_storage.handlers import extraction_utils as utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types

from tensorflow.io import gfile

_ZIP_FILENAME = 'norm_ppmi_12class.zip'
_IGNORED_FILES_REGEX_TEST = r'^README$|train'
_IGNORED_FILES_REGEX_TRAIN = r'^README$|test'


def _label_from_fname(filename: str, label_to_id: Dict[str, int]) -> int:
  """Extracts a label given a filename for the PPMI dataset."""
  label_str = _get_prefix(filename)
  label = label_to_id[label_str]
  return label


def _get_prefix(path: str) -> str:
  pieces = path.split(os.sep)
  if len(pieces) >= 3:
    return os.path.join(pieces[0], pieces[1], pieces[2])
  else:
    return ''


def ppmi_handler(dataset_path: str) -> types.HandlerOutput:
  """Imports PPMI dataset.

  The dataset home page is at
  https://ai.stanford.edu/~bangpeng/ppmi.html#:~:text=People%20Playing%20Musical%20Instrument&text=The%20PPMI%20dataset%20contains%20images,saxophone%2C%20trumpet%2C%20and%20violin.
  The dataset comes as a single zip file containing two directories, one for
  images of people playing the instrument and the other with people holding the
  instrument. Images can be found in paths like:
  {play|with}_instrument/{violin|bassoon|...}/{train|test}/filenmame.jpg
  in total there are 24 classes.

  Args:
    dataset_path: Path with downloaded artifacts.

  Returns:
    Metadata and generator functions.
  """
  ds_file = gfile.listdir(dataset_path)
  assert len(ds_file) == 1
  assert ds_file[0] == _ZIP_FILENAME
  label_to_id = {}
  num_classes = 0
  with zipfile.ZipFile(os.path.join(dataset_path, ds_file[0]), 'r') as zf:
    dirs = list(set([os.path.dirname(x) for x in zf.namelist()]))
    labels = []
    for x in dirs:
      prefix = _get_prefix(x)
      if prefix:
        labels.append(prefix)
    labels = list(set(labels))
    num_classes = len(labels)
    for i, label in enumerate(labels):
      label_to_id[label] = i

  metadata = types.DatasetMetaData(
      num_classes=num_classes,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_to_id=label_to_id,
          task_type='classification',
          image_type='object'))

  def make_gen(ignore_files_regex):
    label_fn = lambda x: _label_from_fname(filename=x, label_to_id=label_to_id)
    return utils.generate_images_from_zip_files(
        dataset_path,
        ds_file,
        path_to_label_fn=label_fn,
        ignored_files_regex=ignore_files_regex)

  make_gen_fn = lambda: make_gen(_IGNORED_FILES_REGEX_TRAIN)
  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN,
      splits.MERGED_TRAIN_AND_DEV)
  per_split_gen['test'] = make_gen(_IGNORED_FILES_REGEX_TEST)

  return metadata, per_split_gen


ppmi_dataset = types.DownloadableDataset(
    name='ppmi',
    download_urls=[
        types.DownloadableArtefact(
            url='http://vision.stanford.edu/Datasets/norm_ppmi_12class.zip',
            checksum='88118d8c6b50d72f0bb37a89269185ab')
    ],
    website_url='https://ai.stanford.edu/~bangpeng/ppmi.html#:~:text=People%20Playing%20Musical%20Instrument&text=The%20PPMI%20dataset%20contains%20images,saxophone%2C%20trumpet%2C%20and%20violin.',
    paper_title='Grouplet: A Structured Image Representation for Recognizing Human and Object Interactions.',
    authors='Bangpeng Yao and Li Fei-Fei.',
    year='2010',
    handler=ppmi_handler,
)
