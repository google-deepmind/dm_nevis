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

"""Tiny ImagaNet handler."""

import os
import zipfile

from dm_nevis.datasets_storage.handlers import extraction_utils as utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types

from tensorflow.io import gfile


_NUM_CLASSES = 200
_PREFIX = 'tiny-imagenet-200'
_TEST_PREFIX = 'val'
_LABEL_FILE = 'wnids.txt'
_TEST_ANNOTATIONS = 'val_annotations.txt'
_IGNORED_REGEX_TRAIN = r'.*?\bval\b.*?|.*?\btest\b.*?|.*\.txt$'
_IGNORED_REGEX_TEST = r'.*?\btrain\b.*?|.*?\btest\b.*?|.*\.txt$'


# pylint:disable=missing-function-docstring
def tiny_imagenet_handler(dataset_path: str) -> types.HandlerOutput:

  dataset_file = gfile.listdir(dataset_path)
  assert len(dataset_file) == 1
  dataset_file = dataset_file[0]
  labels = set()

  with zipfile.ZipFile(os.path.join(dataset_path, dataset_file), 'r') as zf:
    # All object codes are given in a single text file
    with zf.open(os.path.join(_PREFIX, _LABEL_FILE), 'r') as flabel:
      for label in flabel:
        label = label.strip()
        labels.add(label.decode('utf-8'))

    # We use val set as the test set
    with zf.open(os.path.join(_PREFIX, _TEST_PREFIX, _TEST_ANNOTATIONS),
                 'r') as fval:
      # Map val annotations to the labels
      test_ann_to_label = {}
      for line in fval:
        line = line.strip()
        words = line.decode('utf-8').split('\t')
        test_ann_to_label[words[0]] = words[1]

  labels = sorted(labels)
  label_to_id = dict(((label, idx) for idx, label in enumerate(labels)))

  metadata = types.DatasetMetaData(
      num_classes=_NUM_CLASSES,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_to_id=label_to_id,
          task_type='classification',
          image_type='object'))

  def _label_from_filename_tr(filename):
    label = os.path.split(filename)[1].split('_')[0]
    return label_to_id[label]

  def _label_from_filename_test(filename):
    label = os.path.split(filename)[1]
    label = test_ann_to_label[label]
    label = label_to_id[label]
    assert 0 <= label < _NUM_CLASSES
    return label

  def gen_tr():
    return utils.generate_images_from_zip_files(
        dataset_path, [dataset_file],
        _label_from_filename_tr,
        ignored_files_regex=_IGNORED_REGEX_TRAIN,
        convert_mode='RGB')

  def gen_test():
    return utils.generate_images_from_zip_files(
        dataset_path, [dataset_file],
        _label_from_filename_test,
        ignored_files_regex=_IGNORED_REGEX_TEST,
        convert_mode='RGB')

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      gen_tr, splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN,
      splits.MERGED_TRAIN_AND_DEV)
  per_split_gen['test'] = gen_test()

  return metadata, per_split_gen


tiny_imagenet_dataset = types.DownloadableDataset(
    name='tiny_imagenet',
    download_urls=[
        types.DownloadableArtefact(
            url='http://cs231n.stanford.edu/tiny-imagenet-200.zip',
            checksum='90528d7ca1a48142e341f4ef8d21d0de')
    ],
    paper_title='Tiny ImageNet Visual Recognition Challenge',
    authors='Ya Le and Xuan Yang',
    year='2015',
    handler=tiny_imagenet_handler,
)
