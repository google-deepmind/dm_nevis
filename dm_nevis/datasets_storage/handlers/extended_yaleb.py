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

"""Extended Yale-B dataset handler."""

import os
from typing import List
import zipfile

from dm_nevis.datasets_storage.handlers import extraction_utils as utils
from dm_nevis.datasets_storage.handlers import types


_POSES = {
    'train': ['P00', 'P02', 'P03', 'P04', 'P07'],
    'dev': ['P05'],
    'train_and_dev': ['P00', 'P02', 'P03', 'P04', 'P07', 'P05'],
    'dev-test': ['P01'],
    'test': ['P06', 'P08']
}

_IGNORED_FILES_REGEX = r'info$|Ambient\.pgm$'
_SPLITS = ['train', 'dev', 'dev-test', 'train_and_dev', 'test']
_NUM_CLASSES = 28
_FILES_ID_RANGE = (11, 40)
_MISSING_ID = 14


def _get_all_class_names(directories: List[str]) -> List[str]:
  names = set()
  for fname in directories:
    fname = fname.split('/')[-2]
    names.add(fname)
  return sorted(names)


def extended_yaleb_handler(dataset_path: str) -> types.HandlerOutput:
  """Imports Extended Yale-B dataset.

  This is a face identification dataset. There are 28 subjects and images are
  taken under different viewing angle and illumination.
  We are going to split the dataset based on pose information. The meaning of
  the pose id is explained here:
  http://vision.ucsd.edu/~leekc/ExtYaleDatabase/Yale%20Face%20Database.htm
  Essentially, we are taking the more frontal poses for training and using the
  more extreme poses for validation and testing.
  The task is to identify which one of the training subjects is present in the
  input image.

  There is one zip folder per subject. Inside of each these folders there are
  several pgm gray scale images, but also other files (an ambient image without
  the subject, *info files with the list of files).
  An example of filename is: yaleB39_P03A+070E-35.pgm in the format:
  yaleB<subject_id=39>P<pose_id=03>A<azimuth_value>E<elevaltion_value>.pgm


  Link:
  http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html

  Args:
    dataset_path: Path with downloaded datafiles.

  Returns:
    Metadata and generator functions.
  """
  zip_name = os.path.join(dataset_path, 'extended-yale-dataset-b.zip')
  with zipfile.ZipFile(zip_name, 'r') as z:
    directories = z.namelist()
    class_names = _get_all_class_names(directories)

    assert len(class_names) == _NUM_CLASSES, (len(class_names), _NUM_CLASSES)
    label_str_to_int = {}
    for int_id, subject_id in enumerate(class_names):
      label_str_to_int[subject_id] = int_id

  metadata = types.DatasetMetaData(
      num_channels=1,
      num_classes=len(class_names),
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_to_id=label_str_to_int,
          labels=class_names,
          task_type='classification',
          image_type='face'))

  def path_to_label(path: str) -> int:
    fname, extension = os.path.splitext(os.path.basename(path))
    assert extension == '.pgm'
    subject_id, _ = fname.split('_')
    class_id = label_str_to_int[subject_id]
    return class_id

  def gen(split):

    def path_filter_fn(path: str) -> bool:
      image_id, extension = os.path.splitext(os.path.basename(path))
      if extension != '.pgm':
        return False
      pose = image_id[8:11]
      assert pose[0] == 'P'
      return pose in _POSES[split]

    return utils.generate_images_from_zip_files(
        dataset_path=dataset_path,
        zip_file_names=[zip_name],
        path_to_label_fn=path_to_label,
        ignored_files_regex=_IGNORED_FILES_REGEX,
        path_filter=path_filter_fn)

  per_split_gen = {}
  for split in _SPLITS:
    per_split_gen[split] = gen(split)

  return metadata, per_split_gen


extended_yaleb_dataset = types.DownloadableDataset(
    name='extended_yaleb',
    download_urls=[
        types.KaggleDataset(
            dataset_name='souvadrahati/extended-yale-dataset-b',
            checksum='ef37284be91fe0c81dcd96baa948a2db')
    ],
    handler=extended_yaleb_handler,
    paper_title='Acquiring Linear Subspaces for Face Recognition under Variable Lighting',
    authors='Kuang-Chih Lee, Jeffrey Ho, and David Kriegman',
    year='2005',
    website_url='http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html')
