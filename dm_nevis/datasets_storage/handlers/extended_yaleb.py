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

from dm_nevis.datasets_storage.handlers import extraction_utils as utils
from dm_nevis.datasets_storage.handlers import types

from tensorflow.io import gfile


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

MD5 = [
    '9f500ec230e42dd306b9813b86890b50', 'ecc584a42165338ab4691272c74c89be',
    '7898e82336213e9880673826bab51984', '3332d7868f1f2654380be70029f1acbe',
    '5f6d1435eb5c379b489750df38c5ba0a', 'c9b6a6943cb1ecebc17c3d3c371c2b13',
    '5cc4aeb2e9fcc515302d0ae60445a332', 'c3bb2594b056c33e747b59128bd60f5b',
    '29c028ed8fc77f13a22fcc49cb00ad8a', 'd07e0cb25c50fe33bfbc95912eb8e156',
    'd46dc7bc85268e4ea7460a0262df1a9a', '3c2425f1dfb360ca89bd7f757e54c116',
    '73441bab2a59863bdc115c2905477754', '0e858e48bf52cfc7911c4ec618c744fa',
    'f89f08063899dc7da55116d87adc1eb0', '8f634995b0ebc1efcf61899082dec45a',
    '47c54d73d68b9b65e1f26636bef4f8f2', 'b10779da543b92123feff08892f8aa56',
    '096750421595332280880a46793d6aa7', '5811503cdf3a7bb1595d0431e6ebd4f8',
    'e8f3d4bc753b61714113fdef6d2c03f1', 'd02672184c458fd0a694731c5343abae',
    '1a416278420d7ea76101e870699c8a85', '5c3c3a3fc06be8bbe7337b586a3f4050',
    '8220f834edcba9f889daae0c60d23cca', 'e305481abddfd0a7e87d1a3b32b25622',
    '5d0b9285844743113afb022fa46d1072', '7d68d2d6b396d36bd4f61032553c242a'
]


def _get_all_class_names(filenames: List[str]) -> List[str]:
  names = []
  for fname in filenames:
    fields = fname.split('.')
    if len(fields) > 1 and fields[1] == 'zip':
      names.append(fields[0])
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

  filenames = gfile.listdir(dataset_path)
  class_names = _get_all_class_names(filenames)
  assert len(class_names) == _NUM_CLASSES
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
        zip_file_names=filenames,
        path_to_label_fn=path_to_label,
        ignored_files_regex=_IGNORED_FILES_REGEX,
        path_filter=path_filter_fn)

  per_split_gen = {}
  for split in _SPLITS:
    per_split_gen[split] = gen(split)

  return metadata, per_split_gen


file_list = []
prefix = 'http://vision.ucsd.edu/extyaleb/ExtendedYaleBZip/yaleB'
j = 0

for i in range(_FILES_ID_RANGE[0], _FILES_ID_RANGE[1]):
  if i != _MISSING_ID:  # Missing id.
    file_list.append(
        (types.DownloadableArtefact(url=f'{prefix}_{i}.zip', checksum=MD5[j])))
    j += 1

extended_yaleb_dataset = types.DownloadableDataset(
    name='extended_yaleb',
    download_urls=file_list,
    handler=extended_yaleb_handler,
    paper_title='Acquiring Linear Subspaces for Face Recognition under Variable Lighting',
    authors='Kuang-Chih Lee, Jeffrey Ho, and David Kriegman',
    year='2005',
    website_url='http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html')
