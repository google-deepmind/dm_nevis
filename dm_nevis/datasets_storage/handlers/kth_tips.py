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

"""KTH-TIPS datasets handler."""
import functools
import os
import tarfile
from typing import Sequence

from dm_nevis.datasets_storage.handlers import extraction_utils as utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types

from tensorflow.io import gfile


def _get_class_name_list(fname: str) -> Sequence[str]:
  class_name_list = []
  with tarfile.open(fname, 'r') as tf:
    for member in tf.getmembers():
      if member.isdir() and '/' in member.path and 'sample' not in member.path:
        class_name_list.append(os.path.split(member.path)[-1])
  class_name_list.sort()
  return class_name_list


def _get_class_name_for_image(path: str) -> str:
  if 'sample' in path:
    # KTH-TIPS2-a/wool/sample_a/22a-scale_10_im_10_col.png -> wool
    return os.path.split(os.path.split(os.path.dirname(path))[0])[-1]
  else:
    # KTH_TIPS/linen/44-scale_3_im_1_col.png -> linen
    return os.path.split(os.path.dirname(path))[-1]


def _path_to_label_fn(classe_name_list):
  def _path_to_label(path):
    if '.png' in path:
      class_name = _get_class_name_for_image(path)
      return classe_name_list.index(class_name)
    else:
      return None
  return _path_to_label


def kth_tips_handler(dataset_path: str,
                     is_grey: bool) -> types.HandlerOutput:
  """Imports KTH-TIPS datasets.

  Link: https://www.csc.kth.se/cvap/databases/kth-tips/index.html
  This handler is valid for 4 datastes: KTH-TIPS (Colored and Grey),
  KTH-TIPS-2a, KTH-TIPS-2b

  Args:
    dataset_path: Path with downloaded datafiles.
    is_grey: True if images are grey, False otherwise.

  Returns:
    Metadata and generator functions.
  """
  fname = os.path.join(dataset_path, gfile.listdir(dataset_path)[0])
  class_name_list = _get_class_name_list(fname)

  metadata = types.DatasetMetaData(
      num_channels=1 if is_grey else 3,
      num_classes=len(class_name_list),
      image_shape=(),  # Ignored for now.
      preprocessing='random_crop',
      additional_metadata=dict(
          labels=class_name_list,
          task_type='classification',
          image_type='texture'
      ))

  def gen():
    return utils.generate_images_from_tarfiles(
        fname,
        working_directory=dataset_path,
        path_to_label_fn=_path_to_label_fn(class_name_list))

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      gen, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)

  return (metadata, per_split_gen)

# These 4 datasets contain different data from different artifacts but share
# the same logic

kth_tips_grey_dataset = types.DownloadableDataset(
    name='kth_tips_grey',
    download_urls=[
        types.DownloadableArtefact(
            url='https://www.csc.kth.se/cvap/databases/kth-tips/kth_tips_grey_200x200.tar',
            checksum='3aab2bffd539865b237cb3a63dffb14a')
    ],
    handler=functools.partial(kth_tips_handler, is_grey=True))

kth_tips_dataset = types.DownloadableDataset(
    name='kth_tips',
    download_urls=[
        types.DownloadableArtefact(
            url='https://www.csc.kth.se/cvap/databases/kth-tips/kth_tips_col_200x200.tar',
            checksum='4f92fe540feb4f3c66938291e4516f6c')
    ],
    handler=functools.partial(kth_tips_handler, is_grey=False))

kth_tips_2a_dataset = types.DownloadableDataset(
    name='kth_tips_2a',
    download_urls=[
        types.DownloadableArtefact(
            url='https://www.csc.kth.se/cvap/databases/kth-tips/kth-tips2-a_col_200x200.tar',
            checksum='911eb17220748fa36e6524aea71db7d7')
    ],
    handler=functools.partial(kth_tips_handler, is_grey=False))

kth_tips_2b_dataset = types.DownloadableDataset(
    name='kth_tips_2b',
    download_urls=[
        types.DownloadableArtefact(
            url='https://www.csc.kth.se/cvap/databases/kth-tips/kth-tips2-b_col_200x200.tar',
            checksum='00470a104a57f5a5be22cc8a0f234c4e')
    ],
    handler=functools.partial(kth_tips_handler, is_grey=False))
