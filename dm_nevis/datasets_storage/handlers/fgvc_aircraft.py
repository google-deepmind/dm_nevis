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

"""FGVC Aircraft dataset handler."""
import enum
import functools
import os
import re
import tarfile

from dm_nevis.datasets_storage.handlers import extraction_utils as utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types

from tensorflow.io import gfile


# TODO: revisit to have one single dataset instead of 3.

_FAMILY_CLASS_NAMES = 'fgvc-aircraft-2013b/data/families.txt'
_MANUFACTURER_CLASS_NAMES = 'fgvc-aircraft-2013b/data/manufacturers.txt'
_VARIANT_CLASS_NAMES = 'fgvc-aircraft-2013b/data/variants.txt'

_IDX_CLASS_NAME_REGEX = r'^(\d+) (.+)$'


class LabelCategory(enum.Enum):
  FAMILY = 0
  MANUFACTURER = 1
  VARIANT = 2


def _get_split_filename(label_catgory: LabelCategory, split: str) -> str:
  return 'images_' + label_catgory.name.lower() + '_' + split


def _get_labels_filename(label_catgory: LabelCategory):
  if label_catgory == LabelCategory.FAMILY:
    return _FAMILY_CLASS_NAMES
  if label_catgory == LabelCategory.MANUFACTURER:
    return _MANUFACTURER_CLASS_NAMES
  if label_catgory == LabelCategory.VARIANT:
    return _VARIANT_CLASS_NAMES


def _get_class_name_list(fname: str, label_catgory: LabelCategory):
  """Get the label list for the label type."""
  labels_fname = _get_labels_filename(label_catgory)
  lines = []
  with tarfile.open(fname, 'r') as tfile:
    for member in tfile.getmembers():
      if member.isdir():
        continue
      if labels_fname in member.path:
        f_obj = tfile.extractfile(member)
        if f_obj:
          lines = f_obj.readlines()
  return [l.decode('utf-8').strip() for l in lines]


def _extract_idx_class_name_from_line(line: str):
  # `1345202 Cessna Citation` -> `Cessna Citation`
  match = re.match(_IDX_CLASS_NAME_REGEX, line)
  if not match:
    raise ValueError(f'Failed to match index and class for {line}')
  return match.groups()


def _get_idx_class_names_for_split(
    fname: str,
    label_catgory: LabelCategory,
    split: str):
  """Get the image filenames and corresponding labels in the split."""
  split_fname = _get_split_filename(label_catgory, split)
  idx_class_names = {'idx': [], 'class_names': []}

  with tarfile.open(fname, 'r') as tfile:
    for member in tfile.getmembers():
      if member.isdir():
        continue
      if split_fname in member.path:
        f_obj = tfile.extractfile(member)
        if f_obj:
          lines = f_obj.readlines()
          for l in lines:
            i, class_name = _extract_idx_class_name_from_line(l.decode('utf-8'))
            idx_class_names['idx'].append(int(i))
            idx_class_names['class_names'].append(class_name)
  return idx_class_names


def _extract_im_index_from_path(path: str) -> int:
  # 'fgvc-aircraft-2013b/data/images/1236289.jpg' -> 1236289
  return int(os.path.splitext(os.path.basename(path))[0])


def _fn_extract_label_from_path(class_name_list, idx_class_names):
  """Returns a function to get an integer label from path."""
  def _extract_label_from_path(path):
    label = None
    if 'images/' in path:
      im_index = _extract_im_index_from_path(path)
      try:
        i = idx_class_names['idx'].index(im_index)
        class_name = idx_class_names['class_names'][i]
        label = class_name_list.index(class_name)
      except ValueError:
        pass
    return label
  return _extract_label_from_path


def fgvc_aircraft_handler(dataset_path: str,
                          label_catgory: LabelCategory) -> types.HandlerOutput:
  """Imports FGVC Aircraft dataset.

  Link: https://paperswithcode.com/dataset/fgvc-aircraft-1
  The dataset comes with three label types (from finer to coarser):
   - Variant, e.g. Boeing 737-700. A variant collapses all the models that are
   visually indistinguishable into one class.
   - Family, e.g. Boeing 737.
   - Manufacturer, e.g. Boeing.
   For each type, 4 splits are provided, train, val, trainval and test. We
   keep only test and trainval.

  Args:
    dataset_path: Path with downloaded datafiles.
    label_catgory: One of ('family', 'manufacturer', 'variant').

  Returns:
    Metadata and generator functions.
  """

  label_valid_types = frozenset({LabelCategory.FAMILY,
                                 LabelCategory.MANUFACTURER,
                                 LabelCategory.VARIANT})
  assert label_catgory in label_valid_types, 'Unexpected label type'

  fname = os.path.join(dataset_path, gfile.listdir(dataset_path)[0])
  class_name_list = _get_class_name_list(fname, label_catgory)

  metadata = types.DatasetMetaData(
      num_channels=3,
      num_classes=len(class_name_list),
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          labels=class_name_list,
          task_type='classification',
          image_type='object'
      ))

  def gen_data_for_splits(fname, split):
    split_idx_class_names = _get_idx_class_names_for_split(
        fname, label_catgory, split)
    return utils.generate_images_from_tarfiles(
        fname,
        working_directory=dataset_path,
        path_to_label_fn=_fn_extract_label_from_path(
            class_name_list, split_idx_class_names))

  make_gen_fn = lambda: gen_data_for_splits(fname, 'trainval')
  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN,
      splits.MERGED_TRAIN_AND_DEV)
  per_split_gen['test'] = gen_data_for_splits(fname, 'test')

  return (metadata, per_split_gen)


# TODO: redundant DL
fgvc_aircraft_family_dataset = types.DownloadableDataset(
    name='fgvc_aircraft_family',
    download_urls=[types.DownloadableArtefact(
        url='https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz',
        checksum='d4acdd33327262359767eeaa97a4f732')],
    handler=functools.partial(
        fgvc_aircraft_handler, label_catgory=LabelCategory.FAMILY))

fgvc_aircraft_manufacturer_dataset = types.DownloadableDataset(
    name='fgvc_aircraft_manufacturer',
    download_urls=[types.DownloadableArtefact(
        url='https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz',
        checksum='d4acdd33327262359767eeaa97a4f732')],
    handler=functools.partial(
        fgvc_aircraft_handler, label_catgory=LabelCategory.MANUFACTURER))

fgvc_aircraft_variant_dataset = types.DownloadableDataset(
    name='fgvc_aircraft_variant',
    download_urls=[types.DownloadableArtefact(
        url='https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz',
        checksum='d4acdd33327262359767eeaa97a4f732')],
    handler=functools.partial(
        fgvc_aircraft_handler, label_catgory=LabelCategory.VARIANT))
