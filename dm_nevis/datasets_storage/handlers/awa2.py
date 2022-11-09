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

"""AWA2 dataset handler."""

# TODO: Add multi-label support.

import os
import re
from typing import Callable, Dict, List, Tuple
import zipfile
from dm_nevis.datasets_storage.handlers import extraction_utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
import numpy as np
from PIL import Image
from tensorflow.io import gfile
import tensorflow_datasets as tfds


_CLASS_NAME_REGEX = r'(?P<class_name>[a-z\+]+)'
_CLASS_REGEX = r'(?P<class_id>[\d]+)\t(%s)' % _CLASS_NAME_REGEX
_PREFIX = 'Animals_with_Attributes2'
_NUM_ATTRIBUTES = 85


# Resize images for this dataset to the given max size. The original images can
# have much larger sizes, however this is unnecessarily high for the task, and
# results in slower training due to increased decoding time.
MAXIMUM_IMAGE_SIZE = 256


def _read_classes_with_labels(classes_fname: str,
                              zf: zipfile.ZipFile) -> Dict[str, int]:
  """Reads class files for a given zipfile object together with labels."""
  classes_dict = dict()
  with zf.open(os.path.join(_PREFIX, classes_fname)) as f:
    line = f.readline()
    while line:
      result = re.search(_CLASS_REGEX, line.decode('utf-8'))
      classes_dict[result['class_name']] = int(result['class_id'])
      line = f.readline()
  return classes_dict


def _read_classes_without_labels(classes_fname: str,
                                 zf: zipfile.ZipFile) -> Tuple[str]:
  """Reads class files for a given zipfile object.."""
  classes_list = []
  with zf.open(os.path.join(_PREFIX, classes_fname)) as f:
    line = f.readline()
    while line:
      result = re.search(_CLASS_NAME_REGEX, line.decode('utf-8'))
      classes_list.append(result['class_name'])
      line = f.readline()
  return tuple(classes_list)


def _read_attributes(attributes_fname: str, zf: zipfile.ZipFile,
                     get_encoded_attributes_fn: Callable[[str], List[str]]):
  """Reads the attributes for given file and zipfile object."""
  attributes_matrix = []
  with zf.open(f'Animals_with_Attributes2/{attributes_fname}') as f:
    line = f.readline()
    while line:
      encoded_attributes = line.decode('utf-8').split('\n')[0]
      encoded_attributes = get_encoded_attributes_fn(encoded_attributes)
      attributes_matrix.append([float(elem) for elem in encoded_attributes])
      line = f.readline()
  return np.array(attributes_matrix)


def awa2_handler(dataset_path: str) -> types.HandlerOutput:
  """Imports AWA2 dataset.

  The AWA2 dataset is taken from https://cvml.ist.ac.at/AwA2/AwA2-data.zip.



  Args:
    dataset_path: The path to the locally downloaded data assuming that it was
      downloaded from the above link location.

  Returns:
    Metadata and iterables over data for each subset split.
  """
  files = gfile.listdir(dataset_path)

  with zipfile.ZipFile(os.path.join(dataset_path, files[0]), 'r') as zf:
    label_to_id = _read_classes_with_labels('classes.txt', zf)

    # Put labels in [0,...,num_classes]:
    label_to_id = dict(
        ((label, label_id - 1) for (label, label_id) in label_to_id.items()))

    train_classes = _read_classes_without_labels('trainclasses.txt', zf)
    test_classes = _read_classes_without_labels('testclasses.txt', zf)

    predicates_to_id = _read_classes_with_labels('predicates.txt', zf)
    # Put predicate_ids in [0,...,num_classes]:
    predicates_to_id = dict(
        ((predicate, predicate_id - 1)
         for (predicate, predicate_id) in predicates_to_id.items()))

    predicate_matrix_binary = _read_attributes(
        'predicate-matrix-binary.txt',
        zf,
        get_encoded_attributes_fn=lambda x: x.split(' ')).astype(int)
    predicate_matrix_continuous = _read_attributes(
        'predicate-matrix-continuous.txt',
        zf,
        get_encoded_attributes_fn=lambda x: re.findall(r'\d+\.\d+', x))

  metadata = types.DatasetMetaData(
      num_channels=3,
      num_classes=_NUM_ATTRIBUTES,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          image_type='object',
          task_type='multilabel',
          predicates_to_id=predicates_to_id,
          predicate_matrix_binary=predicate_matrix_binary,
          predicate_matrix_continuous=predicate_matrix_continuous,
          label_to_id=label_to_id,
          train_classes=train_classes,
          test_classes=test_classes),
      features=tfds.features.FeaturesDict({
          'multi_label':
              tfds.features.Sequence(
                  tfds.features.ClassLabel(num_classes=_NUM_ATTRIBUTES)),
          'png_encoded_image':
              tfds.features.Image()
      }))

  def gen_split(classes, label_to_id, predicate_matrix_binary):
    classes = set(classes)

    with zipfile.ZipFile(os.path.join(dataset_path, files[0]), 'r') as zf:
      for member in zf.infolist():
        if member.is_dir():
          continue
        if 'JPEGImages' not in member.filename:
          continue
        current_class = os.path.split(os.path.split(member.filename)[0])[1]
        if current_class not in classes:
          continue
        class_id = label_to_id[current_class]
        attributes = np.nonzero(predicate_matrix_binary[class_id])[0].tolist()
        image = Image.open(zf.open(member))
        image.load()
        image.convert('RGB')
        image = extraction_utils.resize_to_max_size(image, MAXIMUM_IMAGE_SIZE)
        yield types.Example(image=image, label=None, multi_label=attributes)

  def make_gen_fn():
    return gen_split(train_classes, label_to_id, predicate_matrix_binary)

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN,
      splits.MERGED_TRAIN_AND_DEV)
  per_split_gen['test'] = gen_split(test_classes, label_to_id,
                                    predicate_matrix_binary)

  return (metadata, per_split_gen)


awa2_dataset = types.DownloadableDataset(
    name='awa2',
    download_urls=[
        # TODO: Deal with multi labels.
        types.DownloadableArtefact(
            url='https://cvml.ist.ac.at/AwA2/AwA2-data.zip',
            checksum='eaa27cf799d5cf55af372356d7281b5e')
    ],
    handler=awa2_handler)
