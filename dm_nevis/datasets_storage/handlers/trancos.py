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

"""Trancos dataset handler."""

import os
import tarfile
from typing import List
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
import numpy as np
from PIL import Image

from tensorflow.io import gfile


_NUM_CLASSES = 10
_IMAGE_PATH = 'TRANCOS_v3/images/'
_SETS_PATH = 'TRANCOS_v3/image_sets/'


def trancos_handler(dataset_path: str) -> types.HandlerOutput:
  """Imports Trancos dataset.

  Link: https://gram.web.uah.es/data/datasets/trancos/index.html

  Args:
    dataset_path: Path with downloaded datafiles.

  Returns:
    Metadata and generator functions.
  """
  dfile = gfile.listdir(dataset_path)
  assert len(dfile) == 1
  dfile = dfile[0]

  def extract_tarinfos(tfile, startstr, endstr):
    return [
        tarinfo for tarinfo in tfile.getmembers()
        if (tarinfo.name.startswith(startstr) and
            tarinfo.name.endswith(endstr))]

  def extract_class_name(path: str, suffix_length: int) -> str:
    return path.split('/')[-1][:-suffix_length]

  # Each image has an associated txt file where each line
  # contains the x,y coordinate of a vehicle.
  # The number of cars is the label we want to predict, in principle.
  # In practice, we will declare the label to be the quantized value
  # of the number of cars in the image, turning counting in standard
  # classification.
  with tarfile.open(os.path.join(dataset_path, dfile), 'r:gz') as tfile:
    label_tarinfos = extract_tarinfos(tfile, _IMAGE_PATH, '.txt')
    imageid_to_count = dict()
    max_size = 200
    histogram = np.zeros(max_size)
    for ltar in label_tarinfos:
      f_obj = tfile.extractfile(ltar)
      assert f_obj
      count = len(f_obj.readlines())
      image_name = extract_class_name(ltar.name, 3) + 'jpg'
      imageid_to_count[image_name] = count
      if count > max_size:
        print('Warning: count above threshold')
        count = max_size - 1
      histogram[count] += 1
    # The idea is to divide the counts into buckets of contiguous values,
    # such that the number of examples in each bucket is roughly the same.
    # In order to do this, we first compute the cumulative sum of the empirical
    # distribution of counts, and then divide the cumulative density
    # distribution into equally sized buckets. This will make sure that each
    # bucket (class) contains rougly the same number of samples.
    tot_num_samples = histogram.sum()
    cumsum = np.cumsum(histogram)
    num_examples_per_bucket = tot_num_samples / _NUM_CLASSES
    intervals = []
    for cnt in range(1, _NUM_CLASSES):
      indices = np.where(cumsum < num_examples_per_bucket * cnt)
      assert indices[0].shape[0] > 0
      intervals.append(indices[0][-1])
    intervals.append(max_size)
    count_to_label = []
    label = 0
    prev_cnt = 0
    classname_to_label = dict()
    for cnt in range(max_size):
      if cnt > intervals[label]:
        classname = '%d-%d' % (prev_cnt, cnt)
        classname_to_label[classname] = label
        label += 1
        prev_cnt = cnt + 1
      count_to_label.append(label)
    classname = '%d-%d' % (prev_cnt, max_size)
    classname_to_label[classname] = label
    assert label == _NUM_CLASSES - 1

  # Extract list of training, valid, test files.
  def extract_image_list(sets_path, filename):
    with tarfile.open(os.path.join(dataset_path, dfile), 'r:gz') as tfile:
      tarinfo = extract_tarinfos(tfile, sets_path, filename)
      assert len(tarinfo) == 1
      f_obj = tfile.extractfile(tarinfo[0])
      assert f_obj
      lines = f_obj.readlines()
      lines = [line.decode('utf-8')[:-1] for line in lines]
      return lines

  train_set = extract_image_list(_SETS_PATH, 'training.txt')
  valid_set = extract_image_list(_SETS_PATH, 'validation.txt')
  test_set = extract_image_list(_SETS_PATH, 'test.txt')

  metadata = types.DatasetMetaData(
      num_classes=_NUM_CLASSES,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_to_id=classname_to_label,
          task_type='regression',
          image_type='counting'))

  def gen_split(imageset: List[str]):
    with tarfile.open(os.path.join(dataset_path, dfile), 'r:gz') as tfile:
      tarinfos = [
          tarinfo for tarinfo in tfile.getmembers()
          if tarinfo.name.split('/')[-1] in imageset]
      assert tarinfos
      for ti in tarinfos:
        f_obj = tfile.extractfile(ti)
        image = Image.open(f_obj)
        image.load()
        imageid = ti.name.split('/')[-1]
        count = imageid_to_count[imageid]
        label = count_to_label[count]
        yield (image, label)

  make_gen_fn = lambda: gen_split(train_set)
  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN_AND_DEV_ONLY,
      splits.MERGED_TRAIN_AND_DEV)
  per_split_gen['dev-test'] = gen_split(valid_set)
  per_split_gen['test'] = gen_split(test_set)
  return metadata, per_split_gen


trancos_dataset = types.DownloadableDataset(
    name='trancos',
    download_urls=[
        types.DownloadableArtefact(
            url='https://universidaddealcala-my.sharepoint.com/:u:/g/personal/gram_uah_es/Eank6osXQgxEqa-1bb0nVsoBc3xO4XDwENc_g0nc6t58BA?&Download=1',
            checksum='e9b4d5a62ab1fe5f542ec8326f2d4fda')
    ],
    handler=trancos_handler)
