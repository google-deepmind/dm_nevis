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

"""ShanghaiTech dataset handler."""

import os
from typing import List
import zipfile
from dm_nevis.datasets_storage.handlers import types
import numpy as np
from PIL import Image
import scipy.io

from tensorflow.io import gfile


_NUM_CLASSES = 10
_PERC_DEV = 0.15
_PERC_DEV_TEST = 0.15


def shanghai_tech_handler(dataset_path: str) -> types.HandlerOutput:
  """Imports Shanghai Tech dataset.

  Link: https://www.kaggle.com/tthien/shanghaitech

  The dataset comes in two parts, A and B. Each has its own training and test
  split. The task is about counting how many people there are in the images.
  Similarly to other counting datasets, we quantize the counts into buckets with
  roughtly the same number of images. We also merge the training and test sets
  of part A and B into a single training and test set.

  The zip files are organized as follows:
  1) the folder ShanghaiTech/part_{A|B}/{train|test}_data/ground-truth contains
  matlab files storing the coordinates of each head in the image. One can get
  the number of people in the image by querying for instance:
  m = scipy.io.loadmat('GT_IMG_1.mat')
  m['image_info'][0][0][0][0][0].shape[0]
  2) the folder  ShanghaiTech/part_{A|B}/{train|test}_data/images contains jpeg
  images, mostly RGB with some gray scale image as well.


  Args:
    dataset_path: Path with downloaded datafiles.

  Returns:
    Metadata and generator functions.
  """
  dfile = gfile.listdir(dataset_path)
  assert len(dfile) == 1
  dfile = dfile[0]
  max_size = 5000
  counts = dict()
  histogram = np.zeros(max_size)
  training_files = []
  test_files = []
  train_files = []
  dev_files = []
  devtest_files = []
  train_dev_files = []

  def convert_filename(fullname):
    # Convert the label filename into image filename.
    pieces = fullname.split('/')
    filename = pieces[-1]
    filename_pieces = filename.split('_')
    return os.path.join(pieces[0], pieces[1], pieces[2],
                        'images',
                        filename_pieces[1] + '_' + filename_pieces[2][:-3] +
                        'jpg')

  with zipfile.ZipFile(os.path.join(dataset_path, dfile), 'r') as zf:
    # Go over all samples and collect the count information to build a
    # histogram, and the to eventually bucket the counts.
    for f in zf.infolist():
      if (f.is_dir() or 'shanghaitech_h5_empty' in f.filename or
          not f.filename.endswith('mat')):
        continue
      with zf.open(f) as fo:
        gf = scipy.io.loadmat(fo)
        curr_count = gf['image_info'][0][0][0][0][0].shape[0]
        image_name = convert_filename(f.filename)
        counts[image_name] = curr_count
        histogram[curr_count if (curr_count < max_size) else max_size-1] += 1
        if 'train' in image_name:
          training_files.append(image_name)
        else:
          test_files.append(image_name)
    # quantize the counts into equally sized buckets.
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
    # Compute count to label mapping,
    # and string to label mapping.
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

    # partition the original training set into train, dev and dev-test
    rng = np.random.default_rng(seed=1)
    tot_num_train_samples = len(training_files)
    shuffle = rng.permutation(tot_num_train_samples).tolist()
    ranges = [0, int((1-_PERC_DEV_TEST-_PERC_DEV)*tot_num_train_samples),
              int((1-_PERC_DEV_TEST)*tot_num_train_samples),
              tot_num_train_samples]
    train_ids = shuffle[ranges[0]:ranges[1]]
    dev_ids = shuffle[ranges[1]:ranges[2]]
    train_dev_ids = shuffle[ranges[0]:ranges[2]]
    devtest_ids = shuffle[ranges[2]:ranges[3]]
    train_files += [training_files[cc] for cc in train_ids]
    dev_files += [training_files[cc] for cc in dev_ids]
    train_dev_files += [training_files[cc] for cc in train_dev_ids]
    devtest_files += [training_files[cc] for cc in devtest_ids]

  metadata = types.DatasetMetaData(
      num_classes=_NUM_CLASSES,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_to_id=classname_to_label,
          task_type='regression',
          image_type='counting'))

  def gen_split(imageset: List[str]):
    with zipfile.ZipFile(os.path.join(dataset_path, dfile), 'r') as zf:
      all_files = [f.filename for f in zf.infolist()]
      for fname in imageset:
        assert fname in all_files
        image = Image.open(zf.open(fname)).convert('RGB')
        image.load()
        label = count_to_label[counts[fname]]
        yield (image, label)

  return (metadata, {
      'train': gen_split(train_files),
      'dev': gen_split(dev_files),
      'train_and_dev': gen_split(train_dev_files),
      'dev-test': gen_split(devtest_files),
      'test': gen_split(test_files),
  })


shanghai_tech_dataset = types.DownloadableDataset(
    name='shanghai_tech',
    download_urls=[
        types.KaggleDataset(
            dataset_name='tthien/shanghaitech',
            checksum='f547d65447063405ea78ab7fa9ae721b')
    ],
    handler=shanghai_tech_handler,
    paper_title='Single-Image Crowd Counting via Multi-Column Convolutional Neural Network',
    authors='Yingying Zhang, Desen Zhou, Siqin Chen, Shenghua Gao, Yi Ma',
    year='2016')
