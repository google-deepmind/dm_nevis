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

"""Mall dataset handler."""

import os
from typing import List
import zipfile
from dm_nevis.datasets_storage.handlers import types
import numpy as np
from PIL import Image
import scipy.io

from tensorflow.io import gfile


_NUM_CLASSES = 10
_PREFIX = 'mall_dataset/'
_LABEL_FILE = 'mall_gt.mat'
_NUM_TRAIN_IMAS = 800
_TOT_IMAS = 2000
_PERC_DEV = 0.15
_PERC_DEV_TEST = 0.15


def mall_handler(dataset_path: str) -> types.HandlerOutput:
  """Imports Mall dataset.

  Link: https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html

  Args:
    dataset_path: Path with downloaded datafiles.

  Returns:
    Metadata and generator functions.
  """
  dfile = gfile.listdir(dataset_path)
  assert len(dfile) == 1
  dfile = dfile[0]
  # The paper says they use the first 800 images for training,
  # and the rest for testing.
  rng = np.random.default_rng(seed=1)
  original_training = (rng.permutation(_NUM_TRAIN_IMAS) + 1).tolist()
  ranges = [0, int((1-_PERC_DEV_TEST-_PERC_DEV)*_NUM_TRAIN_IMAS),
            int((1-_PERC_DEV_TEST)*_NUM_TRAIN_IMAS), _NUM_TRAIN_IMAS]
  train_ids = original_training[ranges[0]:ranges[1]]
  dev_ids = original_training[ranges[1]:ranges[2]]
  train_dev_ids = original_training[ranges[0]:ranges[2]]
  devtest_ids = original_training[ranges[2]:ranges[3]]
  test_ids = list(range(_NUM_TRAIN_IMAS + 1, _TOT_IMAS + 1))
  max_size = 100
  counts = []

  with zipfile.ZipFile(os.path.join(dataset_path, dfile), 'r') as zf:
    # The unzip folder contains a folder, frames/, with
    # 2000 images.
    # In the base directory, there is a matlab file with the count values for
    # each image.
    # Similarly to Trancos, we are going to quantize these values and turn the
    # counting problem into classification. We'll bucket contiguous count values
    # in such a way that there is roughly the same amount of images in each
    # bucket.
    with zf.open(os.path.join(_PREFIX, _LABEL_FILE)) as fo:
      gf = scipy.io.loadmat(fo)
      counts += gf['count'][:, 0].tolist()
      tot_imas = len(counts)
      assert tot_imas == _TOT_IMAS
      # build a histogram
      histogram = np.zeros(max_size)
      for cnt in range(tot_imas):
        histogram[counts[cnt] if (counts[cnt] < max_size) else max_size-1] += 1
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

  metadata = types.DatasetMetaData(
      num_classes=_NUM_CLASSES,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_to_id=classname_to_label,
          task_type='regression',
          image_type='counting'))

  def gen_split(imageset: List[int]):
    with zipfile.ZipFile(os.path.join(dataset_path, dfile), 'r') as zf:
      for f in zf.infolist():
        if f.is_dir() or not f.filename.endswith('jpg'):
          continue
        name = int(f.filename.split('/')[-1][4:-4])
        if name not in imageset:
          continue
        image = Image.open(zf.open(f))
        image.load()
        # image names start from 1, as opposed to 0.
        label = count_to_label[counts[name - 1]]
        yield (image, label)

  return (metadata, {
      'train': gen_split(train_ids),
      'dev': gen_split(dev_ids),
      'train_and_dev': gen_split(train_dev_ids),
      'dev-test': gen_split(devtest_ids),
      'test': gen_split(test_ids),
  })


mall_dataset = types.DownloadableDataset(
    name='mall',
    download_urls=[
        types.DownloadableArtefact(
            url='https://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/mall_dataset.zip',
            checksum='48a772b5e55e8e9a66a3c8f98598fc3b')
    ],
    handler=mall_handler,
    paper_title='Feature Mining for Localised Crowd Counting',
    authors='K. Chen, C. C. Loy, S. Gong, and T. Xiang',
    year=2012)
