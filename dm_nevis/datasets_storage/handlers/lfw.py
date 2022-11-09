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

"""Labeled Faces in the Wild Dataset handler."""

import os
import tarfile
from typing import List
from dm_nevis.datasets_storage.handlers import types
from PIL import Image


_FNAME = 'lfw.tgz'
_MIN_NUM_IMGS = 5
_NUM_CLASSES = 423
_NUM_SAMPLES = 5985
_SPLITS = ['train', 'dev', 'dev-test', 'train_and_dev', 'test']


def _get_range(list_len: int, split: str) -> List[int]:
  """Returns range of indexes for a given split."""
  if split == 'train':
    return list(range(0, list_len - 4))
  elif split == 'dev':
    return list(range(list_len - 4, list_len - 3))
  elif split == 'dev-test':
    return list(range(list_len - 3, list_len - 2))
  elif split == 'train_and_dev':
    return list(range(0, list_len - 3))
  else:
    assert split == 'test'
    return list(range(list_len - 2, list_len))


def lfw_handler(dataset_path: str) -> types.HandlerOutput:
  """LFW dataset.

  LFW is originally a face verification dataset: Given a pair of images, predict
  whether the faces are from the same subject.
  Here we turn the face verification task into face classification: Given a
  single image, predict the subject id. This means that test images must contain
  subjects seen at training time. Therefore, we remove subjects that have less
  than MIN_NUM_IMGS examples, since we need to form a train/dev/dev-test/test
  splt.

  We take the last two images of each subject for testing. The one before for
  dev-test. The one before that one for dev, and the remaining for training.

  The original dataset has 13233 images from 5749 subjects. If we restrict the
  number of subjects to those that have at least 5 images, then we reduce the
  dataset to NUM_SAMPLES images and NUM_CLASSES subjects.

  Args:
    dataset_path: Path with downloaded artifacts.

  Returns:
    Metadata and generator functions.
  """
  data = dict()
  with tarfile.open(os.path.join(dataset_path, _FNAME), 'r|gz') as tf:
    for member in tf:
      if member.isdir():
        continue
      image_fname = os.path.basename(member.path)
      subject_name = os.path.basename(os.path.dirname(member.path))
      if subject_name not in data:
        data[subject_name] = [image_fname]
      else:
        data[subject_name].append(image_fname)
  splits_fnames = {
      'train': [],
      'dev': [],
      'train_and_dev': [],
      'dev-test': [],
      'test': []
  }
  splits_labels = {
      'train': [],
      'dev': [],
      'train_and_dev': [],
      'dev-test': [],
      'test': []
  }
  label_id = 0
  label_str_to_int = dict()
  tot_num_examples = 0
  for subject_name, file_list in data.items():
    n = len(file_list)
    if n < _MIN_NUM_IMGS:
      continue
    for split in _SPLITS:
      srange = _get_range(n, split)
      splits_labels[split] += [label_id] * len(srange)
      splits_fnames[split] += [file_list[i] for i in srange]
      label_str_to_int[subject_name] = label_id
    tot_num_examples += n
    label_id += 1
  assert label_id == _NUM_CLASSES
  assert tot_num_examples == _NUM_SAMPLES

  def gen(split):
    with tarfile.open(os.path.join(dataset_path, _FNAME), 'r|gz') as tf:
      for member in tf:
        if member.isdir():
          continue
        image_fname = os.path.basename(member.path)
        if image_fname not in splits_fnames[split]:
          continue
        index = splits_fnames[split].index(image_fname)
        label = splits_labels[split][index]
        image = Image.open(tf.extractfile(member)).convert('RGB')
        image.load()
        yield (image, label)

  metadata = types.DatasetMetaData(
      num_classes=_NUM_CLASSES,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_to_id=label_str_to_int,
          task_type='classification',
          image_type='face',
      ))

  per_split_gen = {}
  for split in _SPLITS:
    per_split_gen[split] = gen(split)

  return metadata, per_split_gen


lfw_dataset = types.DownloadableDataset(
    name='lfw',
    download_urls=[
        types.DownloadableArtefact(
            url='http://vis-www.cs.umass.edu/lfw/lfw.tgz',
            checksum='a17d05bd522c52d84eca14327a23d494')
    ],
    website_url='http://vis-www.cs.umass.edu/lfw/',
    handler=lfw_handler,
    paper_title='Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments',
    authors='Gary B. Huang, Manu Ramesh, Tamara Berg, and Erik Learned-Miller',
    year='2007')
