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

"""Labeled Faces in the Wild Aligned Dataset handler."""

import os
from typing import List
import zipfile
from dm_nevis.datasets_storage.handlers import types
from PIL import Image


_FNAME = 'lfwa.zip'
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


def lfwa_handler(dataset_path: str) -> types.HandlerOutput:
  """LFW Aligned dataset.

  Please refer to LFW for comments on what this dataset is about, the task and
  structure of the package. This is a version of LFW where faces have been
  aligned, there is no other difference.

  Args:
    dataset_path: Path with downloaded artifacts.

  Returns:
    Metadata and generator functions.
  """
  data = dict()
  with zipfile.ZipFile(os.path.join(dataset_path, _FNAME), 'r') as zf:
    all_fnames = [ff.filename for ff in zf.infolist()]
    all_fnames = [f for f in all_fnames if os.path.splitext(f)[1] == '.jpg']
    for member in all_fnames:
      image_fname = os.path.basename(member)
      subject_name = os.path.basename(os.path.dirname(member))
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
    with zipfile.ZipFile(os.path.join(dataset_path, _FNAME), 'r') as zf:
      all_fnames = [ff.filename for ff in zf.infolist()]
      all_fnames = [f for f in all_fnames if os.path.splitext(f)[1] == '.jpg']
      for member in all_fnames:
        image_fname = os.path.basename(member)
        if image_fname not in splits_fnames[split]:
          continue
        index = splits_fnames[split].index(image_fname)
        label = splits_labels[split][index]
        image = Image.open(zf.open(member)).convert('RGB')
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


lfwa_dataset = types.DownloadableDataset(
    name='lfwa',
    # TODO: Fix download link.
    download_urls=[
        types.DownloadableArtefact(
            url='https://drive.google.com/u/0/uc?id=1p1wjaqpTh_5RHfJu4vUh8JJCdKwYMHCp&export=download&confirm=y',
            checksum='96313a4780499f939bc4a06d5bebaf7d')
    ],
    website_url='https://talhassner.github.io/home/projects/lfwa/index.html',
    handler=lfwa_handler,
    paper_title='Effective Face Recognition by Combining Multiple Descriptors and Learned Background Statistics',
    authors='Lior Wolf, Tal Hassner, and Yaniv Taigman',
    year='2011')
