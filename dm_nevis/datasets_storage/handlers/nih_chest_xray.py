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

"""NIH CHEST X-RAY dataset handler."""

import os
from typing import Dict, List
import zipfile
from dm_nevis.datasets_storage.handlers import extraction_utils as utils
from dm_nevis.datasets_storage.handlers import splits as su
from dm_nevis.datasets_storage.handlers import types
from tensorflow.io import gfile
import tensorflow_datasets as tfds

_TRAIN_LIST_FNAME = 'train_val_list.txt'
_TEST_LIST_FNAME = 'test_list.txt'
_LABEL_FNAME = 'Data_Entry_2017.csv'

_LABELS = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
    'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'No Finding'
]

_IGNORED_FILES_REGEX = '|'.join([
    utils.DEFAULT_IGNORED_FILES_REGEX,
    r'.pdf',
    r'.txt',
    r'__MACOSX',
    r'DS_Store',
])


def _path_to_label_fn(path: str, file_to_labels: Dict[str,
                                                      List[int]]) -> List[int]:
  filename = os.path.basename(path)
  return file_to_labels[filename]


def nih_chest_xray_handler(dataset_path: str) -> types.HandlerOutput:
  """Imports NIH Chest X-Ray dataset.

  The dataset home page is at
  https://www.kaggle.com/datasets/nih-chest-xrays/data
  This dataset contains x-ray images of the lung area. There are over 110,000
  images in total, 15 labels. The task is multilabel classification, as each
  image can belong to multiple categories.
  The dataset comes as a single zip file. The file Data_Entry_2017.csv in the
  base directory hosts the labels for each image. For instance:
  00000001_001.png,Cardiomegaly|Emphysema,... means that this particular image
  has assigned two labels.
  The file train_val_list.txt and test_list.txt are a list of images to be used
  in the training/validation set, and test set respectively.
  For instance, the first three entried of test_list.txt are:
  00000003_000.png
  00000003_001.png
  00000003_002.png
  Then there are 12 image folders containing png images.

  Args:
    dataset_path: Path with downloaded artifacts.

  Returns:
    Metadata and generator functions.
  """
  zip_file_path, *other_files_in_directory = gfile.listdir(dataset_path)
  assert not other_files_in_directory

  label_to_id = dict(
      ((label, label_id) for label_id, label in enumerate(_LABELS)))

  num_classes = len(_LABELS)
  metadata = types.DatasetMetaData(
      num_classes=num_classes,
      num_channels=1,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_to_id=label_to_id,
          task_type='multi-label',
          image_type='xray',
      ),
      features=tfds.features.FeaturesDict({
          'multi_label':
              tfds.features.Sequence(
                  tfds.features.ClassLabel(num_classes=num_classes)),
          'png_encoded_image':
              tfds.features.Image()
      }))

  # Extract train/test splits and labels for each image
  paths_to_train_files = []
  paths_to_test_files = []
  file_to_label_ids = dict()
  with zipfile.ZipFile(os.path.join(dataset_path, zip_file_path), 'r') as zf:
    for name in sorted(zf.namelist()):
      f = zf.getinfo(name)
      if f.filename == _TRAIN_LIST_FNAME:
        with zf.open(name) as infile:
          paths_to_train_files = infile.readlines()
          paths_to_train_files = [
              line.decode('utf-8').strip() for line in paths_to_train_files
          ]
      if f.filename == _TEST_LIST_FNAME:
        with zf.open(name) as infile:
          paths_to_test_files = infile.readlines()
          paths_to_test_files = [
              line.decode('utf-8').strip() for line in paths_to_test_files
          ]
      if f.filename == _LABEL_FNAME:
        with zf.open(name) as infile:
          # Ignore column headers.
          infile.readline()
          for line in infile:
            fields = line.decode('utf-8').split(',')
            img_fname = fields[0]
            # There are multiple labels per example, each separated by '|'.
            labels = fields[1].split('|')
            file_to_label_ids[img_fname] = [label_to_id[lab] for lab in labels]

  # pylint:disable=g-long-lambda
  def make_gen_fn(file_list, file_to_label_ids):
    return utils.generate_images_from_zip_files_with_multilabels(
        dataset_path=dataset_path,
        zip_file_names=[zip_file_path],
        path_to_attributes_fn=lambda path: _path_to_label_fn(
            path, file_to_label_ids),
        ignored_files_regex=_IGNORED_FILES_REGEX,
        path_filter=lambda path: os.path.basename(path) in file_list,
    )

  train_split_gen_fn = lambda: make_gen_fn(paths_to_train_files,
                                           file_to_label_ids)
  per_split_gen = su.random_split_generator_into_splits_with_fractions(
      train_split_gen_fn, su.SPLIT_WITH_FRACTIONS_FOR_TRAIN,
      su.MERGED_TRAIN_AND_DEV)
  per_split_gen['test'] = make_gen_fn(paths_to_test_files, file_to_label_ids)

  return metadata, per_split_gen


nih_chest_xray_dataset = types.DownloadableDataset(
    name='nih_chest_xray',
    download_urls=[
        types.KaggleDataset(
            dataset_name='nih-chest-xrays/data',
            checksum='ddd3acbfa23adf60ac08312b3c4040e2')
    ],
    paper_title='ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases',
    authors='Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, Ronald M. Summers',
    year='2017',
    website_url='https://www.kaggle.com/datasets/nih-chest-xrays/data',
    handler=nih_chest_xray_handler)
