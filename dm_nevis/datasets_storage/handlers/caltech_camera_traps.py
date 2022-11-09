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

"""Caltech Camera Traps handler."""

import json
import os
import tarfile

from typing import Dict
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
from PIL import Image


_IMAGES_FNAME = 'eccv_18_all_images_sm.tar.gz'
_ANNOTATIONS_FNAME = 'eccv_18_annotations.tar.gz'

_ECCV_PATH = 'eccv_18_annotation_files'
_TRAIN_ANNOTATIONS = 'train_annotations.json'
_TRANS_TEST_ANNOTATIONS = 'trans_test_annotations.json'
_TRANS_VAL_ANNOTATIONS = 'trans_val_annotations.json'

_CLASS_NAMES = [
    'bobcat',
    'opossum',
    'empty',
    'coyote',
    'raccoon',
    'bird',
    'dog',
    'cat',
    'squirrel',
    'rabbit',
    'skunk',
    'rodent',
    'badger',
    'deer',
    'car',
    'fox',
]


def _read_annotations(ann_fname: str, label_to_id: Dict[str, int],
                      tf: tarfile.TarFile) -> Dict[str, int]:
  """Creates a dictionary mapping image filename to correspondsing label."""
  annotations_data = json.load(
      tf.extractfile(os.path.join(_ECCV_PATH, ann_fname)))
  categories = annotations_data['categories']
  categories = {category['id']: category['name'] for category in categories}
  annotations = dict()
  for ann in annotations_data['annotations']:
    image_id = ann['image_id']
    category_id = ann['category_id']
    annotations[image_id] = label_to_id[categories[category_id]]
  return annotations


def caltech_camera_traps_handler(dataset_path: str) -> types.HandlerOutput:
  """Caltech Camera Traps handler."""

  label_to_id = {class_name: i for i, class_name in enumerate(_CLASS_NAMES)}

  metadata = types.DatasetMetaData(
      num_classes=16,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_to_id=label_to_id,
          task_type='classification',
          image_type='object',
      ))

  with tarfile.open(os.path.join(dataset_path, _ANNOTATIONS_FNAME)) as tf:
    train_annotations = _read_annotations(_TRAIN_ANNOTATIONS, label_to_id, tf)
    trans_val_ann = _read_annotations(_TRANS_VAL_ANNOTATIONS, label_to_id, tf)
    trans_test_ann = _read_annotations(_TRANS_TEST_ANNOTATIONS, label_to_id, tf)

  def make_gen_fn(annotations):
    with tarfile.open(os.path.join(dataset_path, _IMAGES_FNAME), 'r|gz') as tf:
      for member in tf:
        if member.isdir():
          continue
        image_id = os.path.basename(os.path.splitext(member.path)[0])
        if image_id not in annotations:
          continue
        label = annotations[image_id]
        image = Image.open(tf.extractfile(member)).convert('RGB')
        image.load()
        yield (image, label)

  make_train_gen_fn = lambda: make_gen_fn(train_annotations)
  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_train_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN_AND_DEV_ONLY,
      splits.MERGED_TRAIN_AND_DEV)
  per_split_gen['dev-test'] = make_gen_fn(trans_val_ann)
  per_split_gen['test'] = make_gen_fn(trans_test_ann)

  return metadata, per_split_gen


caltech_camera_traps_dataset = types.DownloadableDataset(
    name='caltech_camera_traps',
    download_urls=[
        types.DownloadableArtefact(
            url='https://lilablobssc.blob.core.windows.net/caltechcameratraps/eccv_18_all_images_sm.tar.gz',
            checksum='8143c17aa2a12872b66f284ff211531f'),
        types.DownloadableArtefact(
            url='https://lilablobssc.blob.core.windows.net/caltechcameratraps/eccv_18_annotations.tar.gz',
            checksum='66a1f481b44aa1edadf75c9cfbd27aba')
    ],
    website_url='https://lila.science/datasets/caltech-camera-traps',
    handler=caltech_camera_traps_handler)
