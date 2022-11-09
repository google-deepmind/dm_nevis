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

"""VOC actions handler."""

import os
import tarfile
from typing import Dict, List

from absl import logging
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
import numpy as np
from PIL import Image
import tensorflow_datasets as tfds
import xmltodict


_TRAIN_DATA_FNAME = 'VOCtrainval_11-May-2012.tar'
_TEST_DATA_FNAME = 'VOC2012test.tar'
_DIR_PREFIX = 'VOCdevkit/VOC2012/ImageSets/Action'
_ANNOTATIONS_PREFIX = 'VOCdevkit/VOC2012/Annotations'
_IMAGES_PREFIX = 'VOCdevkit/VOC2012/JPEGImages'

_ACTIONS = [
    'jumping',
    'phoning',
    'playinginstrument',
    'reading',
    'ridingbike',
    'ridinghorse',
    'running',
    'takingphoto',
    'usingcomputer',
    'walking',
    'other',
]
_NUM_CLASSES = 11


def actions_to_ids(actions: Dict[str, str],
                   label_to_id: Dict[str, int]) -> List[int]:
  action_ids = np.zeros((_NUM_CLASSES,))
  for action, active in actions.items():
    if int(active) == 1:
      action_ids[label_to_id[action]] = 1
  return np.nonzero(action_ids)[0].tolist()


def _extract_image_fnames_for_subset(subset: str, tf: tarfile.TarFile,
                                     actions: List[str]):
  """Extract image filenames for given subset and a set of actions."""
  result = set()
  for action in actions:
    if action == 'other':
      continue
    actions_file = tf.extractfile(
        os.path.join(_DIR_PREFIX, f'{action}_{subset}.txt'))
    if actions_file is not None:
      for line in actions_file:
        (image_fname, _,
         _) = line.decode('utf-8').strip().replace('  ', ' ').split(' ')
        result.add(image_fname)
  return result


def voc_actions_handler(dataset_path: str) -> types.HandlerOutput:
  """VOC actions dataset handler."""

  label_to_id = dict(
      ((label, label_id) for label_id, label in enumerate(_ACTIONS)))

  with tarfile.open(os.path.join(dataset_path, _TRAIN_DATA_FNAME), 'r') as tf:
    image_fnames = _extract_image_fnames_for_subset('trainval', tf, _ACTIONS)

  def make_gen(data_fname, image_fnames):
    with tarfile.open(os.path.join(dataset_path, data_fname), 'r') as tf:
      for image_fname in image_fnames:
        annotations_fname = os.path.join(_ANNOTATIONS_PREFIX,
                                         f'{image_fname}.xml')
        try:
          annotations_file = tf.extractfile(annotations_fname)
        except KeyError:
          logging.warning('Skipping: %s', image_fname)
          continue
        annotations = xmltodict.parse(annotations_file)
        if annotations is None:
          continue
        image = Image.open(
            tf.extractfile(os.path.join(_IMAGES_PREFIX, f'{image_fname}.jpg')))

        objects = annotations['annotation']['object']

        if not isinstance(objects, list):
          objects = [objects]

        for image_object in objects:
          bndbox = image_object['bndbox']
          xmin = int(float(bndbox['xmin']))
          xmax = int(float(bndbox['xmax']))
          ymin = int(float(bndbox['ymin']))
          ymax = int(float(bndbox['ymax']))
          action_image = image.crop((xmin, ymin, xmax, ymax))
          action_annotations = actions_to_ids(image_object['actions'],
                                              label_to_id)
          yield types.Example(
              image=action_image, multi_label=action_annotations, label=None)

  metadata = types.DatasetMetaData(
      num_classes=_NUM_CLASSES,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_to_id=label_to_id,
          task_type='multi-label',
          image_type='object',
      ),
      features=tfds.features.FeaturesDict({
          'multi_label':
              tfds.features.Sequence(
                  tfds.features.ClassLabel(num_classes=_NUM_CLASSES)),
          'png_encoded_image':
              tfds.features.Image()
      }))

  make_gen_fn = lambda: make_gen(_TRAIN_DATA_FNAME, image_fnames)
  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)
  return metadata, per_split_gen


# pylint: disable=line-too-long
voc_actions_dataset = types.DownloadableDataset(
    name='voc_actions',
    download_urls=[
        types.DownloadableArtefact(
            url='http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar',
            checksum='6cd6e144f989b92b3379bac3b3de84fd'),
        types.DownloadableArtefact(
            url='http://pjreddie.com/media/files/VOC2012test.tar',
            checksum='9065beb292b6c291fad82b2725749fda'
        )  # This requires authorisation
    ],
    website_url='http://host.robots.ox.ac.uk/pascal/VOC/voc2012',
    handler=voc_actions_handler)
