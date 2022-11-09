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

"""MS-COCO handler."""

import collections
import io
import json
import os
import zipfile
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
import numpy as np
from PIL import Image
from tensorflow.io import gfile
import tensorflow_datasets as tfds


_TRAIN_IMAGES_FNAME = 'train2017.zip'
_VAL_IMAGES_FNAME = 'val2017.zip'
_TRAIN_ANNOTATIONS = 'annotations_trainval2017.zip'
_NUM_CLASSES = 90
_MIN_TOLERANCE = 16


def _single_label_gen(images_zip_path, data_info, images_info, prefix):
  """Produces a generator for single-label case."""
  with zipfile.ZipFile(images_zip_path) as zf:
    for annotation in data_info['annotations']:
      image_fname = images_info[annotation['image_id']]
      category_id = annotation['category_id'] - 1
      (x, y, width, height) = annotation['bbox']
      if width < _MIN_TOLERANCE or height < _MIN_TOLERANCE:
        continue
      image = Image.open(io.BytesIO(zf.read(os.path.join(prefix, image_fname))))
      x_max = min(image.width, x + width)
      y_max = min(image.height, y + height)
      cropped_image = image.crop((x, y, x_max, y_max))
      yield types.Example(
          image=cropped_image, label=category_id, multi_label=None)


def _multi_label_gen(images_zip_path, image_to_categories, images_info, prefix):
  """Produces a generator for multi-label case."""
  with zipfile.ZipFile(images_zip_path) as zf:
    for image_id, categories in image_to_categories.items():
      image_fname = images_info[image_id]
      image = Image.open(io.BytesIO(zf.read(os.path.join(prefix, image_fname))))
      yield types.Example(
          image=image,
          label=None,
          multi_label=np.nonzero(categories)[0].tolist())


def coco_handler(dataset_path: str,
                 is_multi_label: bool = True) -> types.HandlerOutput:
  """Handler for MS-COCO dataset."""

  train_images = {}
  val_images = {}

  default_categories_fn = lambda: np.zeros((_NUM_CLASSES,))

  train_images_to_categories = collections.defaultdict(default_categories_fn)
  val_images_to_categories = collections.defaultdict(default_categories_fn)

  with zipfile.ZipFile(os.path.join(dataset_path, _TRAIN_ANNOTATIONS)) as zf:
    with gfile.GFile(
        zf.extract('annotations/instances_train2017.json',
                   os.path.join(dataset_path)), 'r') as f:
      train_info = json.load(f)
      for image_info in train_info['images']:
        train_images[image_info['id']] = image_info['file_name']
      for annotation in train_info['annotations']:
        category_id = annotation['category_id'] - 1
        train_images_to_categories[annotation['image_id']][category_id] = 1

    with gfile.GFile(
        zf.extract('annotations/instances_val2017.json',
                   os.path.join(dataset_path)), 'r') as f:
      val_info = json.load(f)
      for image_info in val_info['images']:
        val_images[image_info['id']] = image_info['file_name']
      for annotation in val_info['annotations']:
        category_id = annotation['category_id'] - 1
        val_images_to_categories[annotation['image_id']][category_id] = 1

  metadata = types.DatasetMetaData(
      num_classes=_NUM_CLASSES,
      num_channels=3,
      image_shape=(),
      additional_metadata=dict(
          task_type='classification',
          image_type='object',
      ))

  if is_multi_label:
    metadata.features = tfds.features.FeaturesDict({
        'multi_label':
            tfds.features.Sequence(
                tfds.features.ClassLabel(num_classes=_NUM_CLASSES)),
        'png_encoded_image':
            tfds.features.Image()
    })

  def make_gen_single_label(is_test):
    if is_test:
      return _single_label_gen(
          os.path.join(dataset_path, _VAL_IMAGES_FNAME), val_info, val_images,
          'val2017')
    else:
      return _single_label_gen(
          os.path.join(dataset_path, _TRAIN_IMAGES_FNAME), train_info,
          train_images, 'train2017')

  def make_gen_multi_label(is_test):
    if is_test:
      return _multi_label_gen(
          os.path.join(dataset_path, _VAL_IMAGES_FNAME),
          val_images_to_categories, val_images, 'val2017')
    else:
      return _multi_label_gen(
          os.path.join(dataset_path, _TRAIN_IMAGES_FNAME),
          train_images_to_categories, train_images, 'train2017')

  def make_gen(is_test):
    if is_multi_label:
      return make_gen_multi_label(is_test)
    return make_gen_single_label(is_test)

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      lambda: make_gen(is_test=False), splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN,
      splits.MERGED_TRAIN_AND_DEV)
  per_split_gen['test'] = make_gen(is_test=True)

  return (metadata, per_split_gen)


# TODO: redundant DL
coco_single_label_dataset = types.DownloadableDataset(
    name='coco_single_label',
    download_urls=[
        types.DownloadableArtefact(
            url='http://images.cocodataset.org/zips/train2017.zip',
            checksum='cced6f7f71b7629ddf16f17bbcfab6b2'),
        types.DownloadableArtefact(
            url='http://images.cocodataset.org/zips/val2017.zip',
            checksum='442b8da7639aecaf257c1dceb8ba8c80'),
        types.DownloadableArtefact(
            url='http://images.cocodataset.org/zips/test2017.zip',
            checksum='77ad2c53ac5d0aea611d422c0938fb35'),
        types.DownloadableArtefact(
            url='http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
            checksum='f4bbac642086de4f52a3fdda2de5fa2c'),
        types.DownloadableArtefact(
            url='http://images.cocodataset.org/annotations/image_info_test2017.zip',
            checksum='85da7065e5e600ebfee8af1edb634eb5')
    ],
    handler=lambda ds: coco_handler(ds, is_multi_label=False),
    paper_title='Microsoft COCO: Common Objects in Context',
    authors='Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, Piotr Dollár',
    year='2014',
    website_url='https://cocodataset.org/#home',
)

coco_multi_label_dataset = types.DownloadableDataset(
    name='coco_multi_label',
    download_urls=[
        types.DownloadableArtefact(
            url='http://images.cocodataset.org/zips/train2017.zip',
            checksum='cced6f7f71b7629ddf16f17bbcfab6b2'),
        types.DownloadableArtefact(
            url='http://images.cocodataset.org/zips/val2017.zip',
            checksum='442b8da7639aecaf257c1dceb8ba8c80'),
        types.DownloadableArtefact(
            url='http://images.cocodataset.org/zips/test2017.zip',
            checksum='77ad2c53ac5d0aea611d422c0938fb35'),
        types.DownloadableArtefact(
            url='http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
            checksum='f4bbac642086de4f52a3fdda2de5fa2c'),
        types.DownloadableArtefact(
            url='http://images.cocodataset.org/annotations/image_info_test2017.zip',
            checksum='85da7065e5e600ebfee8af1edb634eb5')
    ],
    handler=lambda ds: coco_handler(ds, is_multi_label=True),
    paper_title='Microsoft COCO: Common Objects in Context',
    authors='Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, Piotr Dollár',
    year='2014',
    website_url='https://cocodataset.org/#home',
)
