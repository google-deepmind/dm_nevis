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

"""SUN-attributes handler."""

import os
import tarfile
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
import numpy as np
from PIL import Image
import scipy.io
import tensorflow_datasets as tfds


_IMAGES_FNAME = 'SUNAttributeDB_Images.tar.gz'
_ATTRIBUTE_FNAME = 'SUNAttributeDB.tar.gz'

_LABELS_FNAME = 'SUNAttributeDB/attributeLabels_continuous.mat'
_IMAGE_FILENAMES_FNAME = 'SUNAttributeDB/images.mat'
_NUM_CLASSES = 102


def sun_attributes_handler(dataset_path: str) -> types.HandlerOutput:
  """Handler for SUN-attributes dataset."""
  with tarfile.open(os.path.join(dataset_path, _ATTRIBUTE_FNAME), 'r') as tf:
    attributes = scipy.io.loadmat(tf.extractfile(_LABELS_FNAME))['labels_cv']
    image_filenames = scipy.io.loadmat(
        tf.extractfile(_IMAGE_FILENAMES_FNAME))['images']

  image_fname_to_attributes = {}
  for (image_fname, image_attributes) in zip(image_filenames, attributes):
    image_attributes = np.nonzero(image_attributes)[0].tolist()
    image_fname_to_attributes[os.path.join(
        'images', image_fname[0].item())] = image_attributes

  def make_gen():
    with tarfile.open(os.path.join(dataset_path, _IMAGES_FNAME), 'r|gz') as tf:
      for member in tf:
        image_fname = member.name
        if image_fname not in image_fname_to_attributes:
          continue
        attributes = image_fname_to_attributes[image_fname]
        image = Image.open(tf.extractfile(member))
        yield types.Example(image=image, multi_label=attributes, label=None)

  metadata = types.DatasetMetaData(
      num_classes=_NUM_CLASSES,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(),
      features=tfds.features.FeaturesDict({
          'multi_label':
              tfds.features.Sequence(
                  tfds.features.ClassLabel(num_classes=_NUM_CLASSES)),
          'png_encoded_image':
              tfds.features.Image()
      }))

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)

  return (metadata, per_split_gen)


sun_attributes_dataset = types.DownloadableDataset(
    name='sun_attributes',
    download_urls=[
        types.DownloadableArtefact(
            url='https://cs.brown.edu/~gmpatter/Attributes/SUNAttributeDB_Images.tar.gz',
            checksum='5966725c7306df6e05cd3ada7f45a18b'),
        types.DownloadableArtefact(
            url='https://cs.brown.edu/~gmpatter/Attributes/SUNAttributeDB.tar.gz',
            checksum='883293e5b645822f6ae0046c6df54183')
    ],
    handler=sun_attributes_handler,
)
