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

"""Silhouettes handler."""

import functools
import os
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
from PIL import Image
import scipy.io


# pylint:disable=missing-function-docstring
def silhouettes_handler(dataset_path: str,
                        size: int = 16) -> types.HandlerOutput:
  silhouettes_fname = 'caltech101_silhouettes_%d.mat' % size
  silhouettes_split_fname = 'caltech101_silhouettes_%d_split1.mat' % size
  silhouettes = scipy.io.loadmat(os.path.join(dataset_path, silhouettes_fname))
  silhouettes_split = scipy.io.loadmat(
      os.path.join(dataset_path, silhouettes_split_fname))

  class_to_label = dict()
  for class_id, class_name in enumerate(silhouettes['classnames'].flatten()):
    class_to_label[class_name.item()] = class_id
  num_classes = len(class_to_label)

  def split_gen(data, split, size):
    if split != 'all':
      images, labels = data['%s_data' % split], data['%s_labels' % split]
      labels = labels[:, 0] - 1
    else:
      images, labels = data['X'], data['Y']
      labels = labels[0] - 1

    # Original images are in [0, 1]
    images *= 255
    for i in range(len(images)):
      image = Image.fromarray(images[i].reshape((size, size)))
      label = labels[i]
      yield image, label

  make_gen_fn = lambda: split_gen(silhouettes_split, 'train', size)
  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN_AND_DEV_ONLY,
      splits.MERGED_TRAIN_AND_DEV)
  per_split_gen['dev-test'] = split_gen(silhouettes_split, 'val', size)
  per_split_gen['test'] = split_gen(silhouettes_split, 'test', size)

  metadata = types.DatasetMetaData(
      num_channels=1,
      num_classes=num_classes,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(class_to_label=class_to_label))
  return metadata, per_split_gen


silhouettes_16_dataset = types.DownloadableDataset(
    name='silhouettes_16',
    download_urls=[
        types.DownloadableArtefact(
            url='https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_16.mat',
            checksum='c79e99a89e9306069ac91b462be1504a'),
        types.DownloadableArtefact(
            url='https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_16_split1.mat',
            checksum='3baf9e2c023aa4a187e1d1b92b5a734a')
    ],
    handler=functools.partial(silhouettes_handler, size=16))

silhouettes_28_dataset = types.DownloadableDataset(
    name='silhouettes_28',
    download_urls=[
        types.DownloadableArtefact(
            url='https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28.mat',
            checksum='1432d2809e8bf111f1104a234731ddb1'),
        types.DownloadableArtefact(
            url='https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28_split1.mat',
            checksum='4483e9c14b188fd09937f9ea6f9ea777')
    ],
    handler=functools.partial(silhouettes_handler, size=28))
