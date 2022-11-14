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

"""MIT-scenes handler.

See https://paperswithcode.com/dataset/mit-indoors-scenes for more information.
"""

import io
import os
import zipfile
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
from PIL import Image

_ARCHIVE_FNAME = 'indoor-scenes-cvpr-2019.zip'
_TRAIN_IMAGES_FNAME = 'TrainImages.txt'
_TEST_IMAGES_FNAME = 'TestImages.txt'


def _path_to_label_fn(path: str, label_to_id):
  label = os.path.split(path)[1].split('_')[0]
  return label_to_id[label]


def mit_scenes_handler(dataset_path: str) -> types.HandlerOutput:
  """MIT indoor scenes dataset."""
  with zipfile.ZipFile(os.path.join(dataset_path, _ARCHIVE_FNAME), 'r') as zf:
    train_images_names = zf.read(_TRAIN_IMAGES_FNAME).decode('utf-8').split(
        '\n')
    test_images_names = zf.read(_TEST_IMAGES_FNAME).decode('utf-8').split('\n')

  labels = [
      'office', 'lobby', 'stairscase', 'winecellar', 'church_inside',
      'studiomusic', 'shoeshop', 'bowling', 'poolinside', 'nursery',
      'meeting_room', 'videostore', 'bathroom', 'library', 'locker_room',
      'movietheater', 'children_room', 'concert_hall', 'clothingstore',
      'pantry', 'subway', 'prisoncell', 'inside_bus', 'garage', 'warehouse',
      'bookstore', 'auditorium', 'laboratorywet', 'tv_studio', 'buffet',
      'waitingroom', 'laundromat', 'bedroom', 'greenhouse', 'cloister',
      'elevator', 'dining_room', 'hairsalon', 'livingroom', 'deli',
      'restaurant_kitchen', 'dentaloffice', 'trainstation', 'casino', 'bar',
      'jewelleryshop', 'kitchen', 'museum', 'grocerystore', 'operating_room',
      'airport_inside', 'gameroom', 'fastfood_restaurant', 'classroom',
      'bakery', 'closet', 'artstudio', 'hospitalroom', 'gym', 'florist',
      'inside_subway', 'toystore', 'kindergarden', 'restaurant', 'mall',
      'corridor', 'computerroom'
  ]
  label_to_id = dict(
      ((label, label_id) for label_id, label in enumerate(labels)))

  metadata = types.DatasetMetaData(
      num_classes=len(labels),
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_to_id=label_to_id,
          task_type='classification',
          image_type='scene',
      ))

  def gen(image_names, label_to_id, base_dir='indoorCVPR_09/Images'):
    with zipfile.ZipFile(os.path.join(dataset_path, _ARCHIVE_FNAME), 'r') as zf:
      for image_name in image_names:
        label = label_to_id[image_name.split('/')[0]]
        image_path = os.path.join(base_dir, image_name)
        image = Image.open(io.BytesIO(zf.read(image_path))).convert('RGB')
        image.load()
        yield (image, label)

  make_gen_fn = lambda: gen(train_images_names, label_to_id)
  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN,
      splits.MERGED_TRAIN_AND_DEV)
  per_split_gen['test'] = gen(test_images_names, label_to_id)

  return metadata, per_split_gen


mit_scenes_dataset = types.DownloadableDataset(
    name='mit_scenes',
    download_urls=[
        types.KaggleDataset(
            dataset_name='itsahmad/indoor-scenes-cvpr-2019',
            checksum='b5a8ee875edc974ab49f4cad3b8607da')
    ],
    website_url='https://www.kaggle.com/itsahmad/indoor-scenes-cvpr-2019',
    handler=mit_scenes_handler)
