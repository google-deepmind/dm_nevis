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

"""Food 101 handler."""

import os
import tarfile
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
from PIL import Image


_FOOD_FNAME = 'food-101.tar.gz'
_TRAIN_CLASSES_FNAME = 'food-101/meta/train.txt'
_TEST_CLASSES_FNAME = 'food-101/meta/test.txt'
_NUM_CLASSES = 101

_LABELS = [
    'takoyaki', 'bruschetta', 'lobster_bisque', 'bread_pudding', 'scallops',
    'pancakes', 'donuts', 'ceviche', 'grilled_salmon', 'ravioli', 'prime_rib',
    'waffles', 'eggs_benedict', 'beef_tartare', 'chicken_wings', 'clam_chowder',
    'panna_cotta', 'ramen', 'french_fries', 'seaweed_salad', 'lasagna',
    'fried_calamari', 'deviled_eggs', 'carrot_cake', 'strawberry_shortcake',
    'chocolate_mousse', 'poutine', 'beignets', 'caesar_salad', 'bibimbap',
    'garlic_bread', 'cheese_plate', 'shrimp_and_grits', 'caprese_salad',
    'beet_salad', 'dumplings', 'macarons', 'churros', 'samosa', 'creme_brulee',
    'miso_soup', 'french_onion_soup', 'risotto', 'pulled_pork_sandwich',
    'hot_and_sour_soup', 'onion_rings', 'spaghetti_bolognese', 'edamame',
    'beef_carpaccio', 'steak', 'grilled_cheese_sandwich', 'peking_duck',
    'frozen_yogurt', 'mussels', 'red_velvet_cake', 'oysters', 'greek_salad',
    'foie_gras', 'pho', 'spaghetti_carbonara', 'pad_thai', 'huevos_rancheros',
    'sashimi', 'sushi', 'gnocchi', 'hummus', 'pork_chop', 'falafel',
    'chicken_curry', 'breakfast_burrito', 'club_sandwich', 'cannoli',
    'chocolate_cake', 'fried_rice', 'apple_pie', 'guacamole',
    'macaroni_and_cheese', 'hot_dog', 'cup_cakes', 'paella', 'ice_cream',
    'escargots', 'spring_rolls', 'crab_cakes', 'croque_madame', 'hamburger',
    'baby_back_ribs', 'baklava', 'pizza', 'filet_mignon', 'cheesecake',
    'lobster_roll_sandwich', 'tiramisu', 'omelette', 'tacos', 'nachos', 'gyoza',
    'chicken_quesadilla', 'french_toast', 'tuna_tartare', 'fish_and_chips'
]


def food101_handler(dataset_path: str) -> types.HandlerOutput:
  """Food 101 dataset handler."""

  label_to_id = dict(
      ((label, label_id) for label_id, label in enumerate(_LABELS)))

  metadata = types.DatasetMetaData(
      num_classes=_NUM_CLASSES,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_to_id=label_to_id,
          task_type='classification',
          image_type='object',
      ))

  with tarfile.open(os.path.join(dataset_path, _FOOD_FNAME)) as tf:
    test_file = tf.extractfile(_TEST_CLASSES_FNAME)
    test_fnames = {fname.decode('utf-8').strip() for fname in test_file}
    train_file = tf.extractfile(_TRAIN_CLASSES_FNAME)
    train_fnames = {fname.decode('utf-8').strip() for fname in train_file}

  def make_gen(split_fnames, class_name_to_label):
    with tarfile.open(os.path.join(dataset_path, _FOOD_FNAME), 'r|gz') as tf:
      for member in tf:
        if member.isdir():
          continue
        path = member.path
        class_name = os.path.basename(os.path.dirname(path))
        image_fname_with_ext = os.path.basename(path)
        image_fname, _ = os.path.splitext(image_fname_with_ext)
        if os.path.join(class_name, image_fname) not in split_fnames:
          continue
        image = Image.open(tf.extractfile(member)).convert('RGB')
        label = class_name_to_label[class_name]
        image.load()
        yield (image, label)

  make_gen_fn = lambda: make_gen(train_fnames, label_to_id)

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN,
      splits.MERGED_TRAIN_AND_DEV)
  per_split_gen['test'] = make_gen(test_fnames, label_to_id)

  return metadata, per_split_gen


food101_dataset = types.DownloadableDataset(
    name='food101',
    download_urls=[
        types.DownloadableArtefact(
            url='http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz',
            checksum='85eeb15f3717b99a5da872d97d918f87')
    ],
    website_url='https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/',
    handler=food101_handler)
