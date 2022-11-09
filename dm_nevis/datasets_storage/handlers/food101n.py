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

"""Food 101 N handler."""

import io
import os
import zipfile
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
from PIL import Image


_FOOD_FNAME = 'Food-101N_release.zip'
_TRAIN_CLASSES_FNAME = 'Food-101N_release/meta/verified_train.tsv'
_TEST_CLASSES_FNAME = 'Food-101N_release/meta/verified_val.tsv'
_NUM_CLASSES = 101
_IMAGE_PREFIX = 'Food-101N_release/images'

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


def food101n_handler(dataset_path: str) -> types.HandlerOutput:
  """Food 101 N dataset handler."""

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

  with zipfile.ZipFile(os.path.join(dataset_path, _FOOD_FNAME)) as zf:
    train_lines = zf.read(_TRAIN_CLASSES_FNAME).decode('utf-8').split(
        '\n')[1:-1]
    train_files = {line.strip().split('\t')[0] for line in train_lines}
    test_lines = zf.read(_TEST_CLASSES_FNAME).decode('utf-8').split('\n')[1:-1]
    test_files = {line.strip().split('\t')[0] for line in test_lines}

  def make_gen(split_fnames, class_name_to_label):
    with zipfile.ZipFile(os.path.join(dataset_path, _FOOD_FNAME)) as zf:
      for fname in zf.namelist():
        if not fname.startswith(_IMAGE_PREFIX):
          continue
        label_name = os.path.basename(os.path.dirname(fname))
        image_fname = os.path.join(label_name, os.path.basename(fname))
        if image_fname not in split_fnames:
          continue
        image = Image.open(io.BytesIO(zf.read(fname))).convert('RGB')
        label = class_name_to_label[label_name]
        yield types.Example(image=image, label=label, multi_label=None)

  make_gen_fn = lambda: make_gen(train_files, label_to_id)

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN,
      splits.MERGED_TRAIN_AND_DEV)
  per_split_gen['test'] = make_gen(test_files, label_to_id)

  return metadata, per_split_gen


food101n_dataset = types.DownloadableDataset(
    name='food101n',
    download_urls=[
        types.DownloadableArtefact(
            url='https://iudata.blob.core.windows.net/food101/Food-101N_release.zip',
            checksum='596b41b48de43342ef1efbb2fd508e06')
    ],
    website_url='https://kuanghuei.github.io/Food-101N/',
    handler=food101n_handler)
