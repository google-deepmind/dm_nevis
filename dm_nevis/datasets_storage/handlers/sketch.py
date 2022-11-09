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

"""Sketch handler."""

import os

from typing import Dict

from dm_nevis.datasets_storage.handlers import extraction_utils as eu
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types

from tensorflow.io import gfile


def _path_to_label_fn(path: str, label_to_id: Dict[str, int]) -> int:
  label = os.path.basename(os.path.dirname(path))
  return label_to_id[label]


def sketch_handler(dataset_path: str) -> types.HandlerOutput:
  """Handler for Sketch dataset."""
  files = gfile.listdir(dataset_path)

  labels = [
      'snail', 'candle', 'crane (machine)', 'parking meter', 'bathtub',
      'loudspeaker', 'bulldozer', 'skateboard', 'ant', 'radio', 'tennis-racket',
      'envelope', 'house', 'person walking', 'basket', 'rainbow', 'paper clip',
      'alarm clock', 'screwdriver', 'cactus', 'umbrella', 'carrot', 'fan',
      'kangaroo', 'bell', 't-shirt', 'hedgehog', 'santa claus', 'angel',
      'trousers', 'eyeglasses', 'pretzel', 'snake', 'elephant', 'frying-pan',
      'bread', 'rollerblades', 'tomato', 'cake', 'couch', 'pizza', 'crown',
      'cannon', 'baseball bat', 'moon', 'potted plant', 'cup', 'syringe',
      'pipe (for smoking)', 'hand', 'telephone', 'pumpkin', 'race car', 'table',
      'brain', 'snowboard', 'flashlight', 'cloud', 'helmet', 'monkey',
      'ice-cream-cone', 'wineglass', 'strawberry', 'speed-boat', 'cigarette',
      'pigeon', 'book', 'lion', 'rabbit', 'violin', 'grenade', 'skull',
      'car (sedan)', 'bee', 'head', 'spoon', 'ship', 'laptop', 'diamond',
      'church', 'spider', 'wheelbarrow', 'flower with stem', 'mosquito',
      'apple', 'giraffe', 'submarine', 'tree', 'harp', 'face', 'boomerang',
      'floor lamp', 'fork', 'horse', 'scissors', 'cell phone', 'comb',
      'beer-mug', 'helicopter', 'hot air balloon', 'bear (animal)', 'lightbulb',
      'trombone', 'computer monitor', 'computer-mouse', 'fire hydrant',
      'squirrel', 'camera', 'binoculars', 'sponge bob', 'streetlight', 'blimp',
      'satellite', 'rooster', 'key', 'windmill', 'duck', 'zebra', 'armchair',
      'skyscraper', 'bookshelf', 'shark', 'tablelamp', 'nose', 'truck', 'fish',
      'chandelier', 'bottle opener', 'mailbox', 'donut', 'door', 'chair',
      'castle', 'ipod', 'power outlet', 'wrist-watch', 'wheel', 'dog', 'bus',
      'foot', 'panda', 'megaphone', 'microscope', 'bench', 'snowman', 'mug',
      'sun', 'wine-bottle', 'suv', 'vase', 'door handle', 'arm', 'ear',
      'hot-dog', 'ladder', 'banana', 'toilet', 'pineapple', 'mushroom',
      'dolphin', 'mouse (animal)', 'owl', 'hat', 'palm tree', 'tv',
      'calculator', 'bicycle', 'standing bird', 'bowl', 'ashtray', 'trumpet',
      'microphone', 'saxophone', 'leaf', 'tire', 'bed', 'shovel', 'socks',
      'traffic light', 'feather', 'crab', 'frog', 'walkie talkie', 'purse',
      'hammer', 'parrot', 'rifle', 'sailboat', 'scorpion', 'head-phones',
      'backpack', 'toothbrush', 'guitar', 'human-skeleton', 'bridge', 'bush',
      'axe', 'sheep', 'mermaid', 'eye', 'pig', 'cabinet', 'sword',
      'satellite dish', 'keyboard', 'stapler', 'teacup', 'present', 'motorbike',
      'airplane', 'canoe', 'pear', 'teddy-bear', 'knife', 'butterfly', 'camel',
      'flying saucer', 'pickup truck', 'van', 'swan', 'sea turtle', 'teapot',
      'tractor', 'hourglass', 'tooth', 'lobster', 'hamburger', 'grapes',
      'mouth', 'parachute', 'dragon', 'person sitting', 'pen', 'train', 'tent',
      'tiger', 'revolver', 'suitcase', 'shoe', 'barn', 'flying bird', 'octopus',
      'seagull', 'space shuttle', 'cat', 'crocodile', 'piano', 'penguin', 'cow',
      'lighter'
  ]

  label_to_id = dict(
      ((label, label_id) for label_id, label in enumerate(labels)))

  metadata = types.DatasetMetaData(
      num_classes=250,
      num_channels=1,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_to_id=label_to_id,
          task_type='classification',
          image_type='object',
      ))

  ignored_files_regex = [eu.DEFAULT_IGNORED_FILES_REGEX, r'filelist.txt']
  ignored_files_regex = '|'.join(ignored_files_regex)

  def make_gen_fn():
    return eu.generate_images_from_zip_files(
        dataset_path,
        files,
        path_to_label_fn=lambda path: _path_to_label_fn(path, label_to_id),
        ignored_files_regex=ignored_files_regex)

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)

  return metadata, per_split_gen


sketch_dataset = types.DownloadableDataset(
    name='sketch',
    download_urls=[
        types.DownloadableArtefact(
            url='https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip',
            checksum='023123df86a928a5273e3ba11990d8fd')
    ],
    website_url='https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch',
    handler=sketch_handler)
