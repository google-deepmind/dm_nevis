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

"""Caltech256 dataset handler."""

import re

from dm_nevis.datasets_storage.handlers import extraction_utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types

LABELS_TO_ID = {
    "ak47": 0,
    "american-flag": 1,
    "backpack": 2,
    "baseball-bat": 3,
    "baseball-glove": 4,
    "basketball-hoop": 5,
    "bat": 6,
    "bathtub": 7,
    "bear": 8,
    "beer-mug": 9,
    "billiards": 10,
    "binoculars": 11,
    "birdbath": 12,
    "blimp": 13,
    "bonsai-101": 14,
    "boom-box": 15,
    "bowling-ball": 16,
    "bowling-pin": 17,
    "boxing-glove": 18,
    "brain-101": 19,
    "breadmaker": 20,
    "buddha-101": 21,
    "bulldozer": 22,
    "butterfly": 23,
    "cactus": 24,
    "cake": 25,
    "calculator": 26,
    "camel": 27,
    "cannon": 28,
    "canoe": 29,
    "car-tire": 30,
    "cartman": 31,
    "cd": 32,
    "centipede": 33,
    "cereal-box": 34,
    "chandelier-101": 35,
    "chess-board": 36,
    "chimp": 37,
    "chopsticks": 38,
    "cockroach": 39,
    "coffee-mug": 40,
    "coffin": 41,
    "coin": 42,
    "comet": 43,
    "computer-keyboard": 44,
    "computer-monitor": 45,
    "computer-mouse": 46,
    "conch": 47,
    "cormorant": 48,
    "covered-wagon": 49,
    "cowboy-hat": 50,
    "crab-101": 51,
    "desk-globe": 52,
    "diamond-ring": 53,
    "dice": 54,
    "dog": 55,
    "dolphin-101": 56,
    "doorknob": 57,
    "drinking-straw": 58,
    "duck": 59,
    "dumb-bell": 60,
    "eiffel-tower": 61,
    "electric-guitar-101": 62,
    "elephant-101": 63,
    "elk": 64,
    "ewer-101": 65,
    "eyeglasses": 66,
    "fern": 67,
    "fighter-jet": 68,
    "fire-extinguisher": 69,
    "fire-hydrant": 70,
    "fire-truck": 71,
    "fireworks": 72,
    "flashlight": 73,
    "floppy-disk": 74,
    "football-helmet": 75,
    "french-horn": 76,
    "fried-egg": 77,
    "frisbee": 78,
    "frog": 79,
    "frying-pan": 80,
    "galaxy": 81,
    "gas-pump": 82,
    "giraffe": 83,
    "goat": 84,
    "golden-gate-bridge": 85,
    "goldfish": 86,
    "golf-ball": 87,
    "goose": 88,
    "gorilla": 89,
    "grand-piano-101": 90,
    "grapes": 91,
    "grasshopper": 92,
    "guitar-pick": 93,
    "hamburger": 94,
    "hammock": 95,
    "harmonica": 96,
    "harp": 97,
    "harpsichord": 98,
    "hawksbill-101": 99,
    "head-phones": 100,
    "helicopter-101": 101,
    "hibiscus": 102,
    "homer-simpson": 103,
    "horse": 104,
    "horseshoe-crab": 105,
    "hot-air-balloon": 106,
    "hot-dog": 107,
    "hot-tub": 108,
    "hourglass": 109,
    "house-fly": 110,
    "human-skeleton": 111,
    "hummingbird": 112,
    "ibis-101": 113,
    "ice-cream-cone": 114,
    "iguana": 115,
    "ipod": 116,
    "iris": 117,
    "jesus-christ": 118,
    "joy-stick": 119,
    "kangaroo-101": 120,
    "kayak": 121,
    "ketch-101": 122,
    "killer-whale": 123,
    "knife": 124,
    "ladder": 125,
    "laptop-101": 126,
    "lathe": 127,
    "leopards-101": 128,
    "license-plate": 129,
    "lightbulb": 130,
    "light-house": 131,
    "lightning": 132,
    "llama-101": 133,
    "mailbox": 134,
    "mandolin": 135,
    "mars": 136,
    "mattress": 137,
    "megaphone": 138,
    "menorah-101": 139,
    "microscope": 140,
    "microwave": 141,
    "minaret": 142,
    "minotaur": 143,
    "motorbikes-101": 144,
    "mountain-bike": 145,
    "mushroom": 146,
    "mussels": 147,
    "necktie": 148,
    "octopus": 149,
    "ostrich": 150,
    "owl": 151,
    "palm-pilot": 152,
    "palm-tree": 153,
    "paperclip": 154,
    "paper-shredder": 155,
    "pci-card": 156,
    "penguin": 157,
    "people": 158,
    "pez-dispenser": 159,
    "photocopier": 160,
    "picnic-table": 161,
    "playing-card": 162,
    "porcupine": 163,
    "pram": 164,
    "praying-mantis": 165,
    "pyramid": 166,
    "raccoon": 167,
    "radio-telescope": 168,
    "rainbow": 169,
    "refrigerator": 170,
    "revolver-101": 171,
    "rifle": 172,
    "rotary-phone": 173,
    "roulette-wheel": 174,
    "saddle": 175,
    "saturn": 176,
    "school-bus": 177,
    "scorpion-101": 178,
    "screwdriver": 179,
    "segway": 180,
    "self-propelled-lawn-mower": 181,
    "sextant": 182,
    "sheet-music": 183,
    "skateboard": 184,
    "skunk": 185,
    "skyscraper": 186,
    "smokestack": 187,
    "snail": 188,
    "snake": 189,
    "sneaker": 190,
    "snowmobile": 191,
    "soccer-ball": 192,
    "socks": 193,
    "soda-can": 194,
    "spaghetti": 195,
    "speed-boat": 196,
    "spider": 197,
    "spoon": 198,
    "stained-glass": 199,
    "starfish-101": 200,
    "steering-wheel": 201,
    "stirrups": 202,
    "sunflower-101": 203,
    "superman": 204,
    "sushi": 205,
    "swan": 206,
    "swiss-army-knife": 207,
    "sword": 208,
    "syringe": 209,
    "tambourine": 210,
    "teapot": 211,
    "teddy-bear": 212,
    "teepee": 213,
    "telephone-box": 214,
    "tennis-ball": 215,
    "tennis-court": 216,
    "tennis-racket": 217,
    "theodolite": 218,
    "toaster": 219,
    "tomato": 220,
    "tombstone": 221,
    "top-hat": 222,
    "touring-bike": 223,
    "tower-pisa": 224,
    "traffic-light": 225,
    "treadmill": 226,
    "triceratops": 227,
    "tricycle": 228,
    "trilobite-101": 229,
    "tripod": 230,
    "t-shirt": 231,
    "tuning-fork": 232,
    "tweezer": 233,
    "umbrella-101": 234,
    "unicorn": 235,
    "vcr": 236,
    "video-projector": 237,
    "washing-machine": 238,
    "watch-101": 239,
    "waterfall": 240,
    "watermelon": 241,
    "welding-mask": 242,
    "wheelbarrow": 243,
    "windmill": 244,
    "wine-bottle": 245,
    "xylophone": 246,
    "yarmulke": 247,
    "yo-yo": 248,
    "zebra": 249,
    "airplanes-101": 250,
    "car-side-101": 251,
    "faces-easy-101": 252,
    "greyhound": 253,
    "tennis-shoes": 254,
    "toad": 255,
    "clutter": 256
}

_OBJECT_CATEGORIES_PATH = "256_ObjectCategories.tar"

# Filnames in the archive look like this:
#  256_ObjectCategories/003.backpack/003_0001.jpg
#
# where 003 is the label-number 1..257 (one-based).
_FILE_PATH_REGEX = re.compile(
    r"256_ObjectCategories/(\d\d\d)\.(.+)/(\d\d\d)_(\d\d\d\d)\.jpg")


def caltech256_handler(download_path: str) -> types.HandlerOutput:
  """Imports Caltech256 dataset."""

  def path_to_label_fn(fname):
    fname_match = _FILE_PATH_REGEX.match(fname)
    if not fname_match:
      return None
    else:
      label_id = int(fname_match.group(1)) - 1
      label_str = fname_match.group(2)
      assert LABELS_TO_ID[label_str] == label_id
      return label_id

  def gen_split():
    yield from extraction_utils.generate_images_from_tarfiles(
        _OBJECT_CATEGORIES_PATH,
        working_directory=download_path,
        path_to_label_fn=path_to_label_fn,
        convert_mode="RGB")

  metadata = types.DatasetMetaData(
      num_classes=len(LABELS_TO_ID),
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_str_to_id=LABELS_TO_ID,
          task_type="classification",
          image_type="object"))

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      gen_split, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)

  return (metadata, per_split_gen)


caltech256_dataset = types.DownloadableDataset(
    name="caltech256",
    download_urls=[
        types.DownloadableArtefact(
            url="https://drive.google.com/u/0/uc?id=1r6o0pSROcV1_VwT4oSjA2FBUSCWGuxLK&export=download&confirm=y",
            checksum="67b4f42ca05d46448c6bb8ecd2220f6d")
    ],
    website_url="http://www.vision.caltech.edu/Image_Datasets/Caltech256/",
    paper_url="The Caltech 256. Caltech Technical Report.",
    authors=" Griffin, G. Holub, AD. Perona, P.",
    papers_with_code_url="https://paperswithcode.com/dataset/caltech-256",
    handler=caltech256_handler)
