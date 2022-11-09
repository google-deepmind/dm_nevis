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

"""Flickr Material Database (FMD).

This dataset was taken from
http://people.csail.mit.edu/celiu/CVPR2010/FMD/index.html.
"""

import os
import zipfile
from dm_nevis.datasets_storage.handlers import extraction_utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
from PIL import Image

FMD_PATH = "FMD.zip"
IMAGE_SHAPE = (512, 384)
CLASS_NAMES = [
    "fabric",
    "foliage",
    "glass",
    "leather",
    "metal",
    "paper",
    "plastic",
    "stone",
    "water",
    "wood",
]

_ignored_files_regex = r"\.asv$|\.m$|\.db$"


def handler(artifacts_path: str) -> types.HandlerOutput:
  """Downloads the Flickr materials database."""

  metadata = types.DatasetMetaData(
      num_classes=len(CLASS_NAMES),
      num_channels=3,
      image_shape=IMAGE_SHAPE,
      additional_metadata={
          "class_names": CLASS_NAMES,
      })

  def gen():
    with zipfile.ZipFile(os.path.join(artifacts_path, FMD_PATH), "r") as zf:
      for img, label in extraction_utils.generate_images_from_zip(
          zf,
          path_to_label_fn=_path_to_label,
          ignored_files_regex=_ignored_files_regex,
          path_filter=lambda x: x.startswith("image"),
          convert_mode="RGB"):

        assert img.size == IMAGE_SHAPE
        yield img, label

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      gen, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)

  return metadata, per_split_gen


def write_fixture(path: str) -> None:
  """Write a fixture to the given path."""

  fixture_paths = [
      "image/fabric/fabric_moderate_019_new.jpg",
      "image/foliage/foliage_final_079_new.jpg",
      "image/glass/glass_moderate_025_new.jpg",
      "image/glass/glass_object_041_new.jpg",
      "image/leather/leather_object_027_new.jpg",
      "image/metal/metal_moderate_050_new.jpg",
      "image/paper/paper_object_001_new.jpg",
      "image/plastic/plastic_object_029_new.jpg",
      "image/stone/stone_object_023_new.jpg",
      "image/water/water_object_011_new.jpg",
      "image/wood/wood_object_037_new.jpg",
  ]

  with zipfile.ZipFile(os.path.join(path, FMD_PATH), "w") as zf:
    for fixture_path in fixture_paths:
      with zf.open(fixture_path, "w") as f:
        image = Image.new("RGB", size=IMAGE_SHAPE, color=(155, 0, 0))
        image.save(f, "jpeg")


def _path_to_label(path: str) -> int:
  return CLASS_NAMES.index(os.path.basename(os.path.dirname(path)))


flickr_material_database_dataset = types.DownloadableDataset(
    name="flickr_material_database",
    download_urls=[
        types.DownloadableArtefact(
            url="http://people.csail.mit.edu/celiu/CVPR2010/FMD/FMD.zip",
            checksum="0721ba72cd981aa9599a81bbfaaebd75")
    ],
    website_url="http://people.csail.mit.edu/celiu/CVPR2010/FMD/index.html",
    handler=handler,
    paper_title="Accuracy and speed of material categorization in real-world images",
    authors="L. Sharan, R. Rosenholtz, E. H. Adelson",
    year=2014,
    fixture_writer=write_fixture)
