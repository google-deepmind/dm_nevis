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

"""Caltech Categories dataset handler.

This dataset has been created from
http://www.vision.caltech.edu/html-files/archive.html.
On this page, datasets containining cars, motorcycles, faces, leaves, airplanes,
and background categories are provided.

At the time of creation, the download links for the leaves, airplanes and
background categories were not available, and so these have been omitted.

This dataset may be created with a subset of the available categories. Each
category is assigned a class label.

Note that the class labels are not comparable across different subsets of the
dataset. The class labels always start at 0 and go to the maximum number of
available classes in the returned dataset.
"""

import io
import os
import tarfile
from typing import Iterable, Sequence, Set
from dm_nevis.datasets_storage.handlers import extraction_utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
from PIL import Image


CARS_2001_PATH = "cars_brad.tar"
CARS_1999_PATH = "cars_markus.tar"
FACES_PATH = "faces.tar"
MOTORCYCLES_2001_PATH = "motorbikes_side.tar"

AVAILABLE_CATEGORIES = frozenset([
    "cars_2001",
    "cars_1999",
    "motorcycles_2001",
])

_IGNORED_FILES_REGEX = r"^README$|\.mat$"


def category_to_class_name(category: str) -> str:
  return {
      "cars_2001": "car",
      "cars_1999": "car",
      "motorcycles_2001": "motorcycle",
  }[category]


def caltech_categories_handler(
    artifacts_path: str,
    *,
    categories: Iterable[str] = AVAILABLE_CATEGORIES) -> types.HandlerOutput:
  """Caltech Categories."""

  categories = set(categories)
  unknown_categories = categories - AVAILABLE_CATEGORIES
  if unknown_categories:
    raise ValueError(f"Categories `{unknown_categories}` are not available")

  classes = _classes_from_categories(categories)

  metadata = types.DatasetMetaData(
      num_classes=len(classes),
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata={
          "class_names": classes,
      })

  def gen():
    for category in sorted(categories):
      path = _category_to_file_path(category)
      class_index = classes.index(category_to_class_name(category))

      def path_to_label_fn(_, label=class_index):
        return label

      yield from extraction_utils.generate_images_from_tarfiles(
          path,
          working_directory=artifacts_path,
          path_to_label_fn=path_to_label_fn,
          ignored_files_regex=_IGNORED_FILES_REGEX,
          convert_mode="RGB")

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      gen, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)

  return metadata, per_split_gen


def write_fixture(path: str) -> None:
  """Writes a fixture of the dataset to the given path."""

  for category in AVAILABLE_CATEGORIES:
    filename = _category_to_file_path(category)
    _write_fixture_images(os.path.join(path, filename), num_images=5)


def _write_fixture_images(path, num_images):
  """Writes a tarfile of fixture images to the given path."""

  with tarfile.open(path, "w") as tf:
    for i in range(num_images):
      image = Image.new("RGB", size=(50, 50), color=(155, 0, 0))
      buffer = io.BytesIO()
      image.save(buffer, "jpeg")
      buffer.seek(0)
      info = tarfile.TarInfo(f"image_{i:04}")
      info.size = len(buffer.getbuffer())
      tf.addfile(info, buffer)


def _classes_from_categories(categories: Set[str]) -> Sequence[str]:
  classes = set(category_to_class_name(category) for category in categories)
  return sorted(classes)


def _category_to_file_path(category: str) -> str:
  return {
      "cars_2001": CARS_2001_PATH,
      "cars_1999": CARS_1999_PATH,
      "motorcycles_2001": MOTORCYCLES_2001_PATH,
  }[category]


caltech_categories_dataset = types.DownloadableDataset(
    name="caltech_categories",
    download_urls=[
        types.DownloadableArtefact(
            url="http://www.vision.caltech.edu/Image_Datasets/cars_brad/cars_brad.tar",
            checksum="19c63604eb0fad32010aab0b342f744b"),
        types.DownloadableArtefact(
            url="http://www.vision.caltech.edu/Image_Datasets/cars_markus/cars_markus.tar",
            checksum="cd38f0d352299f9b1c00d46adb17db05"),
        types.DownloadableArtefact(
            url="http://www.vision.caltech.edu/Image_Datasets/motorbikes_side/motorbikes_side.tar",
            checksum="2d9ecca64baa8c39429d0ff8ba645f37"),
        types.DownloadableArtefact(
            url="http://www.vision.caltech.edu/Image_Datasets/faces/faces.tar",
            checksum="a6e5b794952e362560dba0cb6601307d")
    ],
    website_url="http://www.vision.caltech.edu/html-files/archive.html",
    handler=caltech_categories_handler,
    fixture_writer=write_fixture)
