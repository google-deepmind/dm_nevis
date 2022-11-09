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

"""Belgium TSC dataset handler."""

import os
from typing import Sequence
import zipfile
from dm_nevis.datasets_storage.handlers import extraction_utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
from PIL import Image

from tensorflow.io import gfile


TRAIN_ZIP_PATH = "BelgiumTSC_Training.zip"
TEST_ZIP_PATH = "BelgiumTSC_Testing.zip"
CLASS_NAMES_PATH = "reducedSetTS.txt"

IGNORED_FILES_REGEX = r".*\.csv$|.*\.txt$"


def belgium_tsc_handler(artifacts_path: str) -> types.HandlerOutput:
  """Imports Belgium Traffic Sign Classification dataset.

  The dataset home page is at http://people.ee.ethz.ch/~timofter/traffic_signs/.

  Args:
    artifacts_path: Path with downloaded artifacts.

  Returns:
    Metadata and generator functions.
  """
  with gfile.GFile(os.path.join(artifacts_path, CLASS_NAMES_PATH), "r") as f:
    class_names = f.readlines()[1:]
    assert len(class_names) == 62

  metadata = types.DatasetMetaData(
      num_classes=62,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata={
          "class_names": class_names,
      })

  def gen(path):
    return extraction_utils.generate_images_from_zip_files(
        artifacts_path, [path],
        _label_from_path,
        ignored_files_regex=IGNORED_FILES_REGEX)

  make_gen_fn = lambda: gen(TRAIN_ZIP_PATH)
  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN,
      splits.MERGED_TRAIN_AND_DEV)
  per_split_gen["test"] = gen(TEST_ZIP_PATH)

  return metadata, per_split_gen


def write_fixture(path: str) -> None:
  """Writes a fixture TSC dataset to the given path."""

  with zipfile.ZipFile(os.path.join(path, TRAIN_ZIP_PATH), "w") as zf:
    _write_fixture_images(zf, "Training",
                          [0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])

  with zipfile.ZipFile(os.path.join(path, TEST_ZIP_PATH), "w") as zf:
    _write_fixture_images(zf, "Test", [3, 4, 5, 6])

  fake_class_names = ["ABC"] * 62
  with gfile.GFile(os.path.join(path, CLASS_NAMES_PATH), "w") as f:
    f.write("\n".join(["header", *fake_class_names]))


def _write_fixture_images(zf: zipfile.ZipFile, split_name: str,
                          class_indices: Sequence[int]) -> None:
  for class_index in class_indices:
    path = os.path.join(split_name, f"{class_index:05}", "01957_00002.ppm")
    with zf.open(path, "w") as f:
      image = Image.new("RGBA", size=(50, 50), color=(155, 0, 0))
      image.save(f, "ppm")


def _label_from_path(path: str) -> types.Label:
  return int(os.path.basename(os.path.dirname(path)))


belgium_tsc_dataset = types.DownloadableDataset(
    name="belgium_tsc",
    download_urls=[
        types.DownloadableArtefact(
            url="https://btsd.ethz.ch/shareddata/BelgiumTS/reducedSetTS.txt",
            checksum="e6052a024e24060e5cec84fbda34fb5e"),
        types.DownloadableArtefact(
            url="http://www.vision.ee.ethz.ch/~timofter/BelgiumTSC/BelgiumTSC_Training.zip",
            checksum="c727ca9d00e3964ca676286a1808ccee"),
        types.DownloadableArtefact(
            url="http://www.vision.ee.ethz.ch/~timofter/BelgiumTSC/BelgiumTSC_Testing.zip",
            checksum="d208de4566388791c0028da8d6a545cc"),
    ],
    handler=belgium_tsc_handler,
    fixture_writer=write_fixture)
