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

"""ISBI-ISIC 2017 melanoma classification challenge dataset handler."""

import os
from dm_nevis.datasets_storage.handlers import extraction_utils
from dm_nevis.datasets_storage.handlers import types
import numpy as np

from tensorflow.io import gfile


TRAIN_ZIP_PATH = "ISIC-2017_Training_Data.zip"
TRAIN_LABEL_PATH = "ISIC-2017_Training_Part3_GroundTruth.csv"
VALIDATION_ZIP_PATH = "ISIC-2017_Validation_Data.zip"
VALIDATION_LABEL_PATH = "ISIC-2017_Validation_Part3_GroundTruth.csv"
TEST_ZIP_PATH = "ISIC-2017_Test_v2_Data.zip"
TEST_LABEL_PATH = "ISIC-2017_Test_v2_Part3_GroundTruth.csv"

# Ratio to splitting the training set into train, dev splits.
TRAIN_RATIO = 0.8
CLASS_NAMES = ["melanoma", "seborrheic keratosis", "benign nevi"]


def melanoma_handler(artifacts_path: str) -> types.HandlerOutput:
  """Imports ISBI-ISIC 2017 melanoma classification challenge dataset.

  The dataset home page is at https://challenge.isic-archive.com/data/.

  Args:
    artifacts_path: Path with downloaded artifacts.

  Returns:
    Metadata and generator functions.
  """
  metadata = types.DatasetMetaData(
      num_classes=3,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata={
          "class_names": CLASS_NAMES,
          "train_size": 1600,
          "dev_size": 400,
          "train_and_dev_size": 2000,
          "dev-test_size": 150,
          "test_size": 600,
      })

  def gen(split):
    zip_path, label_path = _get_zip_and_label_file_paths(split)
    path_label_pairs = _create_path_label_pairs(
        artifacts_path, zip_path, label_path)

    # Shuffle for random splitting.
    if split in ["train", "dev", "train_and_dev"]:
      rng = np.random.default_rng(seed=1)
      path_label_pairs = rng.permutation(path_label_pairs)
      num_trains = int(len(path_label_pairs) * TRAIN_RATIO)

    # Split the training set into train and dev subsets.
    if split == "train":
      path_label_pairs = path_label_pairs[:num_trains]
    elif split == "dev":
      path_label_pairs = path_label_pairs[num_trains:]

    path_to_label = dict(path_label_pairs)

    return extraction_utils.generate_images_from_zip_files(
        artifacts_path, [zip_path],
        path_to_label_fn=lambda path: int(path_to_label[path]),
        path_filter=lambda path: path in path_to_label)

  return metadata, {
      "train": gen("train"),
      "train_and_dev": gen("train_and_dev"),
      "dev": gen("dev"),
      "dev-test": gen("dev-test"),
      "test": gen("test"),
  }


def _get_zip_and_label_file_paths(split: str):
  """Returns the zip and label file of a split."""
  if split in ["train", "dev", "train_and_dev"]:
    zip_path = TRAIN_ZIP_PATH
    label_path = TRAIN_LABEL_PATH
  elif split == "dev-test":
    zip_path = VALIDATION_ZIP_PATH
    label_path = VALIDATION_LABEL_PATH
  elif split == "test":
    zip_path = TEST_ZIP_PATH
    label_path = TEST_LABEL_PATH
  else:
    raise ValueError(f"Unsupported split name: {split}.")
  return zip_path, label_path


def _create_path_label_pairs(
    artifacts_path: str, zip_path: str, label_path: str):
  """Reads the label file and return a list of file path and label pairs."""
  zip_path_root = os.path.splitext(zip_path)[0]
  path_label_pairs = []
  with gfile.GFile(os.path.join(artifacts_path, label_path), "r") as f:
    # skip first line
    f.readline()
    for line in f:
      path_label_pairs.append(
          _get_image_path_and_label_for_path(line, zip_path_root))
  return path_label_pairs


def _get_image_path_and_label_for_path(line: str, zip_path_root: str):
  """Parses a line to get the image path and label."""
  # Each line is in the format of "image_id,melanoma,seborrheic_keratosis".
  parts = line.split(",")
  if len(parts) != 3:
    raise ValueError(f"Invalid format in line {line}.")
  image_id = parts[0]
  path = f"{zip_path_root}/{image_id}.jpg"
  melanoma = float(parts[1]) == 1
  seborrheic_keratosis = float(parts[2]) == 1
  if melanoma + seborrheic_keratosis > 1:
    raise ValueError(f"Line {line} contains multiple classes.")
  if melanoma:
    label = 0
  elif seborrheic_keratosis:
    label = 1
  else:
    label = 2
  return path, label


melanoma_dataset = types.DownloadableDataset(
    name="melanoma",
    download_urls=[
        types.DownloadableArtefact(
            url="https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip",
            checksum="a14a7e622c67a358797ae59abb8a0b0c"),
        types.DownloadableArtefact(
            url="https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part3_GroundTruth.csv",
            checksum="0cb4add57c65c22ca1a1cb469ad1f0c5"),
        types.DownloadableArtefact(
            url="https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Data.zip",
            checksum="8d6419d942112f709894c0d82f6c9038"),
        types.DownloadableArtefact(
            url="https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Part3_GroundTruth.csv",
            checksum="8d4826a76adcd8fb928ca52a23ebae4c"),
        types.DownloadableArtefact(
            url="https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Data.zip",
            checksum="5f6a0b5e1f2972bd1f5ea02680489f09"),
        types.DownloadableArtefact(
            url="https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part3_GroundTruth.csv",
            checksum="9e957b72a0c4f9e0d924889fd03b36ed")
    ],
    handler=melanoma_handler,
    website_url="https://challenge.isic-archive.com/data/",
    paper_url="https://arxiv.org/abs/1710.05006",
    paper_title=("Skin Lesion Analysis Toward Melanoma Detection: A Challenge "
                 "at the 2017 International Symposium on Biomedical Imaging "
                 "(ISBI), Hosted by the International Skin Imaging "
                 "Collaboration (ISIC)"),
    authors=("Noel C. F. Codella, David Gutman, M. Emre Celebi, Brian Helba, "
             "Michael A. Marchetti, Stephen W. Dusza, Aadi Kalloo, "
             "Konstantinos Liopyris, Nabin Mishra, Harald Kittler, "
             "Allan Halpern"),
    year=2017,
)
