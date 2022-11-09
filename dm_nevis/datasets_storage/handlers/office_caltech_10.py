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

"""Office-Caltech-10 dataset handler.

Download seemed unreliable.
"""

import re

from dm_nevis.datasets_storage.handlers import extraction_utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types

CALTECH_LABELS_TO_ID = {
    "backpack": 0,
    "touring-bike": 1,
    "calculator": 2,
    "head-phones": 3,
    "computer-keyboard": 4,
    "laptop-101": 5,
    "computer-monitor": 6,
    "computer-mouse": 7,
    "coffee-mug": 8,
    "video-projector": 9,
}

_OBJECT_CATEGORIES_PATH = "256_ObjectCategories.tar"
# Filnames in the archive look like this:
#  256_ObjectCategories/003.backpack/003_0001.jpg
#
# where 003 is the label-number 1..257 (one-based).
_CALTECH_FILE_PATH_REGEX = re.compile(
    r"256_ObjectCategories/(\d\d\d)\.(.+)/(\d\d\d)_(\d\d\d\d)\.jpg")

OFFICE_LABELS_TO_ID = {
    "back_pack": 0,
    "bike": 1,
    "calculator": 2,
    "headphones": 3,
    "keyboard": 4,
    "laptop_computer": 5,
    "monitor": 6,
    "mouse": 7,
    "mug": 8,
    "projector": 9
}

_ARCHIVE_FILENAME = "domain_adaptation_images.tar.gz"

# Filnames in the archive look like this:
#  domain_adaptation_images/amazon/images/bike/frame_0001.jpg
#  domain_adaptation_images/dslr/images/bike/frame_0001.jpg
#  domain_adaptation_images/webcam/images/bike/frame_0003.jpg
#
#
# where 003 is the label-number 1..257 (one-based).
_OFFICE_FILE_PATH_REGEX = re.compile(
    r"(\w+)/images/(.+)/frame_(\d\d\d\d)\.jpg")

SPLIT_WITH_FRACTIONS_FOR_TEST_AND_DEV_TEST = {"test": 0.5, "dev-test": 0.5}


def office_caltech_10_handler(download_path: str) -> types.HandlerOutput:
  """Imports images from Office31 and Caltech256 and select the overlap classes.

  Args:
    download_path: Directory containing the downloaded raw data.

  Returns:
    HandlerOutput
  """
  def office_path_to_label_fn(fname):
    fname_match = _OFFICE_FILE_PATH_REGEX.match(fname)
    if not fname_match:
      return None
    else:
      label_str = fname_match.group(2)
      if label_str not in OFFICE_LABELS_TO_ID:
        return None
      label_id = OFFICE_LABELS_TO_ID[label_str]

      return label_id

  def caltech_path_to_label_fn(fname):
    fname_match = _CALTECH_FILE_PATH_REGEX.match(fname)
    if not fname_match:
      return None
    else:
      label_str = fname_match.group(2)
      if label_str not in CALTECH_LABELS_TO_ID:
        return None
      label_id = CALTECH_LABELS_TO_ID[label_str]
      return label_id

  def office_gen_split():
    yield from extraction_utils.generate_images_from_tarfiles(
        _ARCHIVE_FILENAME,
        working_directory=download_path,
        path_to_label_fn=office_path_to_label_fn)

  def caltech_gen_split():
    yield from extraction_utils.generate_images_from_tarfiles(
        _OBJECT_CATEGORIES_PATH,
        working_directory=download_path,
        path_to_label_fn=caltech_path_to_label_fn,
        convert_mode="RGB")

  metadata = types.DatasetMetaData(
      num_classes=len(CALTECH_LABELS_TO_ID),
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_str_to_id=CALTECH_LABELS_TO_ID,
          task_type="classification",
          image_type="object"))

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      office_gen_split,
      splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN_AND_DEV_ONLY,
      splits.MERGED_TRAIN_AND_DEV)

  test_dev_test_gen = splits.random_split_generator_into_splits_with_fractions(
      caltech_gen_split,
      SPLIT_WITH_FRACTIONS_FOR_TEST_AND_DEV_TEST)

  per_split_gen = per_split_gen | test_dev_test_gen

  return (metadata, per_split_gen)


office_caltech_10_dataset = types.DownloadableDataset(
    name="office_caltech_10",
    download_urls=[
        types.DownloadableArtefact(
            url="https://drive.google.com/u/0/uc?id=1r6o0pSROcV1_VwT4oSjA2FBUSCWGuxLK&export=download&confirm=y",
            checksum="67b4f42ca05d46448c6bb8ecd2220f6d"),
        types.DownloadableArtefact(
            url="https://drive.google.com/u/0/uc?id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE&export=download&confirm=y",
            checksum="1b536d114869a5a8aa4580b89e9758fb")
    ],
    website_url=[
        "http://www.vision.caltech.edu/Image_Datasets/Caltech256/",
        "https://faculty.cc.gatech.edu/~judy/domainadapt/"
    ],
    paper_title="Geodesic Flow Kernel for Unsupervised Domain Adaptation",
    authors="B Gong, Y Shi, F Sha, and K Grauman",
    papers_with_code_url="https://paperswithcode.com/dataset/office-caltech-10",
    handler=office_caltech_10_handler)
