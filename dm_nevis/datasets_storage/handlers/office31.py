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

"""Office31 domain adaptation benchmark dataset.

Download seemed unreliable.
"""

import functools
import re

from dm_nevis.datasets_storage.handlers import extraction_utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types

LABELS_TO_ID = {
    "back_pack": 0,
    "bike": 1,
    "bike_helmet": 2,
    "bookcase": 3,
    "bottle": 4,
    "calculator": 5,
    "desk_chair": 6,
    "desk_lamp": 7,
    "desktop_computer": 8,
    "file_cabinet": 9,
    "headphones": 10,
    "keyboard": 11,
    "laptop_computer": 12,
    "letter_tray": 13,
    "mobile_phone": 14,
    "monitor": 15,
    "mouse": 16,
    "mug": 17,
    "paper_notebook": 18,
    "pen": 19,
    "phone": 20,
    "printer": 21,
    "projector": 22,
    "punchers": 23,
    "ring_binder": 24,
    "ruler": 25,
    "scissors": 26,
    "speaker": 27,
    "stapler": 28,
    "tape_dispenser": 29,
    "trash_can": 30,
}

_ARCHIVE_FILENAME = "domain_adaptation_images.tar.gz"

# Filnames in the archive look like this:
#  domain_adaptation_images/amazon/images/bike/frame_0001.jpg
#  domain_adaptation_images/dslr/images/bike/frame_0001.jpg
#  domain_adaptation_images/webcam/images/bike/frame_0003.jpg
#
#
# where 003 is the label-number 1..257 (one-based).
_FILE_PATH_REGEX = re.compile(
    r"(\w+)/images/(.+)/frame_(\d\d\d\d)\.jpg")

SPLIT_WITH_FRACTIONS_FOR_TEST_AND_DEV_TEST = {"test": 0.5, "dev-test": 0.5}


def office31_handler(download_path: str) -> types.HandlerOutput:
  """Imports images from Office 31 domain adaptation dataset.

  Args:
    download_path: Directory containing the downloaded raw data.

  Returns:
    HandlerOutput
  """
  def path_to_label_fn(fname, domain_ls):
    fname_match = _FILE_PATH_REGEX.match(fname)
    if not fname_match:
      return None
    else:
      domain_str = fname_match.group(1)
      label_str = fname_match.group(2)
      label_id = LABELS_TO_ID[label_str]

      if domain_str not in domain_ls:
        # Only import images with matching domain
        return None

      return label_id

  def gen_split(domain_ls):
    yield from extraction_utils.generate_images_from_tarfiles(
        _ARCHIVE_FILENAME,
        working_directory=download_path,
        path_to_label_fn=functools.partial(path_to_label_fn,
                                           domain_ls=domain_ls))

  metadata = types.DatasetMetaData(
      num_classes=len(LABELS_TO_ID),
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_str_to_id=LABELS_TO_ID,
          task_type="classification",
          image_type="object"))

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      functools.partial(gen_split, domain_ls=["amazon", "dlsr"]),
      splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN_AND_DEV_ONLY,
      splits.MERGED_TRAIN_AND_DEV)

  test_dev_test_gen = splits.random_split_generator_into_splits_with_fractions(
      functools.partial(gen_split, domain_ls=["webcam"]),
      SPLIT_WITH_FRACTIONS_FOR_TEST_AND_DEV_TEST)

  per_split_gen = per_split_gen | test_dev_test_gen

  return (metadata, per_split_gen)


office31_dataset = types.DownloadableDataset(
    name="office31",
    download_urls=[
        types.DownloadableArtefact(
            url="https://drive.google.com/u/0/uc?id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE&export=download&confirm=y",
            checksum="1b536d114869a5a8aa4580b89e9758fb")
    ],
    website_url="https://faculty.cc.gatech.edu/~judy/domainadapt/",
    paper_title="Adapting Visual Category Models to New Domains",
    authors="Kate Saenko, Brian Kulis, Mario Fritz & Trevor Darrell",
    papers_with_code_url="https://paperswithcode.com/dataset/office-31",
    handler=office31_handler)
