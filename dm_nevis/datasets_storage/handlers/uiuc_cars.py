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

"""UIUC Cars dataset handler."""

import os
from typing import Dict

from dm_nevis.datasets_storage.handlers import extraction_utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types

from tensorflow.io import gfile

_IGNORED_FILES_REGEX = r".*\.txt$|TestImages_Scale"


def uiuc_cars_handler(artifacts_path: str) -> types.HandlerOutput:
  """Imports UIUC Cars dataset (classification task).

  The dataset home page is at
  http://host.robots.ox.ac.uk/pascal/VOC/databases.html#UIUC
  The original dataset is a car detection dataset. However, in the paper where
  this task was extracted (namely, "A Single Classifier for View-Invariant
  Multiple Object Class Recognition" available at
  http://www.macs.hw.ac.uk/bmvc2006/papers/081.pdf), they used the original
  training set (which has both positive and negative example) for
  classification. The authors split the original training set into training and
  test sets. Here we'll follow the same procedure but also generate a dev and
  dev-test split.

  Args:
    artifacts_path: Path with downloaded artifacts.

  Returns:
    Metadata and generator functions.
  """
  files = gfile.listdir(artifacts_path)
  assert len(files) == 1
  label_to_class_index = {"neg": 0,  # There is no car.
                          "pos": 1}  # There is a car.

  metadata = types.DatasetMetaData(
      num_classes=2,
      num_channels=1,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          class_names=["neg", "pos"],
          label_to_id=label_to_class_index,
          task_type="classification",
          image_type="object"))

  def path_to_label_fn(path: str,
                       label_to_id: Dict[str, types.Label]) -> types.Label:
    label = "pos" if "pos" in path else "neg"
    return label_to_id[label]

  def make_gen_fn():
    return extraction_utils.generate_images_from_tarfiles(
        os.path.join(artifacts_path, files[0]),
        path_to_label_fn=lambda pp: path_to_label_fn(pp, label_to_class_index),
        ignored_files_regex=_IGNORED_FILES_REGEX
    )

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)

  return metadata, per_split_gen


uiuc_cars_dataset = types.DownloadableDataset(
    name="uiuc_cars",
    download_urls=[
        types.DownloadableArtefact(
            url="http://host.robots.ox.ac.uk/pascal/VOC/download/uiuc.tar.gz",
            checksum="716c6078f57839bb440967fa74116da3")
    ],
    handler=uiuc_cars_handler,
    paper_title="Learning a sparse representation for object detection",
    authors="S. Agarwal and D. Roth",
    year="2002")
