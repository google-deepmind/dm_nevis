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

"""German Traffic Sign Recognition dataset handler."""

import io
import os
import zipfile

from dm_nevis.datasets_storage.handlers import extraction_utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types

_TRAIN_ZIP_PATH = "GTSRB_Final_Training_Images.zip"
_TEST_ZIP_IMAGES_PATH = "GTSRB_Final_Test_Images.zip"
_TEST_ZIP_LABELS_PATH = "GTSRB_Final_Test_GT.zip"
_LABEL_FILE = "GT-final_test.csv"
_IGNORED_FILES_REGEX = r".*\.csv$|.*\.txt$"


def german_tsr_handler(artifacts_path: str) -> types.HandlerOutput:
  """Imports German Traffic Sign Recognition dataset.

  The dataset home page is at https://benchmark.ini.rub.de/gtsrb_dataset.html.
  The dataset comes with two zip files, one for the training and one for the
  testing files.
  The training directory has one folder per class.
  The testing directory has all images in one folder with a csv file specifying
  the labels of the test images.
  Images have different spatial resolution.
  There are 43 classes in total, and about 50,000 images.

  Args:
    artifacts_path: Path with downloaded artifacts.

  Returns:
    Metadata and generator functions.
  """
  label_to_id = {}
  num_classes = 43
  for cc in range(num_classes):
    label_to_id["%05d" % cc] = cc

  metadata = types.DatasetMetaData(
      num_classes=num_classes,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_to_id=label_to_id,
          task_type="classification",
          image_type="object"))

  def gen(path, label_from_path):
    return extraction_utils.generate_images_from_zip_files(
        artifacts_path, [path],
        label_from_path,
        ignored_files_regex=_IGNORED_FILES_REGEX,
        convert_mode="RGB")

  # Prepare the label mapping for the test images.
  path_to_label = dict()
  with zipfile.ZipFile(
      os.path.join(artifacts_path, _TEST_ZIP_LABELS_PATH), "r") as zf:
    with io.TextIOWrapper(zf.open(_LABEL_FILE, "r"), encoding="utf-8") as f:

      for ll in f:
        fields = ll.split(";")
        filename = fields[0]
        label = fields[-1].strip()
        path_to_label[filename] = label

  def _label_from_path_ts(path: str) -> types.Label:
    fields = path.split("/")
    return int(path_to_label[fields[-1].strip()])

  make_gen_fn = lambda: gen(_TRAIN_ZIP_PATH, _label_from_path_tr)
  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN,
      splits.MERGED_TRAIN_AND_DEV)
  per_split_gen["test"] = gen(_TEST_ZIP_IMAGES_PATH, _label_from_path_ts)

  return metadata, per_split_gen


def _label_from_path_tr(path: str) -> types.Label:
  return int(os.path.basename(os.path.dirname(path)))


german_tsr_dataset = types.DownloadableDataset(
    name="german_tsr",
    download_urls=[
        types.DownloadableArtefact(
            url="https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip",
            checksum="f33fd80ac59bff73c82d25ab499e03a3"),
        types.DownloadableArtefact(
            url="https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip",
            checksum="c7e4e6327067d32654124b0fe9e82185"),
        types.DownloadableArtefact(
            url="https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip",
            checksum="fe31e9c9270bbcd7b84b7f21a9d9d9e5")
    ],
    paper_title="The German Traffic Sign Recognition Benchmark: A multi-class classification competition",
    authors="Johannes Stallkamp and Marc Schlipsing and Jan Salmen and Christian Igel",
    year="2011",
    handler=german_tsr_handler)
