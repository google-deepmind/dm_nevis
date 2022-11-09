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

"""CMU AMP expression dataset handler."""


from dm_nevis.datasets_storage.handlers import extraction_utils
from dm_nevis.datasets_storage.handlers import types
import numpy as np

_ZIP_PATH = "faceExpressionDatabase.zip"
_NUM_CLASSES = 13
_TOT_IMAS_PER_CLASS = 75
_IMAGE_SHAPE = (64, 64)
# Split all the images per class into train_all/test with 0.8:0.2. Then split
# the train_all subset into train/dev/dev-test with 0.7:0.15:0.15.
_PERC_TEST = 0.2
_PERC_DEV_IN_TRAIN = 0.15
_PERC_DEV_TEST_IN_TRAIN = 0.15

IGNORED_FILES_REGEX = r"readme\.txt"


def cmu_amp_expression_handler(artifacts_path: str) -> types.HandlerOutput:
  """Imports CMU AMP expression dataset.

  The dataset home page is at
  http://chenlab.ece.cornell.edu/projects/FaceAuthentication/Default.html.

  Each image file is named as: "*##.bmp", where * is the index of subject in
  letter starting from "A", and ## is the index of face image for this subject.
  The image file name is from "A00.bmp" to "M74.bmp".

  Args:
    artifacts_path: Path with downloaded artifacts.

  Returns:
    Metadata and generator functions.
  """
  # Ratios of the train, dev and dev-test splits.
  split_ratios = np.array([
      (1 - _PERC_TEST) * (1 - _PERC_DEV_IN_TRAIN - _PERC_DEV_IN_TRAIN),
      (1 - _PERC_TEST) * _PERC_DEV_IN_TRAIN,
      (1 - _PERC_TEST) * _PERC_DEV_TEST_IN_TRAIN])
  split_sizes = (_TOT_IMAS_PER_CLASS * split_ratios).astype(np.int32)
  acc_split_sizes = np.cumsum(split_sizes)

  # Set the seed for random splitting.
  rng = np.random.default_rng(seed=1)

  train_fnames = []
  dev_fnames = []
  dev_test_fnames = []
  test_fnames = []
  for label in range(_NUM_CLASSES):
    # Randomly splitting the images of every class into train, dev, dev-test,
    # and test subsets.
    inds = rng.permutation(_TOT_IMAS_PER_CLASS)
    train_inds, dev_inds, dev_test_inds, test_inds = np.split(inds,
                                                              acc_split_sizes)
    train_fnames.extend([_make_fname(label, ind) for ind in train_inds])
    dev_fnames.extend([_make_fname(label, id) for id in dev_inds])
    dev_test_fnames.extend([_make_fname(label, id) for id in dev_test_inds])
    test_fnames.extend([_make_fname(label, id) for id in test_inds])
  split_to_fnames = {
      "train": set(train_fnames),
      "dev": set(dev_fnames),
      "dev-test": set(dev_test_fnames),
      "test": set(test_fnames),
      "train_and_dev": set(train_fnames).union(set(dev_fnames)),
  }

  def gen(split):
    split_fnames = split_to_fnames[split]
    return extraction_utils.generate_images_from_zip_files(
        artifacts_path, [_ZIP_PATH],
        path_to_label_fn=_label_from_path,
        ignored_files_regex=IGNORED_FILES_REGEX,
        path_filter=lambda path: path in split_fnames)

  class_names = [_label_to_name(i) for i in range(_NUM_CLASSES)]
  metadata = types.DatasetMetaData(
      num_classes=_NUM_CLASSES,
      num_channels=1,
      image_shape=_IMAGE_SHAPE,  # Ignored for now.
      additional_metadata={
          "class_names": class_names,
          "split_to_fnames": split_to_fnames,
          "train_size_per_class": split_sizes[0],
          "dev_size_per_class": split_sizes[1],
          "dev_test_size_per_class": split_sizes[2],
          "test_size_per_class": _TOT_IMAS_PER_CLASS - acc_split_sizes[-1],
      })

  return metadata, {split: gen(split) for split in split_to_fnames}


def _label_to_name(label: int) -> str:
  return chr(ord("A") + label)


def _make_fname(label: int, ind: int) -> str:
  return f"{_label_to_name(label)}{ind:02d}.bmp"


def _label_from_path(path: str) -> types.Label:
  """Get 0-based label from the first character of a filename like *##.bmp."""
  return ord(path[0]) - ord("A")


cmu_amp_expression_dataset = types.DownloadableDataset(
    name="cmu_amp_expression",
    download_urls=[
        types.DownloadableArtefact(
            url="http://chenlab.ece.cornell.edu/_download/FaceAuthentication/faceExpressionDatabase.zip",
            checksum="f6c2fdf87c095c17e2527c36a0528966")
    ],
    handler=cmu_amp_expression_handler,
    website_url="http://chenlab.ece.cornell.edu/projects/FaceAuthentication/Default.html",
    paper_url="https://doi.org/10.1016/S0031-3203(02)00033-X",
    paper_title="Face Authentication for Multiple Subjects Using Eigenflow",
    authors="Liu, Xiaoming; Chen, Tsuhan; Kumar, B. V. K. Vijaya",
    year=2003,
)
