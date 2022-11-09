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

"""Interact dataset handler.

We use a similar experiment setting as in Sharmanska and Quadrianto (2016) where
we include the category level illustrations in the training set together with
a randomly selected subset of real images, and test on hold-out real images.

Reference:
Sharmanska, Viktoriia, and Quadrianto, Novi. "Learning from the mistakes of
others: Matching errors in cross-dataset learning." In Proceedings of the
IEEE Conference on Computer Vision and Pattern Recognition, pp. 3967-3975. 2016.
"""

import collections
import functools
import os
from typing import Dict, List, Sequence, Tuple
import zipfile
from dm_nevis.datasets_storage.handlers import extraction_utils
from dm_nevis.datasets_storage.handlers import types
import numpy as np

_STAND_ALONE_DATASET_IMAGES_ZIP_PATH = "interact_stand-alone_dataset_images.zip"
_STAND_ALONE_DATASET_ANNOTATION_ZIP_PATH = "interact_stand-alone_dataset_annotations.zip"
_STAND_ALONE_DATASET_ANNOTATION_FILENAME = "interact_final_dataset_labels_and_images.txt"
_ILLUSTRATION_CATEGORY_DATASET_IMAGES_ZIP_PATH = "interact_illustration_category_dataset_images.zip"
_ILLUSTRATION_CATEGORY_DATASET_ANNOTATION_ZIP_PATH = "interact_illustration_category_dataset_annotations.zip"
_ILLUSTRATION_CATEGORY_DATASET_ANNOTATION_FILENAME = "interact_illustration_category_dataset_labels_and_images.txt"
_NUM_CLASSES = 60
_PERC_TEST = 0.2
_PERC_DEV_IN_TRAIN = 0.15
_PERC_DEV_TEST_IN_TRAIN = 0.15

# The following image files raise exceptions about truncation in image.resize.
_TRUNCATED_IMAGES_TO_EXCLUDE = [
    "imgs/22TCSTMOFXRDFDVIK9PDTXJK5OZTQ6_20.jpg",
    "imgs/249YW2GTP9RAU1I25WX94SOKQN6OTI_16.jpg",
    "imgs/24PFCD45XCETAM5TCF9N9TGUNNTZXA_02.jpg"
]


def _split_paths(paths: Sequence[str], rng: np.random.Generator
                 ) -> Dict[str, List[str]]:
  """Randomly splits file paths to train, dev, dev-test, and test subsets."""

  # Ratios of the train, dev and dev-test splits.
  split_ratios = np.array([
      (1 - _PERC_TEST) * (1 - _PERC_DEV_IN_TRAIN - _PERC_DEV_IN_TRAIN),
      (1 - _PERC_TEST) * _PERC_DEV_IN_TRAIN,
      (1 - _PERC_TEST) * _PERC_DEV_TEST_IN_TRAIN])

  total_size = len(paths)
  split_sizes = (total_size * split_ratios).astype(np.int32)
  acc_split_sizes = np.cumsum(split_sizes)

  # Randomly splitting the images of every class into .
  inds = rng.permutation(total_size)
  train_inds, dev_inds, dev_test_inds, test_inds = np.split(inds,
                                                            acc_split_sizes)

  return {
      "train": [paths[i] for i in train_inds],
      "dev": [paths[i] for i in dev_inds],
      "dev-test": [paths[i] for i in dev_test_inds],
      "test": [paths[i] for i in test_inds],
      "train_and_dev": [
          paths[i] for i in np.concatenate([train_inds, dev_inds])
      ],
  }


def interact_handler(artifacts_path: str) -> types.HandlerOutput:
  """Imports interact dataset.

  Each image file is named as: "*##.bmp", where * is the index of subject in
  letter starting from "A", and ## is the index of face image for this subject.
  The image file name is from "A00.bmp" to "M74.bmp".

  Args:
    artifacts_path: Path with downloaded artifacts.

  Returns:
    Metadata and generator functions.
  """

  label_to_paths = collections.defaultdict(list)
  path_to_label = {}
  label_to_name = {}

  # Load stand-alone dataset annotations.
  label_to_paths, path_to_label, label_to_name = _load_annotations(
      artifacts_path, _STAND_ALONE_DATASET_ANNOTATION_ZIP_PATH,
      _STAND_ALONE_DATASET_ANNOTATION_FILENAME)

  labels = sorted(label_to_name.keys())
  if min(labels) != 0 or max(labels) != len(labels) - 1:
    raise ValueError("Class label does not cover a continguous range from 0.")
  if len(labels) != _NUM_CLASSES:
    raise ValueError(f"Number of classes ({len(labels)}) does not match the "
                     f"expected value ({_NUM_CLASSES}).")

  split_to_paths = _split_images_per_class(label_to_paths)

  # Load illustration dataset annotations.
  (illustration_label_to_paths, illustration_path_to_label,
   illustration_label_to_name) = _load_annotations(
       artifacts_path, _ILLUSTRATION_CATEGORY_DATASET_ANNOTATION_ZIP_PATH,
       _ILLUSTRATION_CATEGORY_DATASET_ANNOTATION_FILENAME)

  # Verify the illustration_label_to_name dict matches label_to_name.
  for label, class_name in illustration_label_to_name.items():
    if label not in label_to_name or class_name != label_to_name[label]:
      raise ValueError("Found mismatched category label and name pair.")

  # Add all illustration images to the training set.
  path_to_label.update(illustration_path_to_label)
  for paths in illustration_label_to_paths.values():
    split_to_paths["train"].extend(paths)

  # Convert lists to sets for faster indexing.
  split_to_paths = {split: set(paths)
                    for split, paths in split_to_paths.items()}

  def gen(split):
    split_paths = split_to_paths[split]
    return extraction_utils.generate_images_from_zip_files(
        artifacts_path, [
            _STAND_ALONE_DATASET_IMAGES_ZIP_PATH,
            _ILLUSTRATION_CATEGORY_DATASET_IMAGES_ZIP_PATH
        ],
        path_to_label_fn=lambda path: path_to_label[path],
        path_filter=lambda path: path in split_paths)

  class_names = [label_to_name[i] for i in range(_NUM_CLASSES)]
  metadata = types.DatasetMetaData(
      num_classes=_NUM_CLASSES,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata={
          "class_names": class_names,
          "split_to_paths": split_to_paths,
      })

  return metadata, {split: gen(split) for split in split_to_paths}


def _load_annotations(
    artifacts_path: str, zip_path: str, anno_filename: str
) -> Tuple[Dict[types.Label, List[str]], Dict[str, types.Label], Dict[
    types.Label, str]]:
  """Load annotations from one source."""
  label_to_paths = collections.defaultdict(list)
  path_to_label = {}
  label_to_name = {}
  with zipfile.ZipFile(os.path.join(artifacts_path, zip_path), "r") as zf:
    with zf.open(anno_filename, "r") as f:
      for line in f:
        class_label, class_name, image_path = _parse_line(line)
        if image_path in _TRUNCATED_IMAGES_TO_EXCLUDE:
          continue
        label_to_paths[class_label].append(image_path)
        path_to_label[image_path] = class_label

        # Update the label_to_name dict.
        if class_label not in label_to_name:
          label_to_name[class_label] = class_name
        elif label_to_name[class_label] != class_name:
          raise ValueError("Multiple class names: "
                           f"{label_to_name[class_label]}, {class_name}, "
                           f"correspond to label {class_label}")
  return label_to_paths, path_to_label, label_to_name


def _parse_line(line: bytes) -> Tuple[types.Label, str, str]:
  """Parses one line to image class label, class name and path."""
  parts = line.decode().strip().split(";")
  # Every line is in the following format:
  # category_number_label;image_filename;category_semantic_label
  class_label_str, fname, class_name = parts
  class_label = int(class_label_str) - 1  # 0-based label.
  image_path = f"imgs/{fname}"
  return class_label, class_name, image_path


def _split_images_per_class(label_to_paths: Dict[types.Label, Sequence[str]]
                            ) -> Dict[str, List[str]]:
  """Splits images into subsets per class and merge them."""
  # Set the seed for random splitting.
  rng = np.random.default_rng(seed=1)
  split_to_paths = collections.defaultdict(list)
  for paths in label_to_paths.values():
    split_to_paths_per_class = _split_paths(paths, rng)
    for split, paths_subset in split_to_paths_per_class.items():
      split_to_paths[split].extend(paths_subset)
  return split_to_paths


interact_dataset = types.DownloadableDataset(
    name="interact",
    download_urls=[
        types.DownloadableArtefact(
            url="https://computing.ece.vt.edu/~santol/projects/zsl_via_visual_abstraction/interact/interact_stand-alone_dataset/interact_stand-alone_dataset_images.zip",
            checksum="93d7a853e0e596e7ef7fecc4808ffdf8"),
        types.DownloadableArtefact(
            url="https://computing.ece.vt.edu/~santol/projects/zsl_via_visual_abstraction/interact/interact_stand-alone_dataset/interact_stand-alone_dataset_annotations.zip",
            checksum="aef1de2ad7c596a566cd78414a266f5f"),
        types.DownloadableArtefact(
            url="https://computing.ece.vt.edu/~santol/projects/zsl_via_visual_abstraction/interact/interact_illustration_category_dataset/interact_illustration_category_dataset_images.zip",
            checksum="47c043e360a177c026fa7b7652fdf5b2"),
        types.DownloadableArtefact(
            url="https://computing.ece.vt.edu/~santol/projects/zsl_via_visual_abstraction/interact/interact_illustration_category_dataset/interact_illustration_category_dataset_annotations.zip",
            checksum="56dec629f180077eb2a76b3f67c7b02f")
    ],
    handler=functools.partial(interact_handler),
    website_url="https://computing.ece.vt.edu/~santol/projects/zsl_via_visual_abstraction/interact/index.html",
    paper_url="https://openreview.net/forum?id=rkbYYcWOZB",
    paper_title="Zero-Shot Learning via Visual Abstraction",
    authors="Antol, Stanislaw and Zitnick, C. Lawrence and Parikh, Devi",
    year=2014)
