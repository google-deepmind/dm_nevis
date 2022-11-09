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

"""Magellan Venus Volcanoes dataset handler."""

import os
import re
import tarfile
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
import numpy as np
from PIL import Image

from tensorflow.io import gfile

_IGNORED_FILES_REGEX = r".*\.spr$|Chips|FOA|GroundTruths|Programs|Tables|README"


def magellan_venus_volcanoes_handler(
    artifacts_path: str) -> types.HandlerOutput:
  """Imports Magellan Venus Volcanoes dataset (counting task).

  The dataset home page is at
  http://archive.ics.uci.edu/ml/datasets/volcanoes+on+venus+-+jartool+experiment

  The dataset contains 134 gray-scale images of size 1024x1024.
  The creators provide coordinates of the volcanoes together with a label that
  expresses the confidence of the presence of each volcanoe.
  Some stats: 16 images have no volcanoe, 53 have at least 1 volcanoe with the
  highest confidence level (definitely), 85 images have at least a volcanoe with
  intermediate confidence level (probably), 104 images have at least a volcanoe
  with low confidence level (possibly), and 118 images have at least a volcanoe
  with the lowest confidence level (only a pit is visible).
  Note the number of volcanoes per image vary between 0 and 62.
  For this task, we turn the problem into binary classification. We classify
  whether the image most definitely contains a volcanoe or not.

  The package is structured as follows:
  package/GroundTruths stores the label files. The label file of the first image
  is img1.lxyr and it lists in each row all the detected volcanoes in the
  format: <confidence_level> <other_info>
  We assign a label equal to 1 if there is at least one row starting with 1.
  The images are in the folder: package/Images/
  For instance, the first image is: img1.sdt
  It can be opened in python via:
  (np.fromfile(open('img134.sdt'), np.ubyte)).reshape((1024, 1024))
  with values in 0, 255.


  Args:
    artifacts_path: Path with downloaded artifacts.

  Returns:
    Metadata and generator functions.
  """
  files = gfile.listdir(artifacts_path)
  assert len(files) == 1
  label_to_class_index = {"neg": 0,  # There is no volcanoe.
                          "pos": 1}  # There is at least a volcanoe.
  all_labels = dict()
  # extract the label information
  with tarfile.open(os.path.join(artifacts_path, files[0]), "r:gz") as tar:
    for labelfile in tar.getmembers():
      if "GroundTruths" in labelfile.name and labelfile.name.endswith("lxyr"):
        f_obj = tar.extractfile(labelfile)
        assert f_obj

        def _get_label_from_lines(f_obj):
          for line in f_obj:
            label, *_ = line.decode("utf-8").split(" ")
            if label == "1":
              return 1
          return 0

        label = _get_label_from_lines(f_obj)
        fullname = labelfile.name.split("/")[-1]
        name, _ = os.path.splitext(fullname)
        all_labels[name + ".sdt"] = label  # image filename -> label

  metadata = types.DatasetMetaData(
      num_classes=2,
      num_channels=1,
      image_shape=(),
      additional_metadata=dict(
          label_to_id=label_to_class_index,
          task_type="classification",
          image_type="object"))

  def path_to_label_fn(path: str) -> types.Label:
    fields = path.split("/")
    assert len(fields) == 3
    return all_labels[fields[-1]]

  def make_gen_fn():
    with tarfile.open(os.path.join(artifacts_path, files[0]), "r:gz") as tf:
      for member in tf:
        if member.isdir() or re.search(_IGNORED_FILES_REGEX, member.name):
          continue
        label = path_to_label_fn(member.path)
        f_obj = tf.extractfile(member.name)
        assert f_obj
        np_image = (np.frombuffer(f_obj.read(), dtype="ubyte")
                    ).reshape((1024, 1024))
        image = Image.fromarray(np_image)
        yield (image, label)

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)

  return metadata, per_split_gen


magellan_venus_volcanoes_dataset = types.DownloadableDataset(
    name="magellan_venus_volcanoes",
    download_urls=[
        types.DownloadableArtefact(
            url="http://archive.ics.uci.edu/ml/machine-learning-databases/volcanoes-mld/volcanoes.tar.gz",
            checksum="55143a4ec42b626126c9b4ed618f59f8")
    ],
    handler=magellan_venus_volcanoes_handler,
    paper_title="Learning to Recognize Volcanoes on Venus",
    authors="M.C. Burl, L. Asker, P. Smyth, U. Fayyad, P. Perona, L. Crumpler and J. Aubele",
    year="1998")
