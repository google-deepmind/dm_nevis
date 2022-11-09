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

"""LandSat UCI Repo dataset handler."""

import os
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
import numpy as np
from PIL import Image

from tensorflow.io import gfile

_TRAIN_FNAME = "sat.trn"
_TEST_FNAME = "sat.tst"


def landsat_handler(artifacts_path: str) -> types.HandlerOutput:
  """Imports LandSat dataset.

  The dataset home page is at
  https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite)

  The dataset consits of two ASCII files, one for training and one for testing.
  Each file is a table with as many rows as examples and 38 columns.
  The first column is the row number.
  The last column is the label.
  The remaining values are the values of a 3x3 patch of satellite image with 4
  channels.
  The task is classification:
  1 red soil
  2 cotton crop
  3 grey soil
  4 damp grey soil
  5 soil with vegetation stubble
  6 mixture class (all types present)
  7 very damp grey soil

  We are going to drop the last channel and provide as input images the 3x3 (3
  channels) patches. There are a total of 4435 patches in the training set.


  Args:
    artifacts_path: Path with downloaded artifacts.

  Returns:
    Metadata and generator functions.
  """
  files = gfile.listdir(artifacts_path)
  assert len(files) == 2
  label_to_class_index = {
      "red soil": 0,
      "cotton crop": 1,
      "grey soil": 2,
      "damp grey soil": 3,
      "soil with vegetation stubble": 4,
      "mixture class": 5,
      "very damp grey soil": 6}

  metadata = types.DatasetMetaData(
      num_classes=7,
      num_channels=3,
      image_shape=(3, 3),
      additional_metadata=dict(
          label_to_id=label_to_class_index,
          task_type="classification",
          image_type="object"))

  def gen(path):
    data = np.loadtxt(path)
    labels = data[:, -1]
    tot_num_samples = data.shape[0]
    side = 3
    num_channels = 4
    all_patches = data[:, :-1].reshape((tot_num_samples, side, side,
                                        num_channels))
    for cnt in range(tot_num_samples):
      ima = all_patches[cnt, :, :, :-1]
      label = int(labels[cnt] - 1)
      image = Image.fromarray(ima.astype("uint8"))

      yield (image, label)

  make_gen_fn = lambda: gen(os.path.join(artifacts_path, _TRAIN_FNAME))
  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN,
      splits.MERGED_TRAIN_AND_DEV)
  per_split_gen["test"] = gen(os.path.join(artifacts_path, _TEST_FNAME))

  return metadata, per_split_gen


landsat_dataset = types.DownloadableDataset(
    name="landsat",
    download_urls=[
        types.DownloadableArtefact(
            url="https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.trn",
            checksum="2c5ba2900da0183cab2c41fdb279fa5b"),
        types.DownloadableArtefact(
            url="https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.tst",
            checksum="02c995991fecc864e809b2c4c42cd983")
    ],
    handler=landsat_handler,
    paper_title="{UCI} Machine Learning Repository",
    authors="Dua, Dheeru and Graff, Casey",
    year="2019")
