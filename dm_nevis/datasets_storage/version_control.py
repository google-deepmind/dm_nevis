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

"""Tools to control data versionning."""
# pylint: disable=line-too-long
import os
from dm_nevis.datasets_storage import paths


def get_lscl_data_dir() -> str:
  """Prefix directory where all the data lives."""
  return paths.LSCL_DATA_DIR


def get_dataset_path(dataset: str, version: str = paths.STABLE_DIR_NAME) -> str:
  """Provides directory for the dataset and given version.

  Currently, we assume that data lives in `LSCL_DATA_DIR` directory followed
  by the proposed version. It means that if the version is `stable`, the dataset
  will be living in `LSCL_DATA_DIR/stable/dataset` directory.

  Args:
    dataset: Name of the dataset
    version: Name of the version of the dataset.

  Returns:
    Path to the dataset location for the given version.
  """
  return os.path.join(get_lscl_data_dir(), version, dataset)
