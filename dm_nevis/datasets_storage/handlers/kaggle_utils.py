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

"""Data downloads using the Kaggle CLI.

Forked from tensorflow_datasets.core.download.kaggle.py. We fork to remove the
automatic zip extraction which deleted the archive.zip file. We want to do it
ourselves, so we can check the checksum of the zip file beforehand.
We also customize the dataset dir location where the file is downloaded.
"""
# pylint: disable=raise-missing-from
import enum
import subprocess
import textwrap
from typing import List

from absl import logging


class KaggleOrigin(enum.Enum):
  DATASETS = 'datasets'
  COMPETITIONS = 'competitions'


def _run_command(command_args: List[str]) -> str:
  """Runs kaggle command with subprocess.

  It contains the full command a user would use when working with the CLI,
  as described here: https://www.kaggle.com/docs/api.

  Args:
    command_args: Arguments to the kaggle api.

  Returns:
    output of the command.

  Raises:
    CalledProcessError: If the command terminates with exit status 1.
  """
  command_str = ' '.join(command_args)
  competition_or_dataset = command_args[-1]
  try:
    return subprocess.check_output(command_args, encoding='UTF-8')
  except (subprocess.CalledProcessError, FileNotFoundError) as err:
    if isinstance(err, subprocess.CalledProcessError) and '404' in err.output:
      raise ValueError(
          textwrap.dedent("""\
      Error for command: {}

      Competition {} not found. Please ensure you have spelled the name
      correctly.
      """).format(command_str, competition_or_dataset))
    else:
      raise RuntimeError(
          textwrap.dedent("""\
      Error for command: {}

      To download Kaggle data through TFDS, follow the instructions to install
      the kaggle API and get API credentials:
      https://github.com/Kaggle/kaggle-api#installation

      Additionally, you may have to join the competition through the Kaggle
      website: https://www.kaggle.com/c/{}
      """).format(command_str, competition_or_dataset))


def _download_competition_or_dataset(competition_or_dataset: str,
                                     output_dir: str,
                                     kaggle_origin: KaggleOrigin) -> None:
  """Downloads the data to `archive.zip`."""
  _run_command([
      'kaggle',
      kaggle_origin.value,
      'download',
      '--path',
      output_dir,
      competition_or_dataset,
  ])


def download_kaggle_data(competition_or_dataset: str, download_dir: str,
                         kaggle_origin: KaggleOrigin) -> None:
  """Downloads the kaggle data to the output_dir as a zip file.

  Args:
    competition_or_dataset: Name of the kaggle competition/dataset.
    download_dir: Path to the downloads dir.
    kaggle_origin: Precise whether it is a dataset or a competition
  """
  logging.info('Downloading %s into %s...', competition_or_dataset,
               download_dir)
  _download_competition_or_dataset(competition_or_dataset, download_dir,
                                   kaggle_origin)
