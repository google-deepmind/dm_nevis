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

r"""Script to download all the LSCL datasets.
"""

import os
import shutil
from typing import Sequence

from absl import app
from absl import flags
from absl import logging

from dm_nevis.datasets_storage import download_util as du
from dm_nevis.datasets_storage import handlers
from dm_nevis.datasets_storage import preprocessing
from dm_nevis.datasets_storage import version_control as vc
from dm_nevis.streams import lscl_stream
from tensorflow.io import gfile

# pylint:disable=line-too-long
_NUM_SHARDS = flags.DEFINE_integer(
    'num_shards', default=10, help='Number of shards to write.')
_SHUFFLE_BUFFER_SIZE = flags.DEFINE_integer(
    'shuffle_buffer_size', default=10_000, help='Shuffle buffer size.')
_DATASET = flags.DEFINE_string(
    'dataset',
    default=None,
    help='Dataset name. MANDATORY or provide stream name.')
_STREAM = flags.DEFINE_string(
    'stream',
    default=None,
    help='Stream name. MANDATORY or provide dataset name.')

_LOCAL_DOWNLOAD_DIR = flags.DEFINE_string(
    'local_download_dir',
    default='/tmp/datasets',
    help='Local directory used for downloading.')
_FORCE_DOWNLOADING = flags.DEFINE_boolean(
    'force_downloading',
    default=False,
    help='Whether to download dataset regardless if it is available.')
_FORCE_EXTRACTION = flags.DEFINE_boolean(
    'force_extraction',
    default=False,
    help='Whether to extract dataset regardless if it is already extracted.')
_WRITE_STABLE_VERSION = flags.DEFINE_boolean(
    'write_stable_version',
    default=False,
    help='Whether to write a stable version of the dataset or `tmp_version`.')
_TRY_DOWNLOAD_ARTIFACTS_FROM_URLS = flags.DEFINE_boolean(
    'try_download_artifacts_from_urls',
    default=True,
    help='Whether to try to download artifacts from URLs.')
_TMP_VERSION = flags.DEFINE_string(
    'tmp_version',
    default='temp',
    help='Name of temporary version of the dataset.')
_CLEAN_ARTIFACTS_ON_COMPLETION = flags.DEFINE_boolean(
    'clean_artifacts_on_completion',
    default=False,
    help='Whether to clean up downloaded artifacts on completion.')
_COMPUTE_STATISTICS = flags.DEFINE_boolean(
    'compute_statistics',
    default=False,
    help='Whether or not to compute statistics and add to metadata. '
    'This is not supported for multilabels datasets.')


def _gather_dataset_names(dataset_name: str, stream_name: str) -> list[str]:
  """Returns a list of dataset names for a given dataset or stream."""
  if dataset_name:
    return [dataset_name.lower()]

  stream = lscl_stream.LSCLStreamVariant[stream_name.upper()]
  dataset_names = []
  for dataset_pretty_names in lscl_stream.LSCL_STREAM_PER_YEAR[stream].values():
    for dataset_pretty_name in dataset_pretty_names:
      source = lscl_stream.DATASET_NAME_TO_SOURCE[dataset_pretty_name]
      if isinstance(source, lscl_stream.NevisSource):
        dataset_names.append(source.name)

  return dataset_names


def _download_and_prepare_dataset(
    dataset_name: str,
    artifacts_path: str,
    version: str,
    try_download_artifacts_from_urls: bool,
    force_extraction: bool,
    compute_statistics: bool,
    num_shards: int,
    shuffle_buffer_size: int) -> None:
  """Downloads dataset, extracts it, and creates shards."""
  if try_download_artifacts_from_urls:
    du.lazily_download_artifacts(dataset_name, artifacts_path)

  # Extracting part.

  output_path = vc.get_dataset_path(dataset_name, version)
  gfile.makedirs(output_path)

  logging.info('Extracting dataset %s from dataset artifacts to %s.',
               dataset_name, output_path)

  if du.try_read_status(
      output_path) == du.DatasetDownloadStatus.READY and not force_extraction:
    logging.info('Dataset `%s` already extracted (skipping).', dataset_name)
    return

  downloadable_dataset = handlers.get_dataset(dataset_name)
  du.extract_dataset_and_update_status(
      downloadable_dataset,
      artifacts_path,
      output_path,
      num_shards,
      shuffle_buffer_size,
      compute_statistics=compute_statistics,
      preprocess_image_fn=preprocessing.preprocess_image_fn,
      preprocess_metadata_fn=preprocessing.preprocess_metadata_fn)


def main(argv: Sequence[str]) -> None:
  del argv

  if not gfile.exists(vc.get_lscl_data_dir()):
    gfile.makedirs(vc.get_lscl_data_dir())

  num_shards = _NUM_SHARDS.value
  shuffle_buffer_size = _SHUFFLE_BUFFER_SIZE.value
  compute_statistics = _COMPUTE_STATISTICS.value
  force_downloading = _FORCE_DOWNLOADING.value
  force_extraction = _FORCE_EXTRACTION.value
  try_download_artifacts_from_urls = _TRY_DOWNLOAD_ARTIFACTS_FROM_URLS.value
  version = version = 'stable' if _WRITE_STABLE_VERSION.value else _TMP_VERSION.value
  dataset_name = _DATASET.value
  stream_name = _STREAM.value

  if dataset_name and stream_name:
    raise ValueError('Exactly one of --dataset or --stream should be set.')
  if not dataset_name and not stream_name:
    raise ValueError('Provide dataset name or stream name to download.')

  dataset_names = _gather_dataset_names(dataset_name, stream_name)
  logging.info('%d datasets will be downloaded and prepared.',
               len(dataset_names))

  # Downloading part.
  for i, dataset_name in enumerate(dataset_names, start=1):
    logging.info('%d/%d: downloading dataset `%s`', i, len(dataset_names),
                 dataset_name)
    du.check_dataset_supported_or_exit(dataset_name)

    artifacts_path = os.path.join(_LOCAL_DOWNLOAD_DIR.value, dataset_name)
    if force_downloading:
      du.clean_artifacts_path(artifacts_path)

    _download_and_prepare_dataset(
        dataset_name=dataset_name,
        artifacts_path=artifacts_path,
        version=version,
        try_download_artifacts_from_urls=try_download_artifacts_from_urls,
        compute_statistics=compute_statistics,
        force_extraction=force_extraction,
        num_shards=num_shards,
        shuffle_buffer_size=shuffle_buffer_size)

    if _CLEAN_ARTIFACTS_ON_COMPLETION.value:
      logging.info('Cleaning up artifacts directory at `%s`', artifacts_path)
      shutil.rmtree(artifacts_path)


if __name__ == '__main__':
  app.run(main)
