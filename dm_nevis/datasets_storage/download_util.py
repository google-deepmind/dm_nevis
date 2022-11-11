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

"""Download utils."""

import codecs
import collections
import enum
import hashlib
import os
import re
import shutil
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, TypeVar
import urllib

from absl import logging
import dill
from dm_nevis.datasets_storage import encoding
from dm_nevis.datasets_storage import handlers
from dm_nevis.datasets_storage import paths
from dm_nevis.datasets_storage import statistics
from dm_nevis.datasets_storage.handlers import kaggle_utils
from dm_nevis.datasets_storage.handlers import types
import numpy as np
from PIL import Image as pil_image
import requests
from tensorflow.io import gfile
import tqdm

DEFAULT_CHUNK_SIZE = 1024
DEFAULT_MAX_QUEUE_SIZE = 100
DEFAULT_SHUFFLE_BUFFER_LOG_SECONDS = 10

PreprocessImageFn = Callable[[pil_image.Image, str, Optional[Any], int],
                             pil_image.Image]
PreprocessMetadataFn = Callable[[types.DatasetMetaData], types.DatasetMetaData]


class DatasetDownloadStatus(enum.Enum):
  NOT_READY = 'not_ready'
  READY = 'ready'


def maybe_unpack_archive(link, cur_link_file, ds_path):
  if link.endswith('.zip') or link.endswith('.tar'):
    shutil.unpack_archive(cur_link_file, extract_dir=ds_path)
    os.remove(cur_link_file)


def write_status(dataset_path, status):
  status_file_path = os.path.join(dataset_path, paths.STATUS_FNAME)
  with gfile.GFile(status_file_path, 'w') as f:
    f.write(status.value)


def try_read_status(dataset_path):
  status_file_path = os.path.join(dataset_path, paths.STATUS_FNAME)
  if not gfile.exists(status_file_path):
    return None
  with gfile.GFile(status_file_path, 'r') as f:
    status = DatasetDownloadStatus(f.readline())
  return status


def download_data_from_links(links, local_ds_path):
  """Donwload data artefacts from a list of urls or kaggle dataset/competition).
  """
  for link in links:
    if isinstance(link, types.DownloadableArtefact):
      artefact_path = lazily_download_file(link.url, local_ds_path)
    elif isinstance(link, types.KaggleDataset):
      artefact_path = download_data_from_kaggle(
          link.dataset_name,
          local_ds_path,
          kaggle_origin=kaggle_utils.KaggleOrigin.DATASETS)
    elif isinstance(link, types.KaggleCompetition):
      artefact_path = download_data_from_kaggle(
          link.competition_name,
          local_ds_path,
          kaggle_origin=kaggle_utils.KaggleOrigin.COMPETITIONS)
    else:
      raise TypeError(f'Unknown type link: {type(link)}')

    if link.checksum is not None:
      validate_artefact_checksum(artefact_path, link.checksum)


def download_data_from_kaggle(data_name, local_ds_path, kaggle_origin):
  """Downloads dataset from kaggle website."""
  artefact_path = os.path.join(local_ds_path, f'{data_name.split("/")[-1]}.zip')
  if gfile.exists(artefact_path):
    logging.info('Found existing file at `%s` (skipping download)',
                 artefact_path)
  else:
    kaggle_utils.download_kaggle_data(
        data_name, local_ds_path, kaggle_origin=kaggle_origin)

  return artefact_path


def validate_artefact_checksum(artefact_path: str, checksum: str) -> None:
  """Compares read checksum to expected checksum."""
  hash_fn = hashlib.md5()
  buffer_size = 128 * hash_fn.block_size

  logging.info('Validating checksum of artefact `%s`', artefact_path)
  with gfile.GFile(artefact_path, 'rb') as f:
    while True:
      data = f.read(buffer_size)
      if not data:
        break
      hash_fn.update(data)

  read_checksum = hash_fn.hexdigest()
  if read_checksum != checksum:
    raise ValueError(
        f'{artefact_path} checksum ({read_checksum}) != to expected checksum {checksum}). '
        'This error can be silenced by removing the MD5 hash in the download '
        'URL of the failing dataset.')


def lazily_download_file(url: str, output_dir: str) -> str:
  """Lazily fetch a file and write it into the output dir."""

  # try to infer filename from url, cannot work on google drive
  tentative_filename = _get_filename_from_url(url)
  if _is_among_allowed_extentions(tentative_filename):
    path = os.path.join(output_dir, tentative_filename)
    if gfile.exists(path):
      logging.info('Found existing file at `%s` (skipping download)', path)
      return path

  with requests.head(url) as r:
    # If an HTTP error occurred, it will be raised as an exception here.
    r.raise_for_status()
    filename = _get_filename_from_headers(r.headers) or _get_filename_from_url(
        r.url)
    path = os.path.join(output_dir, filename)
    if gfile.exists(path):
      logging.info('Found existing file at `%s` (skipping download)', path)
      return path

  partial_path = f'{path}.part'

  with requests.get(url, stream=True, allow_redirects=True) as r:
    r.raise_for_status()

    with gfile.GFile(partial_path, 'wb') as w:
      filesize = _content_length_from_headers(r.headers)

      logging.info('Downloading `%s` to `%s`', url, path)
      with tqdm.tqdm(
          unit='B',
          unit_scale=True,
          unit_divisor=1024,
          total=filesize,
          desc=filename) as progress:

        for chunk in r.iter_content(chunk_size=DEFAULT_CHUNK_SIZE):
          w.write(chunk)
          progress.update(n=len(chunk))

  # Atomically move to destination, to avoid caching a partial file on failure.
  os.rename(partial_path, path)
  return path


def _get_filename_from_url(url: str) -> str:
  o = urllib.parse.urlparse(url)
  return o.path.split('/')[-1]


def _is_among_allowed_extentions(filename: str) -> bool:
  ext = filename.split('.')[-1]
  return ext in ('png', 'jpg', 'jpeg', 'gif', 'zip', 'tar', 'gz', 'rar')


def _get_filename_from_headers(headers: Dict[str, Any]) -> Optional[str]:
  """Extracts the filename from HTTP headers, if present."""

  # TODO: This can be vastly more complicated, but this logic should
  # work for most cases we will encounter.

  content = headers.get('Content-Disposition')
  if not content:
    return None

  match = re.findall('filename="(.+)"', content)
  if match:
    return match[0]

  match = re.findall('filename=(.+)', content)
  if match:
    return match[0]

  return None


def _content_length_from_headers(headers: Dict[str, Any]) -> Optional[int]:
  if 'Content-Length' not in headers:
    return None

  return int(headers['Content-Length'])


T = TypeVar('T')


def shuffle_generator_with_buffer(gen: Iterable[T],
                                  buffer_size: int,
                                  seed: int = 0) -> Iterator[T]:
  """Shuffles `gen` using a shuffle buffer."""
  rng = np.random.default_rng(seed)
  buffer = []

  for example in gen:
    if len(buffer) < buffer_size:
      buffer.append(example)
      logging.log_every_n_seconds(logging.INFO,
                                  'Filling shuffle buffer: %d/%d...',
                                  DEFAULT_SHUFFLE_BUFFER_LOG_SECONDS,
                                  len(buffer), buffer_size)
      continue

    idx = rng.integers(0, buffer_size)
    result, buffer[idx] = buffer[idx], example
    yield result

  rng.shuffle(buffer)
  yield from buffer


def _optionally_convert_to_example(example):
  """Converts example tuple into types.Example."""
  if isinstance(example, types.Example):
    return example
  assert len(example) == 2
  image, label = example[:2]
  return types.Example(image=image, label=label, multi_label=None)


def _extract_dataset(
    downloable_dataset: handlers.types.DownloadableDataset,
    artifacts_path: str,
    output_path: str,
    num_shards: int,
    shuffle_buffer_size: int,
    compute_statistics: bool,
    preprocess_image_fn: PreprocessImageFn,
    preprocess_metadata_fn: PreprocessMetadataFn,
    seed: int = 0,
) -> None:
  """Converts dataset to tfrecord examples."""

  logging.info('Extracting dataset %s from dataset artifacts.',
               downloable_dataset.name)

  metadata, splits_generators = downloable_dataset.handler(artifacts_path)
  metadata = preprocess_metadata_fn(metadata)
  num_channels = metadata.num_channels
  num_classes = metadata.num_classes
  # This is required for backwards compatibility. Later on, we will remove need
  # for default features.
  features = metadata.features or encoding.get_default_features(
      num_channels, num_classes)

  if compute_statistics:
    statistics_calculator = statistics.StatisticsCalculator(
        list(splits_generators.keys()), metadata)

  num_data_points_per_split = collections.defaultdict(int)
  for split, split_gen in splits_generators.items():
    logging.info('Writing split: `%s`', split)
    description = f'Writing examples for `{split}`'
    gen = shuffle_generator_with_buffer(split_gen, shuffle_buffer_size)
    rng = np.random.default_rng(seed=seed)

    split_dir = os.path.join(output_path, split)
    if not gfile.exists(split_dir):
      gfile.makedirs(split_dir)

    with encoding.ThreadpoolExampleWriter(
        num_shards,
        output_dir=split_dir,
        example_encoder=encoding.build_example_encoder(features)) as writer:

      with tqdm.tqdm(total=None, desc=description, unit=' examples') as pbar:
        for i, example in enumerate(gen):
          converted_example = _optionally_convert_to_example(example)
          converted_example._replace(
              image=preprocess_image_fn(converted_example.image,
                                        metadata.preprocessing, rng, 0))
          if compute_statistics:
            statistics_calculator.accumulate(converted_example.image,
                                             converted_example.label, split)

          shard_index = i % num_shards

          writer.write(shard_index, converted_example)

          num_data_points_per_split[split] += 1
          pbar.update(1)

  additional_metadata = metadata.additional_metadata

  additional_metadata['num_data_points_per_split'] = num_data_points_per_split
  additional_metadata['splits'] = sorted(splits_generators.keys())
  if compute_statistics:
    additional_metadata['statistics'] = statistics_calculator.merge_statistics()

  with gfile.GFile(os.path.join(output_path, paths.METADATA_FNAME), 'wb') as f:
    # See https://stackoverflow.com/a/67171328
    pickled_object = codecs.encode(
        dill.dumps(metadata, protocol=dill.HIGHEST_PROTOCOL),
        'base64').decode()
    f.write(pickled_object)


def extract_dataset_and_update_status(
    downloadable_dataset: handlers.types.DownloadableDataset,
    artifacts_path: str, output_path: str, num_shards: int,
    shuffle_buffer_size: int, compute_statistics: bool,
    preprocess_image_fn: PreprocessImageFn,
    preprocess_metadata_fn: PreprocessMetadataFn) -> None:
  """Extracts dataset from artifacts_path and updates its status.

  Args:
    downloadable_dataset: The instance of `types.DownloadableDataset` describing
      the dataset together with all the meta information.
    artifacts_path: The directory where fetched artifacts are stored for this
      dataset. These contain e.g. raw zip files and data downloaded from the
      internet. If artifacts are found here matching the paths, they will not be
      re-downloaded.
    output_path: The destination where the final dataset will be written.
    num_shards: The number of shards to write the dataset into.
    shuffle_buffer_size: The size of the shuffle buffer.
    compute_statistics: True if statistics should be computed.
    preprocess_image_fn: Preprocessing function converting PIL.Image into numpy
      array with consistent format.
    preprocess_metadata_fn: Preprocessing function responsible for applying
      consistent transformations to metadata.
  """
  logging.info('Marking dataset `%s` as NOT_READY', downloadable_dataset.name)
  write_status(output_path, DatasetDownloadStatus.NOT_READY)

  _extract_dataset(
      downloadable_dataset,
      artifacts_path,
      output_path,
      num_shards,
      shuffle_buffer_size,
      compute_statistics=compute_statistics,
      preprocess_image_fn=preprocess_image_fn,
      preprocess_metadata_fn=preprocess_metadata_fn)

  logging.info('Marking dataset `%s` as READY', downloadable_dataset.name)
  write_status(output_path, DatasetDownloadStatus.READY)


def lazily_download_artifacts(dataset_name: str, artifacts_path: str) -> None:
  """Downloads artifacts (lazily) to the given path.

  Args:
    dataset_name: The name of the dataset to fetch the artifacts for.
    artifacts_path: The destintion to write the fetched artifacts to.
  """
  if not gfile.exists(artifacts_path):
    gfile.makedirs(artifacts_path)

  downloadable_dataset = handlers.get_dataset(dataset_name)
  if downloadable_dataset.manual_download:
    logging.info('Dataset `%s` has to be manually downloaded (skipping).',
                 dataset_name)
    return

  uris = downloadable_dataset.download_urls
  logging.info('Fetching %d artifact(s) for %s to `%s`', len(uris),
               dataset_name, artifacts_path)
  download_data_from_links(uris, artifacts_path)


def check_dataset_supported_or_exit(dataset_name: str) -> None:
  """Checks whether given dataset has corresponding handler."""
  if not handlers.is_dataset_available(dataset_name):
    logging.error(
        'The dataset `%s` is not recognized. The available datasets are %s',
        dataset_name, ', '.join(handlers.dataset_names()))
    exit(1)


def clean_artifacts_path(artifacts_path: str) -> None:
  """Cleans up `artifacts_path` directory."""
  logging.info('Ensuring artifacts directory is empty at `%s`', artifacts_path)
  if gfile.exists(artifacts_path):
    shutil.rmtree(artifacts_path)
