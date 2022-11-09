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

"""Utilities for dataset handlers."""

import hashlib
import io
import os
import re
import shutil
import tarfile
from typing import Callable, Iterator, List, Optional, Tuple
import zipfile
from dm_nevis.datasets_storage.handlers import types
import numpy as np
from PIL import Image


DEFAULT_IGNORED_FILES_REGEX = r'.*Thumbs.db$'

PathToLabelFn = Callable[[str], Optional[int]]
PathToAttributesFn = Callable[[str], Optional[List[int]]]
PathFilter = Callable[[str], bool]


class ImageDecodingError(ValueError):
  """Raised for files that fail to decode as images."""


def resize_to_max_size(image: Image.Image, max_size: int) -> Image.Image:
  """Resizes an image so that it is no larger than the given size."""

  height, width = image.height, image.width
  longest_side = max(height, width)
  if longest_side <= max_size:
    return image

  factor = max_size / longest_side
  new_height = int(factor * height)
  new_width = int(factor * width)

  return image.resize((new_width, new_height), resample=Image.BICUBIC)


def to_one_hot(label, num_classes):
  one_hot_label = np.zeros((num_classes,)).astype(int)
  np.put(one_hot_label, label, 1)
  return one_hot_label


def unpack_file(packed_file_name, ds_path):
  packed_file_path = os.path.join(ds_path, packed_file_name)
  shutil.unpack_archive(packed_file_path, extract_dir=ds_path)
  os.remove(packed_file_path)


def generate_images_from_tarfiles(
    *paths: str,
    path_to_label_fn: PathToLabelFn,
    working_directory: str = '',
    ignored_files_regex: str = DEFAULT_IGNORED_FILES_REGEX,
    tarfile_read_mode: str = 'r',
    convert_mode: Optional[str] = None,
) -> Iterator[Tuple[Image.Image, types.Label]]:
  """Generates (image, label) pairs from tar file.

  Args:
    *paths: The positional arguments are all treated as paths to tar files.
    path_to_label_fn: A callable returning an integer label given the path of
      the member being extracted. If the label is None, then the file will be
      ignored.
    working_directory: If provided, all paths will be opened relative to this
      path.
    ignored_files_regex: A regex used to ignore files that should not be
      extracted.
    tarfile_read_mode: A extension to use for reading tarfile.
    convert_mode: A mode to convert the image to (no conversion by default).

  Yields:
    (image, label) pairs consecutively from each of the input tar files.
  """
  for path in paths:
    with tarfile.open(os.path.join(working_directory, path),
                      tarfile_read_mode) as tf:
      for member in tf:
        if member.isdir() or re.search(ignored_files_regex, member.name):
          continue

        label = path_to_label_fn(member.path)
        if label is None:
          continue

        try:
          image = Image.open(tf.extractfile(member))
          if convert_mode:
            image = image.convert(convert_mode)
          image.load()
        except Exception as e:
          raise ImageDecodingError(
              f'Failed to decode as image: {member.path}') from e

        yield (image, label)


def generate_images_from_zip(
    zf: zipfile.ZipFile,
    path_to_label_fn: PathToLabelFn,
    ignored_files_regex: str = DEFAULT_IGNORED_FILES_REGEX,
    path_filter: Optional[PathFilter] = None,
    convert_mode: Optional[str] = None,
) -> Iterator[Tuple[Image.Image, types.Label]]:
  """Generates images and labels from z given zipfile.

  Args:
    zf: A zipfile in read mode.
    path_to_label_fn: A callable that maps a file name to the label to use for
      the associated image. The file will be igniored if it returns `None`.
    ignored_files_regex: Regular expression to ignore given files.
    path_filter: a callable to filter a path if it returns False.
    convert_mode: A mode to convert the image to (no conversion by default).

  Yields:
    An iterable over images and labels.
  """
  for name in sorted(zf.namelist()):
    f = zf.getinfo(name)

    if (f.is_dir() or re.search(ignored_files_regex, f.filename) or
        (path_filter and not path_filter(f.filename))):
      continue

    label = path_to_label_fn(f.filename)
    if label is None:
      continue

    try:
      image = Image.open(io.BytesIO(zf.read(f)))
      if convert_mode:
        image = image.convert(convert_mode)
      image.load()
    except Exception as e:
      raise ImageDecodingError(
          f'Failed to decode as image: {f.filename}') from e

    yield (image, label)


def generate_images_from_zip_files(
    dataset_path: str,
    zip_file_names: List[str],
    path_to_label_fn: PathToLabelFn,
    ignored_files_regex: str = DEFAULT_IGNORED_FILES_REGEX,
    path_filter: Optional[PathFilter] = None,
    convert_mode: Optional[str] = None,
):
  """Generates (image, label) pairs from zip file.

  Args:
    dataset_path: Base path prefixed to all filenames.
    zip_file_names: Names of zip files to open.
    path_to_label_fn: A callable that maps a file name to the label_id (int).
      The file will be ignored if it returns `None`.
    ignored_files_regex: Regular expression for files to ignore.
    path_filter: a callable to filter a path if it returns False.
    convert_mode: A mode to convert the image to (no conversion by default).

  Yields:
    (image, label) tuples.
  """
  for zip_fname in zip_file_names:
    with zipfile.ZipFile(os.path.join(dataset_path, zip_fname), 'r') as zf:
      yield from generate_images_from_zip(zf, path_to_label_fn,
                                          ignored_files_regex, path_filter,
                                          convert_mode)


def generate_images_from_zip_with_multilabels(
    zf: zipfile.ZipFile,
    path_to_attributes_fn: PathToAttributesFn,
    ignored_files_regex: str = DEFAULT_IGNORED_FILES_REGEX,
    path_filter: Optional[PathFilter] = None,
    convert_mode: Optional[str] = None,
):
  """Generates images and attributes from z given zipfile.

  Args:
    zf: A zipfile in read mode.
    path_to_attributes_fn: A callable that maps a file name to the attribute
      list to use for the associated image. The file will be ignored if it
      returns `None`.
    ignored_files_regex: Regular expression to ignore given files.
    path_filter: a callable to filter a path if it returns False.
    convert_mode: A mode to convert the image to (no conversion by default).

  Yields:
    An iterable over images and labels.
  """
  for name in sorted(zf.namelist()):
    f = zf.getinfo(name)

    if (f.is_dir() or re.search(ignored_files_regex, f.filename) or
        (path_filter and not path_filter(f.filename))):
      continue

    attributes = path_to_attributes_fn(f.filename)
    if attributes is None:
      continue

    try:
      image = Image.open(io.BytesIO(zf.read(f)))
      if convert_mode:
        image = image.convert(convert_mode)
      image.load()
    except Exception as e:
      raise ImageDecodingError(
          f'Failed to decode as image: {f.filename}') from e

    yield types.Example(image=image, multi_label=attributes, label=None)


def generate_images_from_zip_files_with_multilabels(
    dataset_path: str,
    zip_file_names: List[str],
    path_to_attributes_fn: PathToAttributesFn,
    ignored_files_regex: str = DEFAULT_IGNORED_FILES_REGEX,
    path_filter: Optional[PathFilter] = None,
    convert_mode: Optional[str] = None,
):
  """Generates (image, label) pairs from zip file.

  Args:
    dataset_path: Base path prefixed to all filenames.
    zip_file_names: Names of zip files to open.
    path_to_attributes_fn: A callable that maps a file name to a list of labels.
      The file will be ignored if it returns `None`.
    ignored_files_regex: Regular expression for files to ignore.
    path_filter: a callable to filter a path if it returns False.
    convert_mode: A mode to convert the image to (no conversion by default).

  Yields:
    (image, label) tuples.
  """
  for zip_fname in zip_file_names:
    with zipfile.ZipFile(os.path.join(dataset_path, zip_fname), 'r') as zf:
      yield from generate_images_from_zip_with_multilabels(
          zf, path_to_attributes_fn, ignored_files_regex, path_filter,
          convert_mode)


def deduplicate_data_generator(
    gen: types.DataGenerator) -> Callable[[], types.DataGenerator]:
  """Reads the data from generator and removes duplicates."""
  unique_set = set()
  unique_examples = []
  for example in gen:
    assert not isinstance(example, types.Example)
    assert len(example) == 2
    image, label = example[:2]
    img_hash = hashlib.md5(image.tobytes()).hexdigest()
    key = (img_hash, label)
    if key not in unique_set:
      unique_examples.append((image, label))
      unique_set.add(key)

  def make_gen_fn():
    for example in unique_examples:
      yield example

  return make_gen_fn
