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

"""Coil20 dataset handler."""

import functools
import re

from dm_nevis.datasets_storage.handlers import extraction_utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types


def coil20_handler(download_path: str, processed: bool) -> types.HandlerOutput:
  """Imports Coil20 dataset."""

  if processed:
    # First filnames "coil-20-unproc.zip" are:
    #   coil-20-proc/obj1__11.png
    #   coil-20-proc/obj1__12.png
    #   coil-20-proc/obj1__13.png
    #   [...]
    archive_fname = "coil-20-proc.zip"
    fname_re = re.compile(r"coil-20-proc/obj(\d+)__\d+\.png")
    num_classes = 20
  else:
    # First filnames "coil-20-unproc.zip" are:
    #   coil-20-unproc/obj1__11.png
    #   coil-20-unproc/obj1__12.png
    #   coil-20-unproc/obj1__13.png
    #   [...]
    archive_fname = "coil-20-unproc.zip"
    fname_re = re.compile(r"coil-20-unproc/obj(\d+)__\d+\.png")
    num_classes = 5

  def path_to_label_fn(fname):
    fname_match = fname_re.match(fname)
    if not fname_match:
      return None
    else:
      return int(fname_match.group(1)) - 1

  def gen_split():
    return extraction_utils.generate_images_from_zip_files(
        download_path, [archive_fname],
        path_to_label_fn=path_to_label_fn,
        convert_mode="RGB")

  metadata = types.DatasetMetaData(
      num_classes=num_classes,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          task_type="classification",
          image_type="object"))

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      gen_split, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)

  return (metadata, per_split_gen)


coil_20_dataset = types.DownloadableDataset(
    name="coil20",
    download_urls=[
        types.DownloadableArtefact(
            url="http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip",
            checksum="464dec76a6abfcd00e8de6cf1e7d0acc")
    ],
    website_url="https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php",
    paper_title="Columbia Object Image Library (COIL-20)",
    authors="S. A. Nene, S. K. Nayar and H. Murase",
    year=1996,
    # num_examples=1440,
    handler=functools.partial(coil20_handler, processed=True))

coil_20_unproc_dataset = types.DownloadableDataset(
    name="coil20_unproc",
    download_urls=[
        types.DownloadableArtefact(
            url="http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-unproc.zip",
            checksum="a04fd3c91db987e5e634e7e0945a430a")
    ],
    website_url=coil_20_dataset.website_url,
    paper_title=coil_20_dataset.paper_title,
    authors=coil_20_dataset.authors,
    year=coil_20_dataset.year,
    # num_examples=350,
    handler=functools.partial(coil20_handler, processed=False))
