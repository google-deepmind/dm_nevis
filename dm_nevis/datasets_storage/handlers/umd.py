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

"""UMD handler."""

import os

from dm_nevis.datasets_storage.handlers import extraction_utils as utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types

from tensorflow.io import gfile

_NUM_CLASSES = 25
_IGNORED_FILES_REGEX = r'.*\.db$'


def _label_from_filename(filename: str) -> int:
  """Extracts a label given a filename for the UMD dataset."""
  label = int(os.path.split(os.path.split(filename)[0])[1])
  label -= 1
  assert  0 <= label <= _NUM_CLASSES-1
  return label


def umd_handler(dataset_path: str) -> types.HandlerOutput:
  """Imports UMD Texture dataset.

  The dataset home page is at
  http://users.umiacs.umd.edu/~fer/website-texture/texture.htm.
  The dataset comes with two zip files containing 12 and 13 directories
  (one per class) for a total of 25 classes.
  We define the mapping from directory name to labels by subtracting one from it
  as the class directories start from 1 (and go to 25).
  The dataset does not come with a pre-defined train/ val/ test splits. We
  define those ourselves.

  Args:
    dataset_path: Path with downloaded artifacts.

  Returns:
    Metadata and generator functions.
  """
  ds_files = gfile.listdir(dataset_path)
  assert len(ds_files) == 2

  metadata = types.DatasetMetaData(
      num_classes=_NUM_CLASSES,
      num_channels=1,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          task_type='classification',
          image_type='texture'))

  def make_gen_fn():
    return utils.generate_images_from_zip_files(
        dataset_path,
        ds_files,
        path_to_label_fn=_label_from_filename,
        ignored_files_regex=_IGNORED_FILES_REGEX)

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)
  return metadata, per_split_gen


umd_dataset = types.DownloadableDataset(
    name='umd',
    download_urls=[
        types.DownloadableArtefact(
            url='http://users.umiacs.umd.edu/~fer/High-resolution-data-base/textures-1.zip',
            checksum='818b5b13035374cffd4db604e718ddbf'),
        types.DownloadableArtefact(
            url='http://users.umiacs.umd.edu/~fer/High-resolution-data-base/textures-2.zip',
            checksum='e9853d0f7eaa9e57c4756e9017d0cbc9')
    ],
    website_url='http://users.umiacs.umd.edu/~fer/website-texture/texture.htm',
    paper_title='A projective invariant for textures',
    authors='Yong Xu and Hui Ji and Cornelia Fermuller',
    year='2006',
    handler=umd_handler,
)
