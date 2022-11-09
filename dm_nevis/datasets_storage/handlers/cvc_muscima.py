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

"""CVC-MUSCIMA handler."""

import io
import os
from typing import Dict
import zipfile
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
from PIL import Image

_LABELS = [
    'w-23', 'w-14', 'w-18', 'w-11', 'w-40', 'w-42', 'w-28', 'w-03', 'w-05',
    'w-34', 'w-36', 'w-37', 'w-12', 'w-31', 'w-06', 'w-22', 'w-25', 'w-21',
    'w-38', 'w-48', 'w-32', 'w-07', 'w-39', 'w-15', 'w-10', 'w-17', 'w-45',
    'w-50', 'w-02', 'w-08', 'w-01', 'w-20', 'w-35', 'w-29', 'w-46', 'w-47',
    'w-13', 'w-30', 'w-33', 'w-09', 'w-16', 'w-49', 'w-43', 'w-44', 'w-24',
    'w-19', 'w-04', 'w-26', 'w-41', 'w-27'
]


def _path_to_label_fn(path: str, label_to_id: Dict[str, int]) -> int:
  label = os.path.basename(os.path.dirname(path))
  return label_to_id[label]


_IMAGE_PREFIX = 'CVCMUSCIMA_WI/PNG_GT_Gray'
_PARTITIONS_FNAME = 'Partitions_Set.zip'
_IMAGES_FNAME = 'CVCMUSCIMA_WI.zip'
_TEST_SPLIT_FNAME = 'Partitions_Set/set_2_Independent/set_2_testing_01.txt'
_TRAIN_SPLIT_FNAME = 'Partitions_Set/set_2_Independent/set_2_training_01.txt'


def cvc_muscima_handler(dataset_path: str) -> types.HandlerOutput:
  """Handler for CVC-MUSCIMA dataset."""
  with zipfile.ZipFile(os.path.join(dataset_path, _PARTITIONS_FNAME),
                       'r') as zf:
    with zf.open(_TRAIN_SPLIT_FNAME) as f:
      train_fnames = {line.decode('utf-8').strip() for line in f}
    with zf.open(_TEST_SPLIT_FNAME) as f:
      test_fnames = {line.decode('utf-8').strip() for line in f}

  label_to_id = dict(
      ((label, label_id) for label_id, label in enumerate(_LABELS)))

  metadata = types.DatasetMetaData(
      num_classes=50,
      num_channels=1,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_to_id=label_to_id,
          task_type='classification',
          image_type='ocr',
      ))

  def make_gen(fnames, label_to_id):
    with zipfile.ZipFile(os.path.join(dataset_path, _IMAGES_FNAME), 'r') as zf:
      for member in zf.infolist():
        if member.is_dir():
          continue
        fname = member.filename
        image_fname = os.path.basename(fname)

        if not fname.startswith(_IMAGE_PREFIX):
          continue

        label = os.path.basename(os.path.dirname(fname))
        if os.path.join(label, image_fname) not in fnames:
          continue

        label = label_to_id[label]
        image = Image.open(io.BytesIO(zf.read(member)))
        image.load()

        yield (image, label)

  make_gen_fn = lambda: make_gen(train_fnames, label_to_id)
  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN,
      splits.MERGED_TRAIN_AND_DEV)
  per_split_gen['test'] = make_gen(test_fnames, label_to_id)

  return metadata, per_split_gen


cvc_muscima_dataset = types.DownloadableDataset(
    name='cvc_muscima',
    download_urls=[
        types.DownloadableArtefact(
            url='http://www.cvc.uab.es/cvcmuscima/CVCMUSCIMA_WI.zip',
            checksum='33d7464a3dc376a9456bbfe7aad8c18f'),
        types.DownloadableArtefact(
            url='http://www.cvc.uab.es/cvcmuscima/Partitions_Set.zip',
            checksum='dd22cff47fd50ca01ee077b757a978cd')
    ],
    website_url='http://www.cvc.uab.es/cvcmuscima/index_database.html',
    handler=cvc_muscima_handler)
