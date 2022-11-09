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

"""Butterflies dataset handler."""

import io
import os
import re
import zipfile
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
from PIL import Image

# pylint: disable=line-too-long


def butterflies_handler(dataset_path: str) -> types.HandlerOutput:
  """Imports Butterflies dataset.

  Semi-Local Affine Parts for Object Recognition
  S. Lazebnik, C. Schmid, J. Ponce
  Published in BMVC 7 September 2004

  Link:
  https://web.archive.org/web/20070829035029/http://www-cvr.ai.uiuc.edu/ponce_grp/data/index.html

  Args:
    dataset_path: Path with downloaded datafiles.

  Returns:
    Metadata and generator functions.
  """
  datafile = os.path.join(dataset_path, 'butterflies.zip')
  with zipfile.ZipFile(os.path.join(dataset_path, datafile), 'r') as zf:
    labels = set([os.path.split(member)[0] for member in zf.namelist()])

  num_classes = len(labels)
  label_str_to_id = {l: i for i, l in enumerate(labels)}

  metadata = types.DatasetMetaData(
      num_classes=num_classes,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(label_str_to_id=label_str_to_id))

  # The test-split members are cut'n'pasted from
  # http://www-cvr.ai.uiuc.edu/ponce_grp/data/butterflies/butterflies_f_order.txt
  test_images = {
      'admiral': {14, 103, 41, 86, 9, 53, 6, 55, 66, 92},
      'black_swallowtail': {25, 41, 2, 22, 31, 15, 28, 35, 4, 27},
      'machaon': {12, 24, 20, 52, 53, 29, 82, 25, 83, 14},
      'monarch_closed': {29, 9, 58, 72, 23, 44, 67, 36, 27, 28},
      'monarch_open': {46, 35, 58, 8, 76, 21, 34, 15, 83, 78},
      'peacock': {1, 96, 131, 124, 22, 59, 126, 52, 20, 86},
      'zebra': {57, 18, 21, 26, 43, 78, 31, 90, 6, 75}
  }

  assert set(test_images.keys()) == set(labels)
  assert all([len(s) == 10 for s in test_images.values()])

  def gen_split(select_test_split: bool):
    with zipfile.ZipFile(datafile, 'r') as zf:
      for member in zf.infolist():
        if member.is_dir():
          continue
        label_str, basename = os.path.split(member.filename)
        label_id = label_str_to_id[label_str]

        image_idx_match = re.search(r'[a-z]+(\d+)\.jpg', basename)
        if not image_idx_match:
          raise ValueError('Could not parse image filename')
        else:
          image_idx = int(image_idx_match.group(1))
        is_test_image = image_idx in test_images[label_str]
        if select_test_split != is_test_image:
          # Ignore image if not in the requested split
          continue

        image = Image.open(io.BytesIO(zf.read(member))).convert('RGB')
        image.load()
        yield (image, label_id)

  make_gen_fn = lambda: gen_split(select_test_split=False)
  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN,
      splits.MERGED_TRAIN_AND_DEV)
  per_split_gen['test'] = gen_split(select_test_split=True)

  return (metadata, per_split_gen)


butterflies_dataset = types.DownloadableDataset(
    name='butterflies',
    paper_title='Semi-Local Affine Parts for Object Recognition',
    paper_url='http://www.bmva.org/bmvc/2004/papers/paper_038.html',
    authors='S. Lazebnik, C. Schmid, J. Ponce',
    year=2004,
    website_url='https://web.archive.org/web/20070829035029/http://www-cvr.ai.uiuc.edu/ponce_grp/data/index.html',
    # This dataset cannot be downloaded from Cloudtop instance.
    download_urls=[
        types.DownloadableArtefact(
            url='https://web.archive.org/web/20070829035029/http://www-cvr.ai.uiuc.edu/ponce_grp/data/butterflies/butterflies.zip',
            checksum='cc9bfe5e22f9001262b785ff33221581')
    ],
    handler=butterflies_handler)
