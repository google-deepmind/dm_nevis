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

"""Aberdeen handler."""

import io
import os
import re
import zipfile
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
from PIL import Image


_REGEX = r'[a-zA-Z_]+'
_FNAME = 'Aberdeen.zip'

_DUPLICATES = {
    'marieg': 'marie',
    'heatherg': 'heather',
    'kayg': 'kay',
    'neilg': 'neil',
    'graemeg': 'graeme',
    'dhawleyg': 'dhawley',
    'clean_adrian': 'adrian',
    'jsheenang': 'jsheenan',
}

_LABELS = [
    'neil',
    'jenni',
    'chris_harbron',
    'john_mccal',
    'blaw',
    'dpearson',
    'ruth',
    'richard_hardwick',
    'olive',
    'dlow',
    'simon',
    'dougal_grant',
    'mnicholson',
    'hack',
    'caroline',
    'lynn',
    'adrian',
    'kay',
    'fiona_hogarth',
    'annanena',
    'heather',
    'barry',
    'jsheenang',
    'michael',
    'alister',
    'amellanby',
    'george',
    'graham_brown',
    'itaylor',
    'marie',
    'david',
    'jim',
    'alison',
    'trevor',
    'iroy',
    'scott',
    'louise',
    'dsmith',
    'gfindley',
    'irene',
    'tracy',
    'johannes',
    'chris_pin',
    'anon_one',
    'stewart',
    'lynn_james',
    'peter',
    'paul',
    'pkyle',
    'andrew',
    'mmanson',
    'graeme',
    'fiona',
    'ghall',
    'paol',
    'david_imray',
    'john_thom',
    'stephen',
    'gordon',
    'gillian',
    'dhands',
    'joanna',
    'nick',
    'bfegan',
    'grant_cumming',
    'alec',
    'milly',
    'merilyn',
    'kirsty',
    'peter_macgeorge',
    'dbell',
    'chris',
    'miranda',
    'johnny_page',
    'pat',
    'terry_johnstone',
    'tock',
    'catherine',
    'blair',
    'kieran',
    'martin',
    'hin',
    'meggan',
    'jsheenan',
    'brian_ho',
    'mark',
    'dhawley',
    'derek',
    'lisa',
    'ian',
    'kim',
    'dave_faquhar',
]


def aberdeen_handler(dataset_path: str) -> types.HandlerOutput:
  """Handler for aberdeen dataset."""
  num_classes = len(_LABELS)
  label_to_id = dict(((label, idx) for idx, label in enumerate(_LABELS)))

  def make_gen():
    with zipfile.ZipFile(os.path.join(dataset_path, _FNAME), 'r') as zf:
      for member in zf.infolist():
        img_fname = os.path.splitext(member.filename)[0]
        label_name = re.search(_REGEX, img_fname)[0]
        if label_name in _DUPLICATES:
          label_name = _DUPLICATES[label_name]
        label = label_to_id[label_name]
        image = Image.open(io.BytesIO(zf.read(member)))
        yield image, label

  metadata = types.DatasetMetaData(
      num_classes=num_classes,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(label_to_id=label_to_id))

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)

  return (metadata, per_split_gen)


aberdeen_dataset = types.DownloadableDataset(
    name='aberdeen',
    download_urls=[
        types.DownloadableArtefact(
            url='http://pics.stir.ac.uk/zips/Aberdeen.zip',
            checksum='1f7044bd5f0bed01286263aa580a7a87'),
    ],
    handler=aberdeen_handler,
)
