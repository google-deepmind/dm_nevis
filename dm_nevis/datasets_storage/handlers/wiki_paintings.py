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

"""Wiki Paintrings handler."""

import io
import os
import zipfile
import zlib

from absl import logging
from dm_nevis.datasets_storage.handlers import extraction_utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
import pandas as pd
from PIL import Image


# Resize images for this dataset to the given max size. The original images can
# have much larger sizes, however this is unnecessarily high for the task, and
# results in slower training due to increased decoding time.
MAXIMUM_IMAGE_SIZE = 256

_IMAGE_FNAME = 'wikiart.zip'
_CSV_FNAME = 'wikiart_csv.zip'

# Style, Artist, and Genre classification
_ARTIST_TO_CLASS_ID = {
    'Albrecht_Durer': 0,
    'Boris_Kustodiev': 1,
    'Camille_Pissarro': 2,
    'Childe_Hassam': 3,
    'Claude_Monet': 4,
    'Edgar_Degas': 5,
    'Eugene_Boudin': 6,
    'Gustave_Dore': 7,
    'Ilya_Repin': 8,
    'Ivan_Aivazovsky': 9,
    'Ivan_Shishkin': 10,
    'John_Singer_Sargent': 11,
    'Marc_Chagall': 12,
    'Martiros_Saryan': 13,
    'Nicholas_Roerich': 14,
    'Pablo_Picasso': 15,
    'Paul_Cezanne': 16,
    'Pierre_Auguste_Renoir': 17,
    'Pyotr_Konchalovsky': 18,
    'Raphael_Kirchner': 19,
    'Rembrandt': 20,
    'Salvador_Dali': 21,
    'Vincent_van_Gogh': 22,
}

STYLE_TO_CLASS_ID = {
    'Abstract_Expressionism': 0,
    'Action_painting': 1,
    'Analytical_Cubism': 2,
    'Art_Nouveau': 3,
    'Baroque': 4,
    'Color_Field_Painting': 5,
    'Contemporary_Realism': 6,
    'Cubism': 7,
    'Early_Renaissance': 8,
    'Expressionism': 9,
    'Fauvism': 10,
    'High_Renaissance': 11,
    'Impressionism': 12,
    'Mannerism_Late_Renaissance': 13,
    'Minimalism': 14,
    'Naive_Art_Primitivism': 15,
    'New_Realism': 16,
    'Northern_Renaissance': 17,
    'Pointillism': 18,
    'Pop_Art': 19,
    'Post_Impressionism': 20,
    'Realism': 21,
    'Rococo': 22,
    'Romanticism': 23,
    'Symbolism': 24,
    'Synthetic_Cubism': 25,
    'Ukiyo_e': 26,
}

_GENRE_TO_CLASS_ID = {
    'abstract_painting': 0,
    'cityscape': 1,
    'genre_painting': 2,
    'illustration': 3,
    'landscape': 4,
    'nude_painting': 5,
    'portrait': 6,
    'religious_painting': 7,
    'sketch_and_study': 8,
    'still_life': 9,
}

_TASKS = ['artist', 'style', 'genre']


def wiki_paintings_handler(dataset_path: str,
                           task: str = 'style') -> types.HandlerOutput:
  """Handler for Wiki Paintings dataset."""
  assert task in _TASKS

  train_csv = f'{task}_train.csv'
  val_csv = f'{task}_val.csv'

  with zipfile.ZipFile(os.path.join(dataset_path, _CSV_FNAME), 'r') as zf:
    train_ids = pd.read_csv(zf.extract(train_csv, path=dataset_path))
    val_ids = pd.read_csv(zf.extract(val_csv, path=dataset_path))

  def gen(ids):
    with zipfile.ZipFile(os.path.join(dataset_path, _IMAGE_FNAME), 'r') as zf:
      for _, row in ids.iterrows():
        image_name, class_id = row
        image_fname = os.path.join('wikiart', image_name)
        try:
          image = Image.open(io.BytesIO(zf.read(image_fname)))
          image = extraction_utils.resize_to_max_size(image, MAXIMUM_IMAGE_SIZE)
          yield types.Example(image=image, label=class_id, multi_label=None)
        except (zlib.error, zipfile.BadZipFile):
          # Very few images cannot be read.
          logging.warning('Skipping %s', image_fname)

  train_gen_fn = lambda: gen(train_ids)
  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      train_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN,
      splits.MERGED_TRAIN_AND_DEV)
  per_split_gen['test'] = gen(val_ids)

  if task == 'artist':
    label_to_id = _ARTIST_TO_CLASS_ID
    num_classes = 23
  elif task == 'style':
    label_to_id = STYLE_TO_CLASS_ID
    num_classes = 27
  elif task == 'genre':
    label_to_id = _GENRE_TO_CLASS_ID
    num_classes = 10

  metadata = types.DatasetMetaData(
      num_classes=num_classes,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(label_to_id=label_to_id))

  return (metadata, per_split_gen)


img_artefact = types.DownloadableArtefact(
    url='http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip',
    checksum='c6b43cbcd474b875a5626ffde3b627e1')
csv_artefact = types.DownloadableArtefact(
    url='http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart_csv.zip',
    checksum='0d221e7cb0812da6b59044cfca9aafee')

# TODO: redundant DL
wiki_paintings_dataset_artist = types.DownloadableDataset(
    name='wiki_paintings_artist',
    download_urls=[img_artefact, csv_artefact],
    website_url='https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset',
    handler=lambda ds: wiki_paintings_handler(ds, task='artist'),
)

wiki_paintings_dataset_style = types.DownloadableDataset(
    name='wiki_paintings_style',
    download_urls=[img_artefact, csv_artefact],
    website_url='https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset',
    handler=lambda ds: wiki_paintings_handler(ds, task='style'),
)

wiki_paintings_dataset_genre = types.DownloadableDataset(
    name='wiki_paintings_genre',
    download_urls=[img_artefact, csv_artefact],
    website_url='https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset',
    handler=lambda ds: wiki_paintings_handler(ds, task='genre'),
)
