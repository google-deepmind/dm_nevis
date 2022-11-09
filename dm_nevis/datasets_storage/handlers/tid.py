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

"""TID handler."""

import os
import subprocess
import tarfile
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
from PIL import Image

from tensorflow.io import gfile

_NUM_SUBJECTS = 25
_PROPERTIES_PER_YEAR = {
    2008: dict(num_distortion_types=17, num_distortion_levels=4),
    2013: dict(num_distortion_types=24, num_distortion_levels=5),
}
_NOISE_FNAME_2008 = 'MSSIM.TXT'
_NOISE_FNAME_2013 = 'MSSIM.txt'
_NUM_BUCKETS = 5


def tid_handler(
    dataset_path: str,
    year: int,
    apply_unrar: bool = True
) -> types.HandlerOutput:
  """Handler for TID dataset."""

  metadata = types.DatasetMetaData(
      num_classes=_NUM_BUCKETS,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          task_type='classification',
          image_type='quality',
      ))

  # Unarachive the images.
  if apply_unrar:
    # Normally, we assume that we have access to unrar utility and use
    # rar archive.
    subprocess.call(['unrar', 'e', '-y', '-idq', f'tid{year}.rar'],
                    cwd=dataset_path)
  else:
    # In this case, we assume that we pre-archived a file in a tar format.
    tarfile.open(os.path.join(dataset_path, f'tid{year}.tar.gz'),
                 'r|gz').extractall(path=dataset_path)

  image_fname_to_distortion = {}

  noise_fname = _NOISE_FNAME_2008 if year == 2008 else _NOISE_FNAME_2013

  with gfile.GFile(os.path.join(dataset_path, noise_fname), 'r') as f:
    lines = iter(f.readlines())

  num_distortion_types = _PROPERTIES_PER_YEAR[year]['num_distortion_types']
  num_distortion_levels = _PROPERTIES_PER_YEAR[year]['num_distortion_levels']

  for subject_id in range(1, _NUM_SUBJECTS + 1):
    for distortion_id in range(1, num_distortion_types + 1):
      for distortion_level in range(1, num_distortion_levels + 1):
        image_fname = f'i{subject_id:02d}_{distortion_id:02d}_{distortion_level}.bmp'
        distortion = float(next(lines).strip())
        image_fname_to_distortion[image_fname] = distortion

  tot_num_samples = len(image_fname_to_distortion)
  num_examples_per_bucket = tot_num_samples // _NUM_BUCKETS
  sorted_distortions = sorted(image_fname_to_distortion.values())
  buckets = []
  for bucket_id in range(_NUM_BUCKETS):
    buckets.append(
        sorted_distortions[bucket_id * num_examples_per_bucket:(bucket_id + 1) *
                           num_examples_per_bucket])

  def find_bucket(distortion, buckets):
    for bucket_id, bucket in enumerate(buckets):
      if distortion in bucket:
        return bucket_id

  def gen(image_fname_to_distortion, buckets):
    for fname in gfile.listdir(dataset_path):
      if not fname.endswith('bmp'):
        continue
      image_fname = fname.lower()
      if image_fname not in image_fname_to_distortion:
        continue
      distortion = image_fname_to_distortion[image_fname]
      bucket_id = find_bucket(distortion, buckets)
      image = Image.open(os.path.join(dataset_path, fname))
      yield types.Example(image=image, label=bucket_id, multi_label=None)

  make_gen = lambda: gen(image_fname_to_distortion, buckets)

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)

  return metadata, per_split_gen


tid2008_dataset = types.DownloadableDataset(
    name='tid2008',
    download_urls=[
        types.DownloadableArtefact(
            url='https://ponomarenko.info/tid/tid2008.rar',
            checksum='732aeec405a57b589aeee4f1c58a4c1b')
    ],
    website_url='https://ponomarenko.info/tid2008.htm',
    handler=lambda ds: tid_handler(ds, 2008))

tid2013_dataset = types.DownloadableDataset(
    name='tid2013',
    download_urls=[
        types.DownloadableArtefact(
            url='https://ponomarenko.info/tid2013/tid2013.rar',
            checksum='2c42bec2407adc49cf4d7858daf8f99b')
    ],
    website_url='https://ponomarenko.info/tid2013.htm',
    handler=lambda ds: tid_handler(ds, 2013))
