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

"""Pascal VOC 2005 dataset handler."""

import os
import tarfile
from typing import Dict, List
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
from PIL import Image

from tensorflow.io import gfile


def pascal_voc2005_handler(dataset_path: str) -> types.HandlerOutput:
  """Imports Pascal VOC 2005 dataset.

  Links: http://host.robots.ox.ac.uk/pascal/VOC/databases.html#VOC2005_1
         http://host.robots.ox.ac.uk/pascal/VOC/databases.html#VOC2005_2

  Args:
    dataset_path: Path with datafiles to download.

  Returns:
    Metadata and generator functions.
  """
  files = gfile.listdir(dataset_path)
  raw_file_path = dataset_path
  assert len(files) == 2
  labels = ['car', 'motorbike', 'bike', 'person']
  label_to_id = {'car': 0,
                 'motorbike': 1,
                 'bike': 2,
                 'person': 3}
  # Note: we removed from the original directories  testing files from training
  # set and training files from test set.
  training_paths = {'car': ['VOC2005_1/PNGImages/Caltech_cars',
                            'VOC2005_1/PNGImages/ETHZ_sideviews-cars',
                            'VOC2005_1/PNGImages/TUGraz_cars'],
                    'bike': ['VOC2005_1/PNGImages/TUGraz_bike'],
                    'motorbike': [
                        'VOC2005_1/PNGImages/Caltech_motorbikes_side'],
                    'person': ['VOC2005_1/PNGImages/TUGraz_person']}
  test_paths = {'car': ['VOC2005_2/PNGImages/car',
                        'VOC2005_2/PNGImages/voiture'],
                'bike': ['VOC2005_2/PNGImages/bicycle',
                         'VOC2005_2/PNGImages/bike',
                         'VOC2005_2/PNGImages/velo'],
                'motorbike': ['VOC2005_2/PNGImages/motocyclette',
                              'VOC2005_2/PNGImages/motorbike',
                              'VOC2005_2/PNGImages/motorcycle'],
                'person': ['VOC2005_2/PNGImages/INRIA_graz-person-test',
                           'VOC2005_2/PNGImages/INRIA_inria-person-test',
                           'VOC2005_2/PNGImages/pedestrian',
                           'VOC2005_2/PNGImages/pieton']}

  num_classes = len(labels)

  metadata = types.DatasetMetaData(
      num_classes=num_classes,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_to_id=label_to_id,
          task_type='classification',
          image_type='object'))

  def get_image_label(file: str, all_paths: Dict[str, List[str]],
                      raw_file_path: str):
    with tarfile.open(os.path.join(raw_file_path, file), 'r:gz') as tar:
      for label, paths in all_paths.items():
        for path in paths:
          all_images = [
              tarinfo for tarinfo in tar.getmembers()
              if (tarinfo.name.startswith(path) and
                  tarinfo.name.endswith('png'))]
          assert all_images
          for image_file in all_images:
            f_obj = tar.extractfile(image_file)
            image = Image.open(f_obj)
            image.load()
            yield (image, label_to_id[label])

  def gen_split(is_test_split: bool):
    if is_test_split:
      # extract test set
      file = [file for file in files if file.endswith('2.tar.gz')]
      assert len(file) == 1
      return get_image_label(file[0], test_paths, raw_file_path)
    else:
      # extract training set
      file = [file for file in files if file.endswith('1.tar.gz')]
      assert len(file) == 1
      return get_image_label(file[0], training_paths, raw_file_path)

  make_gen_fn = lambda: gen_split(is_test_split=False)
  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN,
      splits.MERGED_TRAIN_AND_DEV)
  per_split_gen['test'] = gen_split(is_test_split=True)

  return (metadata, per_split_gen)


pascal_voc2005_dataset = types.DownloadableDataset(
    name='pascal_voc2005',
    download_urls=[
        types.DownloadableArtefact(
            url='http://host.robots.ox.ac.uk/pascal/VOC/download/voc2005_1.tar.gz',
            checksum='6fbeaee73a81c462b190ca837b977896'),
        types.DownloadableArtefact(
            url='http://host.robots.ox.ac.uk/pascal/VOC/download/voc2005_2.tar.gz',
            checksum='15ec3d318b84ffdfa25f1e05de0014e2')
    ],
    handler=pascal_voc2005_handler)
