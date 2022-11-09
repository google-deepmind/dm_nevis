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

"""Pascal VOC 2006 dataset handler."""

import os
import tarfile
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
from PIL import Image

from tensorflow.io import gfile

_LABEL_PATH = 'VOCdevkit/VOC2006/ImageSets/'


def pascal_voc2006_handler(dataset_path: str) -> types.HandlerOutput:
  """Imports Pascal VOC 2006 dataset.

  Link: http://host.robots.ox.ac.uk/pascal/VOC/databases.html#VOC2006

  Args:
    dataset_path: Path with downloaded datafiles.

  Returns:
    Metadata and generator functions.
  """
  files = gfile.listdir(dataset_path)
  assert len(files) == 2
  first_file = files[0]
  is_train = first_file.endswith('_trainval.tar')
  pattern = '_trainval.txt' if is_train else '_test.txt'
  suffix_length = len(pattern)
  raw_file_path = dataset_path
  # Extract class names and their number.
  with tarfile.open(os.path.join(dataset_path, first_file)) as tfile:
    class_files = [
        tarinfo for tarinfo in tfile.getmembers()
        if (tarinfo.name.startswith(_LABEL_PATH) and
            tarinfo.name.endswith(pattern))]
    classes = [cf.name.split('/')[-1][:-suffix_length] for cf in class_files]
    num_classes = len(classes)
  label_to_id = dict()
  for cc in range(num_classes):
    label_to_id[classes[cc]] = cc

  metadata = types.DatasetMetaData(
      num_classes=num_classes,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_to_id=label_to_id,
          task_type='classification',
          image_type='object'))

  def get_image_label_pair(file: str):
    is_train = file.endswith('_trainval.tar')
    pattern = '_trainval.txt' if is_train else '_test.txt'
    suffix_length = len(pattern)
    with tarfile.open(os.path.join(raw_file_path, file)) as tar:
      # Extract list of images and their label.
      class_files = [
          tarinfo for tarinfo in tar.getmembers()
          if (tarinfo.name.startswith(_LABEL_PATH) and
              tarinfo.name.endswith(pattern))]
      image_dict = dict()
      for cf in class_files:
        class_name = cf.name.split('/')[-1][:-suffix_length]
        f_obj = tar.extractfile(cf)
        assert f_obj
        lines = f_obj.readlines()
        lines = [ll.decode('utf-8') for ll in lines]
        curr_image_list = [ll[:-4] for ll in lines if ll.endswith(' 1\n')]
        for ci in curr_image_list:
          image_dict[ci + '.png'] = label_to_id[class_name]
      # Extract actual images.
      tarinfos = [tarinfo for tarinfo in tar.getmembers()
                  if tarinfo.name.split('/')[-1] in image_dict.keys()]
      assert tarinfos
      for ti in tarinfos:
        f_obj = tar.extractfile(ti)
        image = Image.open(f_obj)
        image.load()
        label = image_dict[ti.name.split('/')[-1]]
        yield (image, label)

  def gen_split(is_test_split: bool):
    if is_test_split:
      # extract test set
      file = [file for file in files if file.endswith('_test.tar')]
      assert len(file) == 1
      return get_image_label_pair(file[0])
    else:
      # extract training set
      file = [file for file in files if file.endswith('_trainval.tar')]
      assert len(file) == 1
      return get_image_label_pair(file[0])

  make_gen_fn = lambda: gen_split(is_test_split=False)
  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN,
      splits.MERGED_TRAIN_AND_DEV)
  per_split_gen['test'] = gen_split(is_test_split=True)

  return (metadata, per_split_gen)


pascal_voc2006_dataset = types.DownloadableDataset(
    name='pascal_voc2006',
    download_urls=[
        types.DownloadableArtefact(
            url='http://host.robots.ox.ac.uk/pascal/VOC/download/voc2006_trainval.tar',
            checksum='af06612e5ad9863bde6fa7aae55f8866'),
        types.DownloadableArtefact(
            url='http://host.robots.ox.ac.uk/pascal/VOC/download/voc2006_test.tar',
            checksum='6bd028d82d057621c4fc69e9c56517ef')
    ],
    handler=pascal_voc2006_handler)
