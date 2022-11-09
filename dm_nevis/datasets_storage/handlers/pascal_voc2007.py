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

"""Pascal VOC 2007 dataset handler."""

import os
import tarfile
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
from PIL import Image
from tensorflow.io import gfile
import tensorflow_datasets as tfds

_LABEL_PATH = 'VOCdevkit/VOC2007/ImageSets/Main/'


def pascal_voc2007_handler(dataset_path: str) -> types.HandlerOutput:
  """Imports Pascal VOC 2007 dataset.

  Link: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/

  Args:
    dataset_path: Path with downloaded datafiles.

  Returns:
    Metadata and generator functions.
  """
  files = gfile.listdir(dataset_path)
  assert len(files) == 2
  first_file = files[0]
  is_train = 'trainval' in first_file
  pattern = '_trainval.txt' if is_train else '_test.txt'
  suffix_length = len(pattern)
  raw_file_path = dataset_path

  def extract_class_name(path: str, suffix_length: int):
    return path.split('/')[-1][:-suffix_length]

  def extract_tarinfos(tarfile_name: str, startstr: str, endstr: str):
    with tarfile.open(os.path.join(dataset_path, tarfile_name)) as tfile:
      return [
          tarinfo for tarinfo in tfile.getmembers()
          if (tarinfo.name.startswith(startstr) and
              tarinfo.name.endswith(endstr))]

  class_files = extract_tarinfos(first_file, _LABEL_PATH, pattern)
  classes = [extract_class_name(cf.name, suffix_length) for cf in class_files]
  num_classes = len(classes)
  assert num_classes == 20
  label_to_id = dict()
  for cc in range(num_classes):
    label_to_id[classes[cc]] = cc

  metadata = types.DatasetMetaData(
      num_classes=num_classes,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(label_to_id=label_to_id, image_type='object'),
      features=tfds.features.FeaturesDict({
          'multi_label':
              tfds.features.Sequence(
                  tfds.features.ClassLabel(num_classes=num_classes)),
          'png_encoded_image':
              tfds.features.Image()
      }))

  def get_image_label_pair(file_name: str):
    is_train = 'trainval' in file_name
    pattern = '_trainval.txt' if is_train else '_test.txt'
    suffix_length = len(pattern)
    class_files = extract_tarinfos(file_name, _LABEL_PATH, pattern)
    with tarfile.open(os.path.join(raw_file_path, file_name)) as tar:
      image_dict = dict()
      for cf in class_files:
        class_name = extract_class_name(cf.name, suffix_length)
        f_obj = tar.extractfile(cf)
        assert f_obj
        lines = f_obj.readlines()
        lines = [line.decode('utf-8') for line in lines]
        curr_image_list = [line[:-4] for line in lines if line.endswith(' 1\n')]
        for ci in curr_image_list:
          curr_key = ci + '.jpg'
          if curr_key not in image_dict.keys():
            image_dict[curr_key] = []
          image_dict[curr_key].append(label_to_id[class_name])
      # Extract actual images.
      tarinfos = [tarinfo for tarinfo in tar.getmembers()
                  if tarinfo.name.split('/')[-1] in image_dict.keys()]
      assert tarinfos
      for ti in tarinfos:
        f_obj = tar.extractfile(ti)
        image = Image.open(f_obj)
        image.load()
        attributes = image_dict[ti.name.split('/')[-1]]
        yield types.Example(image=image, multi_label=attributes, label=None)

  def gen_split(is_test_split: bool):
    if is_test_split:
      # extract test set
      file = [file for file in files if 'test' in file]
      assert len(file) == 1
      return get_image_label_pair(file[0])
    else:
      # extract training set
      file = [file for file in files if 'trainval' in file]
      assert len(file) == 1
      return get_image_label_pair(file[0])

  make_gen_fn = lambda: gen_split(is_test_split=False)
  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN,
      splits.MERGED_TRAIN_AND_DEV)
  per_split_gen['test'] = gen_split(is_test_split=True)

  return (metadata, per_split_gen)


pascal_voc2007_dataset = types.DownloadableDataset(
    name='pascal_voc2007',
    download_urls=[
        types.DownloadableArtefact(
            url='http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar',
            checksum='c52e279531787c972589f7e41ab4ae64'),
        types.DownloadableArtefact(
            url='http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar',
            checksum='b6e924de25625d8de591ea690078ad9f')
    ],
    handler=pascal_voc2007_handler)
