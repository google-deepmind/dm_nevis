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

"""BIWI dataset handler."""

import io
import math
import os
from typing import Dict, List
import zipfile
from dm_nevis.datasets_storage.handlers import extraction_utils as utils
from dm_nevis.datasets_storage.handlers import types
import numpy as np
import pyquaternion
from tensorflow.io import gfile
import tensorflow_datasets as tfds


_IDS = {
    'test': [1, 4, 8, 10],
    'dev-test': [3, 9],
    'dev': [2, 7],
    'train': [
        5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    'train_and_dev': [
        5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 2, 7]
}

_POSE_SUFFIX = '_pose.txt'
_LEN_POSE_SUFFIX = len(_POSE_SUFFIX)
_NUM_CLASSES = 5  # Number of quantization bins for the 3 angles.
_SPLITS = ['train', 'dev', 'dev-test', 'train_and_dev', 'test']
_ANGLES = ['roll', 'pitch', 'yaw']
_NANGLES = len(_ANGLES)

_IGNORED_FILES_REGEX = '|'.join([
    utils.DEFAULT_IGNORED_FILES_REGEX,
    r'_depth.bin',
    r'rgb.cal',
    r'readme.txt',
    r'_pose.bin',
    r'_mask.png',
])


def _path_to_label_fn(path: str, file_to_labels: Dict[str,
                                                      List[int]]) -> List[int]:
  filename = os.path.basename(path)
  return file_to_labels[filename]


def biwi_handler(dataset_path: str) -> types.HandlerOutput:
  """Imports BIWI dataset.

  The dataset home page is at
  https://www.kaggle.com/datasets/kmader/biwi-kinect-head-pose-database
  This dataset contains over 20,000 face images of 20 people under different
  3D pose conditions. The task is about pose estimation.
  The creators provide camera calibration, and for each image a 3D point in
  space with a 3x3 rotation matrix.
  We turn this into a multilabel classification task by:
  a) Computing roll, pitch and yaw.
  b) Quantizing the above quantities.
  Therefore for every image we need to predict 3 categories.

  In the paper where we sourced this task, Pan et al. Self-Paced Deep Regression
  Forests with Consideration on Underrepresented Examples ECCV 2020, the authors
  create splits by randomly shuffling images.
  Instead we create splits by using non-overlapping set of subjects for each
  split, which tests for a harder generalization.

  The package is organized in three folders: db_annotations, faces_0,
  head_pose_masks. The data of interest is in the folder faces_0.
  There we find one folder per subject.
  For instance:
  faces_0/08/frame_00150_rgb.png
  is an image of subject 08, and the corresponding pose information is at:
  faces_0/08/frame_00150_pose.txt
  which in this case contains:
  0.784991 -0.0678137 0.615784
  -0.117134 0.959814 0.255021
  -0.608332 -0.272319 0.745503

  88.7665 -4.08406 878.874
  i.e. 3x3 rotation matrix and vector of 3D coordinates.

  Args:
    dataset_path: Path with downloaded artifacts.

  Returns:
    Metadata and generator functions.
  """
  zip_file_path, *other_files_in_directory = gfile.listdir(dataset_path)
  assert not other_files_in_directory

  all_angles = []
  data = dict()
  # Read all the pose data, convert it to roll, pitch and yaw.
  with zipfile.ZipFile(os.path.join(dataset_path, zip_file_path), 'r') as zf:
    for name in sorted(zf.namelist()):
      f = zf.getinfo(name)
      if f.filename.endswith(_POSE_SUFFIX):
        with io.TextIOWrapper(zf.open(f), encoding='utf-8') as fp:
          rotation_list = []
          counter = 0
          for line in fp:
            vals = line.strip().split(' ')
            rotation_list.append([float(v) for v in vals])
            counter += 1
            if counter == _NANGLES:
              break  # Disregard 3D location.
          rotation_mat = np.array(rotation_list)
          q = pyquaternion.Quaternion(matrix=rotation_mat, atol=1e-4)
          angles = q.yaw_pitch_roll
          angles = [math.degrees(a) for a in angles]
          all_angles.append(list(angles))
          filename = f.filename[:-_LEN_POSE_SUFFIX]
          img_fname = f'{filename}_rgb.png'
          data[img_fname] = angles

    # Quantize angles.
    angles_mat = np.array(all_angles)  # num_examples x 3

    def uniform_bins(data, num_bins):
      num_samples = len(data)
      return np.interp(
          # Add one to the number of bins because the first and last value are
          # min and max, and if we want K intervals we need K+1 boundaries.
          np.linspace(0, num_samples, num_bins + 1), np.arange(num_samples),
          np.sort(data))

    bins = []
    for i in range(_NANGLES):
      bins.append(uniform_bins(angles_mat[:, i], _NUM_CLASSES))

    for name in data:
      deg_angles = data[name]
      sample_labels = []
      for i in range(_NANGLES):
        assert deg_angles[i] <= bins[i][-1] and deg_angles[i] >= bins[i][0]
        quantized_angle = np.digitize(deg_angles[i], bins[i])
        # Note: np.digitize returns indexes of bins s.t.:
        # bins[i][j-1] <= x < bins[i][j]
        # If x == bin[i][-1], it will be assigned to the next bin, with index
        # _NUM_CLASSES + 1. Instead we want that boundary value to be mapped to
        # the last bin, bins[i][-1].
        if deg_angles[i] == bins[i][-1]:
          quantized_angle = _NUM_CLASSES
        # This will create a list of multinomial labels, which is currently not
        # supported.
        # sample_labels.append(quantized_angle - 1)  # in [0, _NUM_CLASSES-1]
        # Instead, we use binary labels, and here we list the indexes
        # of the non-zero labels.
        sample_labels.append((quantized_angle - 1) + _NUM_CLASSES * i)
      data[name] = sample_labels

  metadata = types.DatasetMetaData(
      # In the binary setting only, this is the total number of binary labels.
      num_classes=_NUM_CLASSES * _NANGLES,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          task_type='multi-label',
          image_type='faces',
      ),
      features=tfds.features.FeaturesDict({
          'multi_label':
              tfds.features.Sequence(
                  tfds.features.ClassLabel(num_classes=_NUM_CLASSES)),
          'png_encoded_image':
              tfds.features.Image()
      }))

  def gen(split):

    def path_filter_fn(path: str) -> bool:
      # Remove files which are not rgb images.
      if path not in data.keys():
        return False
      # Get subject id.
      subject_id = os.path.basename(os.path.dirname(path))
      assert len(subject_id) == 2
      subject_id = int(subject_id)
      return subject_id in _IDS[split]

    return utils.generate_images_from_zip_files_with_multilabels(
        dataset_path=dataset_path,
        zip_file_names=[zip_file_path],
        path_to_attributes_fn=lambda path: data[path],
        ignored_files_regex=_IGNORED_FILES_REGEX,
        path_filter=path_filter_fn)

  per_split_gen = {}
  for split in _SPLITS:
    per_split_gen[split] = gen(split)

  return metadata, per_split_gen


biwi_dataset = types.DownloadableDataset(
    name='biwi',
    download_urls=[
        types.KaggleDataset(
            dataset_name='kmader/biwi-kinect-head-pose-database',
            checksum='59d49d96e5719f248f6d66f8ff205569')
    ],
    paper_title='Random Forests for Real Time 3D Face Analysis',
    authors='Fanelli, Gabriele and Dantone, Matthias and Gall, Juergen and Fossati, Andrea and Van Gool, Luc',
    year='2013',
    website_url='https://www.kaggle.com/datasets/kmader/biwi-kinect-head-pose-database',
    handler=biwi_handler)
