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

"""IAPRTC-12 dataset handler."""

import collections
import os
import pathlib
import zipfile
from dm_nevis.datasets_storage.handlers import extraction_utils
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
import numpy as np
import scipy.io
from tensorflow.io import gfile
import tensorflow_datasets as tfds


_LABEL_MAPPING_PATH = 'saiaprtc12ok/benchmark/wlist.txt'
_TRAINING_LIST_PATH = 'saiaprtc12ok/matlab/matlab/training.mat'
_VALIDATION_LIST_PATH = 'saiaprtc12ok/matlab/matlab/validation.mat'
_TEST_LIST_PATH = 'saiaprtc12ok/matlab/matlab/testing.mat'
_LABEL_PATH = 'saiaprtc12ok/benchmark/saiapr_tc-12/' + '{idx:0=2d}' + '/labels.txt'
_IMAGE_PATH = '/images/'
_DUPLICATE = 'saiaprtc12ok'
_NUM_IMAGE_DIRECTORIES = 41

IGNORED_FILES_REGEX = r'.*\.eps$|.*\.txt$|.*\.mat$|.*\.pdf$'


def iaprtc12_handler(artifacts_path: str) -> types.HandlerOutput:
  """Imports IAPRTC-12 dataset.

  The dataset homepage is at:
  https://www.kaggle.com/datasets/nastyatima/iapr-tc12
  The paper proposing the oringial dataset is:
  The IAPR Benchmark: A New Evaluation Resource for Visual Information Systems,
  Grubinger, Michael, Clough Paul D., Müller Henning, and Deselaers Thomas ,
  International Conference on Language Resources and Evaluation, 24/05/2006,
  Genoa, Italy, (2006)
  In this handler we use the version with additional annotation and
  training/valid/test splits provided by H.J. Escalante et al. (see below for a
  full reference).

  This is a multi-label classification dataset of natural images.
  There are 20000 images in total, and 276 possible classes.
  In the directory saiaprtc12ok/matlab/matlab there are:
  testing.mat   training.mat    validation.mat
  which store the ids of the corresponding splits.

  saiaprtc12ok/benchmark/wlist.txt stores the association between label string
  and label integer id.

  saiaprtc12ok/benchmark/saiapr_tc-12/xx/labels.txt with xx in [00..40] is a
  file storing the image id in that folder, region id (not used by us) and the
  label id.
  saiaprtc12ok/benchmark/saiapr_tc-12/xx/images/yyy.jpg with xx in [00..40] and
  yyy the image id are the folders storing the actual images.

  Args:
    artifacts_path: Path with downloaded artifacts.

  Returns:
    Metadata and generator functions.
  """
  files = gfile.listdir(artifacts_path)
  assert len(files) == 1
  class_name_to_index = dict()
  ids = dict()
  with zipfile.ZipFile(os.path.join(artifacts_path, files[0]), 'r') as zf:
    with zf.open(_LABEL_MAPPING_PATH) as fid:
      for line in fid:
        fields = line.decode('UTF-8').split('\t')
        fields[1] = fields[1].strip()  # remove trailing \n
        class_name_to_index[
            fields[1] if fields[1] else ' '] = int(fields[0]) - 1

  def get_image_ids(path, name):
    with zipfile.ZipFile(os.path.join(artifacts_path, files[0]), 'r') as zf:
      with zf.open(path) as fo:
        mat = scipy.io.loadmat(fo)
        return mat[name].squeeze().tolist()

  ids['training'] = get_image_ids(_TRAINING_LIST_PATH, 'training')
  ids['validation'] = get_image_ids(_VALIDATION_LIST_PATH, 'validation')
  ids['all_training'] = ids['training'] + ids['validation']
  ids['testing'] = get_image_ids(_TEST_LIST_PATH, 'testing')

  def extract_labels(path):
    # This is a dictionary where keys are image ids, and values are labels.
    # Since it is a multi-label classification problem, there could be a
    # variable (>1) number of labels per image.
    output = collections.defaultdict(list)  # init with empty list
    with zipfile.ZipFile(os.path.join(artifacts_path, files[0]), 'r') as zf:
      if path in zf.namelist():
        with zf.open(path) as fo:
          data = np.loadtxt(fo)
          for cnt in range(data.shape[0]):
            if output[data[cnt][0]]:
              output[int(data[cnt][0])] = [int(data[cnt][2] - 1)]
            else:
              output[int(data[cnt][0])].append(int(data[cnt][2] - 1))
    return output

  imageid_to_label = dict()
  for folder_id in range(0, _NUM_IMAGE_DIRECTORIES):
    imageid_to_label.update(extract_labels(_LABEL_PATH.format(idx=folder_id)))

  def _labels_from_path_fn(fname):
    name = int(pathlib.Path(fname).stem)
    assert name in imageid_to_label.keys()
    all_labels = list(set(imageid_to_label[name]))
    return all_labels

  def _path_filter(fname, is_train):
    img_ids = ids['all_training'] if is_train else ids['testing']
    img_id = int(pathlib.Path(fname).stem)
    return ((_DUPLICATE not in fname) and  # Remove duplicate folder.
            (_IMAGE_PATH in fname) and  # Remove segmentations.
            (img_id in img_ids))  # Consider images in the desired set.

  def gen_split(is_train):
    return extraction_utils.generate_images_from_zip_files_with_multilabels(
        artifacts_path,
        files,
        _labels_from_path_fn,
        ignored_files_regex=IGNORED_FILES_REGEX,
        path_filter=lambda path: _path_filter(path, is_train),
        convert_mode='RGB')

  num_classes = len(class_name_to_index.keys())
  metadata = types.DatasetMetaData(
      num_classes=num_classes,
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_to_id=class_name_to_index,
          task_type='multi-label',
          image_type='object'),
      features=tfds.features.FeaturesDict({
          'multi_label':
              tfds.features.Sequence(
                  tfds.features.ClassLabel(num_classes=num_classes)),
          'png_encoded_image':
              tfds.features.Image()
      }))

  # TODO: Make more efficient deduplication algorithm.
  make_unique_gen_fn = lambda: gen_split(is_train=True)
  # TODO: re-enable de-duplication.
  # extraction_utils.deduplicate_data_generator(
  #    itertools.chain(gen_split(is_train=True), gen_split(is_train=False)))

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_unique_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)

  return metadata, per_split_gen

iaprtc12_dataset = types.DownloadableDataset(
    name='iaprtc12',
    download_urls=[
        types.KaggleDataset(
            dataset_name='nastyatima/iapr-tc12',
            checksum='ee251615ac2dbb55eea5a7e5e710740a')
    ],
    website_url='https://www.kaggle.com/datasets/nastyatima/iapr-tc12',
    paper_url='https://ccc.inaoep.mx/~emorales/Papers/2010/hugo.pdf',
    authors='Hugo Jair Escalante, and Carlos A. Hern<E1>ndez, and Jesus A. Gonzalez, and A. L<F3>pez-L<F3>pez, and Manuel Montes, and Eduardo F. Morales, and L. Enrique Sucar, and Luis Villase<F1>or and Michael Grubinger',
    papers_with_code_url='https://paperswithcode.com/dataset/iapr-tc-12',
    handler=iaprtc12_handler)
