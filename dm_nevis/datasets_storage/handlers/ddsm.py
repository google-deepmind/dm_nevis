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

"""DDSM dataset handler.

Breast-cancer dataset, described in
https://www.kaggle.com/awsaf49/cbis-ddsm-breast-cancer-image-dataset. The
dataset comes with a possibility of using full or cropped image. In the handler,
we made a choice of using the full image. Moreover, the dataset comes with 2
different train/test splits called `mass_case` and `calc_case`. We merged
together the 2 splits into single train/test split.

"""

import os
import zipfile
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
import pandas as pd
from PIL import Image

# Can also use 'cropped image file path'
_IMAGE_FILE_PATH_NAME = 'image file path'
_DICOM_INFO_FILE_PATH = 'csv/dicom_info.csv'
_CALC_TEST_SET_FILE_PATH = 'csv/calc_case_description_test_set.csv'
_CALC_TRAIN_SET_FILE_PATH = 'csv/calc_case_description_train_set.csv'
_MASS_TEST_FILE_PATH = 'csv/mass_case_description_test_set.csv'
_MASS_TRAIN_FILE_PATH = 'csv/mass_case_description_train_set.csv'
_ARCHIVE_NAME = 'cbis-ddsm-breast-cancer-image-dataset.zip'


# pylint:disable=missing-function-docstring
def ddsm_handler(dataset_path: str) -> types.HandlerOutput:
  labels = ['BENIGN', 'MALIGNANT', 'BENIGN_WITHOUT_CALLBACK']

  label_to_id = dict(
      ((label, label_id) for label_id, label in enumerate(labels)))

  metadata = types.DatasetMetaData(
      num_classes=len(labels),
      num_channels=1,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_to_id=label_to_id,
          task_type='classification',
          image_type='xray',
      ))

  with zipfile.ZipFile(os.path.join(dataset_path, _ARCHIVE_NAME), 'r') as zf:

    dicom_info = pd.read_csv(zf.open(_DICOM_INFO_FILE_PATH))
    dicom_info['extracted_patient_id'] = dicom_info['PatientID']

    def _extract_image_path(path):
      return os.path.join('jpeg', os.path.basename(os.path.dirname(path)),
                          os.path.basename(path))

    dicom_info['image_path'] = dicom_info['image_path'].apply(
        _extract_image_path)

    calc_case_description_test_set = pd.read_csv(
        zf.open(_CALC_TEST_SET_FILE_PATH))
    calc_case_description_train_set = pd.read_csv(
        zf.open(_CALC_TRAIN_SET_FILE_PATH))
    mass_case_description_test_set = pd.read_csv(zf.open(_MASS_TEST_FILE_PATH))
    mass_case_description_train_set = pd.read_csv(
        zf.open(_MASS_TRAIN_FILE_PATH))

    train_set = pd.concat(
        [calc_case_description_train_set, mass_case_description_train_set])
    test_set = pd.concat(
        [calc_case_description_test_set, mass_case_description_test_set])
    train_set['extracted_patient_id'] = train_set[_IMAGE_FILE_PATH_NAME].apply(
        lambda x: x.split('/')[0])
    test_set['extracted_patient_id'] = test_set[_IMAGE_FILE_PATH_NAME].apply(
        lambda x: x.split('/')[0])

    train_set = pd.merge(
        train_set, dicom_info, on='extracted_patient_id', how='inner')
    train_set = train_set.drop_duplicates(
        subset='image_path', ignore_index=True)
    test_set = pd.merge(
        test_set, dicom_info, on='extracted_patient_id', how='inner')
    test_set = test_set.drop_duplicates(subset='image_path', ignore_index=True)

  def gen(data_df, label_to_id):
    with zipfile.ZipFile(os.path.join(dataset_path, _ARCHIVE_NAME), 'r') as zf:
      for _, row in data_df.iterrows():
        image_path = row['image_path']
        label = label_to_id[row.pathology]
        image = Image.open(zf.open(image_path))
        image.load()
        yield (image, label)

  make_gen_fn = lambda: gen(train_set, label_to_id)
  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN,
      splits.MERGED_TRAIN_AND_DEV)
  per_split_gen['test'] = gen(test_set, label_to_id)

  return metadata, per_split_gen


ddsm_dataset = types.DownloadableDataset(
    name='ddsm',
    download_urls=[
        types.KaggleDataset(
            dataset_name='awsaf49/cbis-ddsm-breast-cancer-image-dataset',
            checksum='eba16e95a30193fcbda1d2668d96015f')
    ],
    website_url='https://www.kaggle.com/awsaf49/cbis-ddsm-breast-cancer-image-dataset',
    handler=ddsm_handler)
