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

r"""Defines the LSCL main stream.

To check that this stream is working as intended, a binary is provided at
iterate_lscl_stream.py in this directory. Running this binary will iterate
the stream and print the first example in every (successfully fetched) dataset.
"""

import collections
from concurrent import futures
import enum
from typing import Iterator, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

from absl import logging
from dm_nevis.benchmarker.datasets import datasets
from dm_nevis.benchmarker.datasets import streams
from dm_nevis.benchmarker.datasets.builders import coil100
from dm_nevis.benchmarker.datasets.builders import domainnet
from dm_nevis.benchmarker.datasets.builders import smallnorb
from dm_nevis.benchmarker.datasets.builders import tfds_builder
from dm_nevis.datasets_storage import dataset_loader
from dm_nevis.streams import nevis as nevis_dataset_loader
import numpy as np
import tensorflow_datasets as tfds

from dm_nevis.datasets_storage import paths
LSCL_DATA_DIR = paths.LSCL_DATA_DIR

# Years up to (but not including) 2020 are used for development and
# hyperparameter selecton. Only at the very end we run on the ramining years.
DEFAULT_LSCL_STOP_YEAR = 2020

DEFAULT_NUM_THREADPOOL_WORKERS = 10


class LSCLStreamVariant(enum.Enum):
  """Known task streams."""
  FULL = 'FULL'
  SHORT = 'SHORT'
  TINY = 'TINY'
  DEBUG = 'DEBUG'
  MULTILABEL_ONLY = 'MULTILABEL_ONLY'
  FINETUNING_DEBUG = 'FINETUNING_DEBUG'
  IMAGENET_ONLY = 'IMAGENET_ONLY'
  MAJOR_DOMAIN_ONLY = 'MAJOR_DOMAIN_ONLY'
  LARGE_DATASET_ONLY = 'LARGE_DATASET_ONLY'


class AblationStreamVariant(enum.Enum):
  """Known cripple streams."""
  REMOVE_FIRST_30_TASKS = 'REMOVE_FIRST_30_TASKS'
  REMOVE_LAST_30_TASKS = 'REMOVE_LAST_30_TASKS'
  NO_IMAGENET = 'NO_IMAGENET'
  RANDOM_DATASET = 'RANDOM_DATASET'


class Split(enum.Enum):
  """Data split names."""
  TRAIN = 'train'
  DEV = 'dev'
  DEV_TEST = 'dev_test'
  TEST = 'test'


LSCL_STREAM_PER_YEAR: Mapping[LSCLStreamVariant, Mapping[int, Sequence[str]]] = {
    LSCLStreamVariant.DEBUG: {
        1999: ('Pascal 2007',),
        2000: ('COIL 20',),
    },
    LSCLStreamVariant.FINETUNING_DEBUG: {
        1999: ('CIFAR 10',
              ),
        2000: ('CIFAR 10',
              ),
    },
    LSCLStreamVariant.FULL: {
        1989: (),
        1992:
            ('Magellan Venus Volcanoes', 'Aberdeen face database.'),
        1998: ('LandSat UCI repo', 'Brodatz', 'Olivetti Face Dataset'
              ),
        2000: ('COIL 20',),
        2001: ('COIL 100', 'MPEG-7'
              ),
        2002: (),
        2003: (),
        2004:
            ('Butterfly dataset', 'MNIST'),
        2005: ('Caltech 101', 'UMIST', 'CMU AMP expression'),
        2006: ('Pascal 2005', 'Caltech cars, motorbikes', 'UIUC cars', 'ALOI'),
        2007: ('8 Scene Dataset',),
        2008: ('15 Scenes',),
        2009: ('Pascal 2006', 'Extended YaleB'
              ),
        2010: ('Pascal 2007', 'Graz-02', 'Olivetti Face Dataset', 'PPMI'),
        2011:
            ('Caltech 256', 'ImageNet', 'LFW', 'Oxford Flowers',
             'Flicker Material Dataset', 'Oxford Flowers 102',
             'Belgium Traffic Sign Dataset',
             'German Traffic Sign Recognition Benchmark', 'Brodatz', 'VisTex'),
        2012: ('UMD', 'KTH-TIPS', 'UIUC texture', 'KTH-TIPS2-b', 'CVC-MUSCIMA'),
        2013: ('IAPRTC-12', 'sketch dataset', 'KTH-TIPS2-a', 'Pascal 2012',
               'NORB'),
        2014: ('Wikipaintings',),
        2015:
            ('MNIST', 'MIT Scenes', 'SUN 397', 'CIFAR 10'),
        2016: ('CUB 200', 'Stanford Cars', 'FGVC Aircraft', 'DTD', 'MS COCO',
               'Caltech 101', 'Oxford IIIT Pets', 'Stanford Dogs', 'ANIMAL',
               'Caltech 101 Silhouettes', 'Interact', 'VOC Actions', 'SVHN',
               'Chars74K'),
        2017: ('CelebA', 'LFWA', 'SUN Attribute', 'CIFAR 100'
              ),
        2018: ('TID2008', 'TID2013', 'USPS', 'Semeion', 'MNIST-m',
               'Office Caltech', 'PACS', 'Caltech Camera Traps',
               'EMNIST Balanced', 'CASIA-HWDB1.1',
               'ISBI-ISIC 2017 melanoma classification challenge'),
        2019:
            ('Trancos', 'Mall dataset', 'Fashion MNIST', 'NotMNIST',
             'Tiny Imagenet', 'STL10', 'Food 101', 'MNIST-rot', 'AWA2',
             '15 Scenes', 'Food 101 N', 'COIL 20', 'COIL 100'),
        2020: ('ShanghaiTech', 'AnimalWeb', 'BIWI'
              ),
        2021: ('ImageNet', 'Oxford Flowers 102', 'DomainNet-Real',
               'Pneumonia Chest X-ray', 'Path MNIST', 'NIH Chest X-ray',
               'Tubercolosis',
               'covid-19 x-ray', 'PatchCamelyon', 'DDSM',
               'Synthetic COVID-19 Chest X-ray Dataset'),
    },
    LSCLStreamVariant.SHORT: {
        2004: ('COIL 100', 'MNIST'
              ),
        2006: ('Pascal 2005', 'Caltech cars, motorbikes', 'UIUC cars'),
        2009: ('Pascal 2006',),
        2010: ('Caltech 101',),
        2011: (
            'Graz-02',
            '15 Scenes',
            'Pascal 2007',
            'LFW',
        ),
        2013: ('sketch dataset', 'Brodatz'),
        2014: ('ImageNet', 'Pascal 2012', 'Caltech 256'
              ),
        2018: (
            'CIFAR 100',
            'CIFAR 10',
            'USPS',
            'MNIST',
            'MNIST-m',
            'Office Caltech',
            'PACS',
            'ISBI-ISIC 2017 melanoma classification challenge',
        ),
        2019: ('Fashion MNIST',),
        2020: ('Stanford Dogs', 'CUB 200', 'Stanford Cars', 'FGVC Aircraft'),
    },
    LSCLStreamVariant.TINY: {
        2004: ('MNIST',),
        2010: ('Caltech 101',),
        2014: ('Pascal 2012',),
        2018: (
            'CIFAR 100',
            'MNIST',
            'ISBI-ISIC 2017 melanoma classification challenge',
        ),
        2020: ('CUB 200',),
    },
    LSCLStreamVariant.IMAGENET_ONLY: {
        2014: ('ImageNet',
              ),
    },
    LSCLStreamVariant.MULTILABEL_ONLY: {
        2010: ('Pascal 2007',),
        2013: ('IAPRTC-12',),
        2015: ('SUN 397',),
        2016: ('VOC Actions',),
        2017: ('SUN Attribute',),
        2019: ('AWA2',
              ),
        2021: ('NIH Chest X-ray',),
    },
    LSCLStreamVariant.MAJOR_DOMAIN_ONLY: {
        # Exclude satellite, face, texture, shape, ocr, quality, medical
        1989: (),
        1992: (),
        1998: (),
        2000: ('COIL 20',),
        2001: ('COIL 100',),
        2002: (),
        2003: (),
        2004: ('Butterfly dataset',),
        2005: ('Caltech 101',),
        2006: ('Pascal 2005', 'Caltech cars, motorbikes', 'UIUC cars', 'ALOI'),
        2007: ('8 Scene Dataset',),
        2008: ('15 Scenes',),
        2009: ('Pascal 2006',),
        2010: ('Pascal 2007', 'Graz-02', 'PPMI'),
        2011: ('Caltech 256', 'ImageNet', 'Oxford Flowers',
               'Oxford Flowers 102', 'Belgium Traffic Sign Dataset',
               'German Traffic Sign Recognition Benchmark'),
        2012: (),
        2013: ('IAPRTC-12', 'sketch dataset', 'Pascal 2012', 'NORB'),
        2014: ('Wikipaintings',),
        2015: ('MIT Scenes', 'SUN 397', 'CIFAR 10'),
        2016: ('CUB 200', 'Stanford Cars', 'FGVC Aircraft', 'MS COCO',
               'Caltech 101', 'Oxford IIIT Pets', 'Stanford Dogs', 'Interact',
               'VOC Actions', 'SVHN'),
        2017: ('SUN Attribute', 'CIFAR 100'),
        2018: ('Office Caltech', 'PACS', 'Caltech Camera Traps'),
        2019: ('Trancos', 'Mall dataset', 'Fashion MNIST', 'NotMNIST',
               'Tiny Imagenet', 'STL10', 'Food 101', 'MNIST-rot', 'AWA2',
               '15 Scenes', 'Food 101 N', 'COIL 20', 'COIL 100'),
        2020: ('ShanghaiTech', 'AnimalWeb', 'BIWI'),
        2021: ('ImageNet', 'Oxford Flowers 102', 'DomainNet-Real',
               'Pneumonia Chest X-ray', 'Path MNIST', 'NIH Chest X-ray',
               'Tubercolosis', 'covid-19 x-ray', 'PatchCamelyon', 'DDSM',
               'Synthetic COVID-19 Chest X-ray Dataset'),
    },
    LSCLStreamVariant.LARGE_DATASET_ONLY: {
        # Exclude datasets with less than 10_000 samples,
        # all splits combined.
        1989: (),
        1992: (),
        1998: (),
        2000: (),
        2001: (),
        2002: (),
        2003: (),
        2004: ('MNIST',),
        2005: ('Caltech 101',),
        2006: ('ALOI',),
        2007: (),
        2008: (),
        2009: ('Extended YaleB',),
        2010: ('Pascal 2007',),
        2011: ('Caltech 256', 'ImageNet', 'LFW', 'Belgium Traffic Sign Dataset',
               'German Traffic Sign Recognition Benchmark'),
        2012: (),
        2013: ('IAPRTC-12', 'sketch dataset', 'Pascal 2012', 'NORB'),
        2014: ('Wikipaintings',),
        2015: ('MNIST', 'MIT Scenes', 'SUN 397', 'CIFAR 10'),
        2016: ('CUB 200', 'Stanford Cars', 'FGVC Aircraft', 'MS COCO',
               'Caltech 101', 'Oxford IIIT Pets', 'Stanford Dogs',
               'Caltech 101 Silhouettes', 'VOC Actions', 'SVHN', 'Chars74K'),
        2017: ('CelebA', 'LFWA', 'SUN Attribute', 'CIFAR 100'),
        2018: ('USPS', 'MNIST-m', 'PACS', 'Caltech Camera Traps',
               'EMNIST Balanced', 'CASIA-HWDB1.1'),
        2019: ('Trancos', 'Mall dataset', 'Fashion MNIST', 'NotMNIST',
               'Tiny Imagenet', 'STL10', 'Food 101', 'MNIST-rot', 'AWA2',
               '15 Scenes', 'Food 101 N', 'COIL 20', 'COIL 100'),
        2020: ('ShanghaiTech', 'AnimalWeb', 'BIWI'),
        2021: ('ImageNet', 'Oxford Flowers 102', 'DomainNet-Real',
               'Pneumonia Chest X-ray', 'Path MNIST', 'NIH Chest X-ray',
               'Tubercolosis', 'covid-19 x-ray', 'PatchCamelyon', 'DDSM',
               'Synthetic COVID-19 Chest X-ray Dataset')
    },
}


# List of datasets for parallel runs. These should be automatically derived from
# the LSCLStream; unfortunately this requires reverse-mapping names and opening
# the stream. To keep things simple we use a static list for now; should be
# changed when we refactor for the OSS release.

# All datasets in the SHORT stream, including held-out years
PARALLEL_DATASETS_SHORT = [
    'COIL 100', 'MNIST', 'Pascal 2005', 'Caltech cars, motorbikes', 'UIUC cars',
    'Pascal 2006', 'Caltech 101', 'Graz-02', '15 Scenes', 'Pascal 2007', 'LFW',
    'sketch dataset', 'Brodatz', 'ImageNet', 'Pascal 2012', 'Caltech 256',
    'CIFAR 100', 'CIFAR 10', 'USPS', 'MNIST', 'MNIST-m', 'Office Caltech',
    'PACS', 'ISBI-ISIC 2017 melanoma classification challenge', 'Fashion MNIST',
    'Stanford Dogs', 'CUB 200', 'Stanford Cars', 'FGVC Aircraft'
]

# All datasets in the FULL stream, including held-out years
PARALLEL_DATASETS = [
    'Magellan Venus Volcanoes', 'Aberdeen face database.', 'LandSat UCI repo',
    'Brodatz', 'Olivetti Face Dataset', 'COIL 20', 'COIL 100', 'MPEG-7',
    'Butterfly dataset', 'MNIST', 'Caltech 101', 'UMIST', 'CMU AMP expression',
    'Pascal 2005', 'Caltech cars, motorbikes', 'UIUC cars', 'ALOI',
    '8 Scene Dataset', '15 Scenes', 'Pascal 2006', 'Extended YaleB',
    'Pascal 2007', 'Graz-02', 'Olivetti Face Dataset', 'PPMI', 'Caltech 256',
    'ImageNet', 'LFW', 'Oxford Flowers', 'Flicker Material Dataset',
    'Oxford Flowers 102', 'Belgium Traffic Sign Dataset',
    'German Traffic Sign Recognition Benchmark', 'Brodatz', 'VisTex', 'UMD',
    'KTH-TIPS', 'UIUC texture', 'KTH-TIPS2-b', 'CVC-MUSCIMA', 'IAPRTC-12',
    'sketch dataset', 'KTH-TIPS2-a', 'Pascal 2012', 'NORB', 'Wikipaintings',
    'MNIST', 'MIT Scenes', 'SUN 397', 'CIFAR 10', 'CUB 200', 'Stanford Cars',
    'FGVC Aircraft', 'DTD', 'MS COCO', 'Caltech 101', 'Oxford IIIT Pets',
    'Stanford Dogs', 'ANIMAL', 'Caltech 101 Silhouettes', 'Interact',
    'VOC Actions', 'SVHN', 'Chars74K', 'CelebA', 'LFWA', 'SUN Attribute',
    'CIFAR 100', 'TID2008', 'TID2013', 'USPS', 'Semeion', 'MNIST-m',
    'Office Caltech', 'PACS', 'Caltech Camera Traps', 'EMNIST Balanced',
    'CASIA-HWDB1.1', 'ISBI-ISIC 2017 melanoma classification challenge',
    'Trancos', 'Mall dataset', 'Fashion MNIST', 'NotMNIST', 'Tiny Imagenet',
    'STL10', 'Food 101', 'MNIST-rot', 'AWA2', '15 Scenes', 'Food 101 N',
    'COIL 20', 'COIL 100', 'ShanghaiTech', 'AnimalWeb', 'BIWI', 'ImageNet',
    'Oxford Flowers 102', 'DomainNet-Real', 'Pneumonia Chest X-ray',
    'Path MNIST', 'NIH Chest X-ray', 'Tubercolosis', 'covid-19 x-ray',
    'PatchCamelyon', 'DDSM', 'Synthetic COVID-19 Chest X-ray Dataset'
]

DEFAULT_THREADPOOL_WORKERS = 30


class NevisSource(NamedTuple):
  """Represents a dataset implemented in nevis."""
  name: str


class TFDSSource(NamedTuple):
  """Represents a dataset implemented in tfds."""
  name: str


class KeyAndDataset(NamedTuple):
  key: streams.DatasetKey
  dataset: datasets.Dataset


class DatasetSplits(NamedTuple):
  train: KeyAndDataset
  dev: KeyAndDataset
  train_and_dev: KeyAndDataset
  dev_test: KeyAndDataset
  test: KeyAndDataset


# pylint: disable=line-too-long
# pyformat: disable
DATASET_NAME_TO_SOURCE = {
    '15 Scenes': NevisSource('scenes15'),
    '8 Scene Dataset': NevisSource('scenes8'),
    'Aberdeen face database.': NevisSource('aberdeen'),
    'ALOI': NevisSource('aloi'),
    'ANIMAL': NevisSource('animal'),
    'AnimalWeb': NevisSource('animal_web'),
    'AWA2': NevisSource('awa2'),
    'Belgium Traffic Sign Dataset': NevisSource('belgium_tsc'),
    'BIWI': NevisSource('biwi'),
    'Brodatz': NevisSource('brodatz'),
    'Butterfly dataset': NevisSource('butterflies'),
    'Caltech 101 Silhouettes': NevisSource('silhouettes_28'),
    'Caltech 101': TFDSSource('caltech101'),
    'Caltech 256': NevisSource('caltech256'),
    'Caltech Camera Traps': NevisSource('caltech_camera_traps'),
    'Caltech cars, motorbikes': NevisSource('caltech_categories'),
    'CASIA-HWDB1.1': NevisSource('casia_hwdb'),
    'CelebA': TFDSSource('celeb_a'),
    'Chars74K': NevisSource('chars74k'),
    'CIFAR 10': TFDSSource('cifar10'),
    'CIFAR 100': TFDSSource('cifar100'),
    'CMU AMP expression': NevisSource('cmu_amp_expression'),
    'COIL 100': TFDSSource('coil100'),
    'COIL 20': NevisSource('coil20'),
    'covid-19 x-ray': NevisSource('covid_19_xray'),
    'CUB 200': TFDSSource('caltech_birds2011'),
    'CVC-MUSCIMA': NevisSource('cvc_muscima'),
    'DDSM': NevisSource('ddsm'),
    'DomainNet-Real': TFDSSource('domainnet'),
    'DTD': TFDSSource('dtd'),
    'EMNIST Balanced': TFDSSource('emnist/balanced'),
    'Extended YaleB': NevisSource('extended_yaleb'),
    'Fashion MNIST': TFDSSource('fashion_mnist'),
    'FGVC Aircraft': NevisSource('fgvc_aircraft_family'),
    'Flicker Material Dataset': NevisSource('flickr_material_database'),
    'Food 101 N': NevisSource('food101n'),
    'Food 101': NevisSource('food101'),
    'German Traffic Sign Recognition Benchmark': NevisSource('german_tsr'),
    'Graz-02': NevisSource('ig02'),
    'IAPRTC-12': NevisSource('iaprtc12'),
    'ImageNet': TFDSSource('imagenet2012'),
    'Interact': NevisSource('interact'),
    'ISBI-ISIC 2017 melanoma classification challenge': NevisSource('melanoma'),
    'KTH-TIPS': NevisSource('kth_tips'),
    'KTH-TIPS2-a': NevisSource('kth_tips_2a'),
    'KTH-TIPS2-b': NevisSource('kth_tips_2b'),
    'LandSat UCI repo': NevisSource('landsat'),
    'LFW': NevisSource('lfw'),
    'LFWA': NevisSource('lfwa'),
    'Magellan Venus Volcanoes': NevisSource('magellan_venus_volcanoes'),
    'Mall dataset': NevisSource('mall'),
    'MIT Scenes': NevisSource('mit_scenes'),
    'MNIST-m': NevisSource('mnist_m'),
    'MNIST-rot': NevisSource('mnist_rotation'),
    'MNIST': TFDSSource('mnist'),
    'MPEG-7': NevisSource('mpeg7'),
    'MS COCO': NevisSource('coco_single_label'),
    'NIH Chest X-ray': NevisSource('nih_chest_xray'),
    'NORB': TFDSSource('smallnorb'),
    'NotMNIST': NevisSource('not_mnist'),
    'Office 31 amazon': NevisSource('office31_amazon'),
    'Office 31 dslr': NevisSource('office31_dslr'),
    'Office 31 webcam': NevisSource('office31_webcam'),
    'Office Caltech': NevisSource('office_caltech_10'),
    'Olivetti Face Dataset': NevisSource('olivetti_face'),
    'Oxford Flowers 102': TFDSSource('oxford_flowers102'),
    'Oxford Flowers': NevisSource('oxford_flowers_17'),
    'Oxford IIIT Pets': TFDSSource('oxford_iiit_pet'),
    'PACS': NevisSource('pacs'),
    'Pascal 2005': NevisSource('pascal_voc2005'),
    'Pascal 2006': NevisSource('pascal_voc2006'),
    'Pascal 2007': TFDSSource('voc/2007'),
    'Pascal 2012': TFDSSource('voc/2012'),
    'PatchCamelyon': TFDSSource('patch_camelyon'),
    'Path MNIST': NevisSource('path_mnist'),
    'Pneumonia Chest X-ray': NevisSource('pneumonia_chest_xray'),
    'PPMI': NevisSource('ppmi'),
    'Semeion': NevisSource('semeion'),
    'ShanghaiTech': NevisSource('shanghai_tech'),
    'sketch dataset': NevisSource('sketch'),
    'Stanford Cars': NevisSource('stanford_cars'),
    'Stanford Dogs': TFDSSource('stanford_dogs'),
    'STL10': TFDSSource('stl10'),
    'SUN 397': TFDSSource('sun397'),
    'SUN Attribute': NevisSource('sun_attributes'),
    'SVHN': TFDSSource('svhn_cropped'),
    'Synthetic COVID-19 Chest X-ray Dataset': NevisSource('synthetic_covid19_xray'),
    'TID2008': NevisSource('tid2008'),
    'TID2013': NevisSource('tid2013'),
    'Tiny Imagenet': NevisSource('tiny_imagenet'),
    'Trancos': NevisSource('trancos'),
    'Tubercolosis': NevisSource('tubercolosis'),
    'UIUC cars': NevisSource('uiuc_cars'),
    'UIUC texture': NevisSource('uiuc_texture'),
    'UMD': NevisSource('umd'),
    'UMIST': NevisSource('umist'),
    'USPS': NevisSource('usps'),
    'VisTex': NevisSource('vistex'),
    'VOC Actions': NevisSource('voc_actions'),
    'Wikipaintings': NevisSource('wiki_paintings_style'),
}
# pyformat: enable
# pylint: enable=line-too-long


class LSCLStream:
  """The LSCL benchmark stream.

  The stream adds a train event for each instance of the train data in the
  stream.

  Additionally, a predict event is added containing the test dataset after
  every instance of a train dataset.

  Once the stream is complete, a further predict event is added for every
  seen train event. This makes it possible to compare the performance on tasks
  from train time to the end of the stream.
  """

  def __init__(
      self,
      stream_variant: LSCLStreamVariant = LSCLStreamVariant.FULL,
      stop_year: int = DEFAULT_LSCL_STOP_YEAR,
      *,
      predict_event_splits: Sequence[Split] = (Split.DEV_TEST,),
      shuffle_seed: int = 1,
      shuffle_within_year: bool = False,
      shuffle_datasets_order: bool = False,
  ):
    """Instantiates a LSCL task stream.

    Args:
      stream_variant: Which of the streams to use (see `LSCLStreamVariant`).
      stop_year: The stream will only include tasks before the given year.
      predict_event_splits: Sequence of splits to use for prediction.
      shuffle_seed: An integer denoting a seed for shuffling logic when
        `shuffle_within_year` or `shuffle_datasets_order` are ative.
      shuffle_within_year: Whether to shuffle the order of datasets within a
        year.
      shuffle_datasets_order: Whether to shuffle the order of datasets randomly
        across years.
    """
    logging.info('Reading LSCL stream from LSCL_STREAM_PER_YEAR.')
    self._events, self._datasets_by_key = _get_events_and_lookup(
        LSCL_STREAM_PER_YEAR[stream_variant],
        stop_year,
        predict_event_splits=predict_event_splits,
        shuffle_seed=shuffle_seed,
        shuffle_within_year=shuffle_within_year,
        shuffle_datasets_order=shuffle_datasets_order)

  def get_dataset_by_key(self,
                         dataset_key: streams.DatasetKey) -> datasets.Dataset:

    return self._datasets_by_key[dataset_key]

  def events(self) -> Iterator[streams.Event]:
    return iter(self._events)


class IndividualDatasetStream:
  """A train and predict event for an individual, or a pair of datasets."""

  def __init__(
      self,
      dataset_name: str,
      second_dataset_name: Optional[str] = None,
      predict_event_splits: Sequence[Split] = (Split.DEV_TEST,),
  ):
    """A stream with a train and predict event for an individual dataset.

    Args:
      dataset_name: One of the dataset names from `DATASET_NAME_TO_SOURCE`.
      second_dataset_name: Optional second dataset in the stream. If
        it is either None or equal to the first dataset, it will not be added.
      predict_event_splits: Sequence of splits to use for prediction.
    """
    dataset_split = _get_splits_for_dataset_name(dataset_name)

    if dataset_split is None:
      logging.warning('Skipping `%s`', dataset_name)
      self._events = []
      self._datasets_by_key = {}
    else:
      self._events = [
          streams.TrainingEvent(
              train_dataset_key=dataset_split.train.key,
              dev_dataset_key=dataset_split.dev.key,
              train_and_dev_dataset_key=dataset_split.train_and_dev.key),
          ]
      for split in predict_event_splits:
        self._events.append(
            streams.PredictionEvent(split_to_key(split, dataset_split)))

      self._datasets_by_key = {
          dataset_split.train.key: dataset_split.train.dataset,
          dataset_split.dev.key: dataset_split.dev.dataset,
          dataset_split.train_and_dev.key: dataset_split.train_and_dev.dataset,
          dataset_split.test.key: dataset_split.test.dataset,
          dataset_split.dev_test.key: dataset_split.dev_test.dataset,
      }

    if second_dataset_name is None:
      return

    dataset_split = _get_splits_for_dataset_name(second_dataset_name)
    if dataset_split is None:
      raise ValueError(
          'Could not find second dataset `%s`' % second_dataset_name
      )
    else:
      self._events.append(
          streams.TrainingEvent(
              train_dataset_key=dataset_split.train.key,
              dev_dataset_key=dataset_split.dev.key,
              train_and_dev_dataset_key=dataset_split.train_and_dev.key))
      for split in predict_event_splits:
        self._events.append(
            streams.PredictionEvent(split_to_key(split, dataset_split)))

      self._datasets_by_key.update({
          dataset_split.train.key: dataset_split.train.dataset,
          dataset_split.dev.key: dataset_split.dev.dataset,
          dataset_split.train_and_dev.key: dataset_split.train_and_dev.dataset,
          dataset_split.test.key: dataset_split.test.dataset,
          dataset_split.dev_test.key: dataset_split.dev_test.dataset,
      })

  def get_dataset_by_key(self,
                         dataset_key: streams.DatasetKey) -> datasets.Dataset:
    return self._datasets_by_key[dataset_key]

  def events(self) -> Iterator[streams.Event]:
    return iter(self._events)


class AblationStream:
  """The LSCL benchmark ablation stream."""

  def __init__(self,
               stream_variant: AblationStreamVariant,
               meta_train_stop_year: int = DEFAULT_LSCL_STOP_YEAR,
               stop_year: int = DEFAULT_LSCL_STOP_YEAR + 2,
               *,
               predict_event_splits: Sequence[Split] = (Split.DEV_TEST,),
               **kwargs):
    """Instantiates a LSCL ablation stream."""
    logging.info('Reading LSCL ablation stream.')
    assert stop_year > meta_train_stop_year, ('Full stream stop year needs to '
                                              'be larger than meta_train stop '
                                              'year')
    self._meta_train_stop_year = meta_train_stop_year
    self._stop_year = stop_year

    datasets_by_year = LSCL_STREAM_PER_YEAR[LSCLStreamVariant.FULL]

    if stream_variant is AblationStreamVariant.REMOVE_FIRST_30_TASKS:
      filtered_datasets_by_year = self._remove_k_datasets_from_stream(
          datasets_by_year=datasets_by_year, k=30)
    elif stream_variant is AblationStreamVariant.REMOVE_LAST_30_TASKS:
      filtered_datasets_by_year = self._remove_k_datasets_from_stream(
          datasets_by_year=datasets_by_year, k=30, reverse=True)
    elif stream_variant is AblationStreamVariant.NO_IMAGENET:
      filtered_datasets_by_year = remove_imagenet_from_stream(
          datasets_by_year=datasets_by_year,
          stop_year=self._meta_train_stop_year)
    elif stream_variant is AblationStreamVariant.RANDOM_DATASET:
      assert 'num_random_datasets' in kwargs, ('num_random_dataset needed for '
                                               'defining random dataset stream')
      num_random_datasets = kwargs['num_random_datasets']
      random_seed = kwargs.get('random_seed', 0)
      filtered_datasets_by_year = self._get_random_dataset_from_stream(
          datasets_by_year=datasets_by_year,
          num_random_datasets=num_random_datasets,
          random_seed=random_seed)
    else:
      raise ValueError('Ablation stream variant not defined')

    self._events, self._datasets_by_key = _get_events_and_lookup(
        filtered_datasets_by_year,
        stop_year,
        predict_event_splits=predict_event_splits)

  def get_dataset_by_key(self,
                         dataset_key: streams.DatasetKey) -> datasets.Dataset:

    return self._datasets_by_key[dataset_key]

  def events(self) -> Iterator[streams.Event]:
    return iter(self._events)

  def _get_random_dataset_from_stream(
      self, datasets_by_year: Mapping[int,
                                      Sequence[str]], num_random_datasets: int,
      random_seed: int) -> Mapping[int, Sequence[str]]:
    """Randomly picks datasets from a stream."""
    rng = np.random.default_rng(random_seed)
    train_stream_tasks = []
    test_stream_tasks = []
    for year, dataset_names_by_year in datasets_by_year.items():
      if year >= self._meta_train_stop_year:
        test_stream_tasks += [
            (year, dataset_name) for dataset_name in dataset_names_by_year
        ]
      else:
        train_stream_tasks += [
            (year, dataset_name) for dataset_name in dataset_names_by_year
        ]
    assert num_random_datasets > len(
        test_stream_tasks), 'Need at least one dataset for train stream.'
    # Only shuffle tasks in the traininig stream
    rng.shuffle(train_stream_tasks)
    random_stream_tasks = test_stream_tasks + train_stream_tasks
    random_stream_tasks = random_stream_tasks[:num_random_datasets]

    result_datasets_by_year = collections.defaultdict(list)

    for year, dataset_names_by_year in datasets_by_year.items():
      # Retain datasets in the random stream and follow the within-year order.
      for dataset_name in dataset_names_by_year:
        if (year, dataset_name) in random_stream_tasks:
          result_datasets_by_year[year].append(dataset_name)

    filtered_datasets_by_year = {}
    for year, dataset_names_by_year in result_datasets_by_year.items():
      filtered_datasets_by_year[year] = tuple(
          dataset for dataset in dataset_names_by_year)
    return filtered_datasets_by_year

  def _remove_k_datasets_from_stream(
      self,
      datasets_by_year: Mapping[int,
                                Sequence[str]],
      k: int,
      reverse=False) -> Mapping[int, Sequence[str]]:
    """Removes k tasks from stream.

    Args:
      datasets_by_year: A stream of datasets by year.
      k: number of tasks to remove from the stream.
      reverse: If reverse=False, remove the first k datasets from stream. remove
        the last k datasets if reverse is set to True.

    Returns:
      A stream of datasets with k datasets removed.
    """
    filtered_datasets_by_year = {}
    dataset_index = 0
    for year, dataset_names_by_year in sorted(
        datasets_by_year.items(), reverse=reverse):
      if year >= self._meta_train_stop_year or dataset_index >= k:
        filtered_datasets_by_year[year] = dataset_names_by_year
      else:
        num_skipped_dataset = min(
            len(dataset_names_by_year), k - dataset_index)
        task_list = dataset_names_by_year[num_skipped_dataset:]
        dataset_index += len(dataset_names_by_year)
        if task_list:
          filtered_datasets_by_year[year] = task_list
    return filtered_datasets_by_year


def remove_imagenet_from_stream(
    datasets_by_year: Mapping[int, Sequence[str]],
    stop_year: int = DEFAULT_LSCL_STOP_YEAR) -> Mapping[int, Sequence[str]]:
  """Removes ImageNet from stream."""
  filtered_datasets_by_year = {}
  for year, dataset_names_by_year in datasets_by_year.items():
    if year >= stop_year:
      filtered_datasets_by_year[year] = dataset_names_by_year
    else:
      filtered_datasets_by_year[year] = tuple(
          dataset_name for dataset_name in dataset_names_by_year
          if dataset_name != 'ImageNet')
  return filtered_datasets_by_year


def datasets_in_stream(
    stream_variant: LSCLStreamVariant = LSCLStreamVariant.FULL,
    stop_year: int = DEFAULT_LSCL_STOP_YEAR,
    remove_duplicates: bool = True,
    check_availability: bool = False,
) -> Sequence[str]:
  """Returns the list of datasets in the stream.

  Args:
    stream_variant: Which of the streams to use (see `LSCLStreamVariant`).
    stop_year: Only include datasets before the given year.
    remove_duplicates: Remove duplicate datasets or not.
    check_availability: Only include datasets that are available.

  Returns:
    A list or dataset names.
  """
  dataset_names = []
  for year, datasets_in_year in LSCL_STREAM_PER_YEAR[stream_variant].items():
    if year >= stop_year:
      break
    dataset_names.extend(datasets_in_year)

  if remove_duplicates:
    # Remove duplicates while preserving order
    dataset_names = list(dict.fromkeys(dataset_names))

  if check_availability:
    # Filter out datasets for which we can't load split information.
    with futures.ThreadPoolExecutor(max_workers=10) as executor:
      # pylint: disable=g-long-lambda
      dataset_names = executor.map(
          lambda name: name
          if _get_splits_for_dataset_name(name) else None, dataset_names)
      dataset_names = list(filter(None, dataset_names))

  return dataset_names


def _filter_to_datasets_that_are_available(
    dataset_names: List[str]) -> List[str]:
  """Returns datasets with unavailable datasets filtered."""

  def dataset_name_or_none(dataset_name):
    if _get_splits_for_dataset_name(dataset_name,) is not None:
      return dataset_name
    return None

  # This operation is slow, so we use a thread pool to parallelize the IO.
  with futures.ThreadPoolExecutor(
      max_workers=DEFAULT_NUM_THREADPOOL_WORKERS) as pool:

    dataset_names = pool.map(dataset_name_or_none, dataset_names)

  return [name for name in dataset_names if name is not None]


def split_to_key(split: Split,
                 dataset_split: DatasetSplits) -> streams.DatasetKey:
  if split is Split.DEV:
    return dataset_split.dev.key
  elif split is Split.DEV_TEST:
    return dataset_split.dev_test.key
  elif split is Split.TEST:
    return dataset_split.test.key
  else:
    raise ValueError(f'Unsupported split: {split}')


def _get_events_and_lookup(
    datasets_by_year: Mapping[int, Sequence[str]],
    stop_year: int,
    *,
    predict_event_splits: Sequence[Split] = (Split.DEV_TEST,),
    shuffle_seed: int = 1,
    shuffle_within_year: bool = False,
    shuffle_datasets_order: bool = False,
) -> Tuple[Sequence[streams.Event], Mapping[streams.DatasetKey,
                                            datasets.Dataset]]:
  """Constructs a sequence of events and a dataset lookup."""
  events = []
  lookup = {}
  datasets_by_key = {}

  dataset_names = set()
  for dataset_names_by_year in datasets_by_year.values():
    dataset_names.update(dataset_names_by_year)

  lookup = _build_lookup(sorted(dataset_names))

  rng = np.random.default_rng(shuffle_seed)

  iterable_datasets_by_year = sorted(datasets_by_year.items())

  if shuffle_datasets_order:
    rng.shuffle(iterable_datasets_by_year)

  for year, dataset_names in iterable_datasets_by_year:
    if year >= stop_year:
      break

    if shuffle_within_year:
      dataset_names = list(dataset_names)
      rng.shuffle(dataset_names)

    for dataset_name in dataset_names:
      result = lookup[dataset_name]
      if result is None:
        logging.warning('Skipping for %d: `%s`', year, dataset_name)
        continue

      train_event = streams.TrainingEvent(
          train_dataset_key=result.train.key,
          dev_dataset_key=result.dev.key,
          train_and_dev_dataset_key=result.train_and_dev.key)
      events.append(train_event)

      for split in predict_event_splits:
        events.append(streams.PredictionEvent(split_to_key(split, result)))

      datasets_by_key[result.train.key] = result.train.dataset
      datasets_by_key[result.test.key] = result.test.dataset
      datasets_by_key[result.dev.key] = result.dev.dataset
      datasets_by_key[result.dev_test.key] = result.dev_test.dataset
      datasets_by_key[result.train_and_dev.key] = result.train_and_dev.dataset

  total_available_datasets = sum(1 for x in lookup.values() if x is not None)
  logging.info('Total available datasets: %d/%d', total_available_datasets,
               len(lookup.keys()))

  return events, datasets_by_key


def _get_splits_for_dataset_name(dataset_name: str) -> Optional[DatasetSplits]:
  """Gets train and test datasets for a dataset by name."""

  if dataset_name not in DATASET_NAME_TO_SOURCE:
    raise ValueError(f'Unknown source for dataset named: `{dataset_name}`')

  source = DATASET_NAME_TO_SOURCE.get(dataset_name)
  if source is None:
    logging.warning('Source not yet available for `%s`', dataset_name)
    return None

  try:
    result = _get_splits_for_source(source)
  except dataset_loader.DatasetNotReadyError:
    logging.warning('Dataset found but not yet ready:  %s', dataset_name)
    return None

  if result is None:
    logging.warning('Dataset found but not yet available: `%s`', dataset_name)
    return None

  return result


def _get_splits_for_source(
    source: Union[NevisSource, TFDSSource]) -> DatasetSplits:
  """Constructs the keys and datasets for the given source."""

  if isinstance(source, NevisSource):
    return _dataset_splits_for_nevis(source)

  elif isinstance(source, TFDSSource):
    return _dataset_splits_for_tfds(source)

  raise ValueError(f'Unknown source type: {type(source)}')


_TFDS_DATASETS_TRAIN_TEST_DATASETS = [
    'mnist',
    'cifar10',
    'cifar100',
    'caltech101',
    'caltech_birds2011',
    'emnist/balanced',
    'fashion_mnist',
    'oxford_iiit_pet',
    'stanford_dogs',
    'stl10',
    'svhn_cropped',
]

_TFDS_DATASETS_TRAIN_VALIDATION_TEST_DATASETS = [
    'dtd',
    'oxford_flowers102',
    'voc/2007',
    'patch_camelyon',
    'sun397',
    'celeb_a',
]


def _dataset_splits_for_tfds(source: TFDSSource) -> DatasetSplits:
  """Constructs key and dataset for tfds dataset."""

  dataset_name = source.name
  dataset_key_prefix = _canonicalize_name(dataset_name)

  test_key = f'{dataset_key_prefix}_test'
  dev_test_key = f'{dataset_key_prefix}_dev_test'
  train_key = f'{dataset_key_prefix}_train'
  train_and_dev_key = f'{dataset_key_prefix}_train_and_dev'
  dev_key = f'{dataset_key_prefix}_dev'

  dataset_info = tfds.builder(dataset_name).info
  if dataset_name in _TFDS_DATASETS_TRAIN_TEST_DATASETS:
    train_fraction = 0.7
    dev_fraction = 0.15
    num_examples = dataset_info.splits['train'].num_examples
    num_train_examples = int(num_examples * train_fraction)
    num_dev_examples = int(num_examples * dev_fraction)
    train_dataset = tfds_builder.get_dataset(
        dataset_name, split='train', end=num_train_examples)
    dev_dataset = tfds_builder.get_dataset(
        dataset_name,
        split='train',
        start=num_train_examples,
        end=num_train_examples + num_dev_examples)
    train_and_dev_dataset = tfds_builder.get_dataset(
        dataset_name, split='train', end=num_train_examples + num_dev_examples)
    dev_test_dataset = tfds_builder.get_dataset(
        dataset_name,
        split='train',
        start=num_train_examples + num_dev_examples)
    test_dataset = tfds_builder.get_dataset(dataset_name, split='test')

  elif dataset_name in _TFDS_DATASETS_TRAIN_VALIDATION_TEST_DATASETS:
    train_fraction = 0.8
    dev_fraction = 0.2
    num_examples = dataset_info.splits['train'].num_examples
    num_train_examples = int(num_examples * train_fraction)
    train_dataset = tfds_builder.get_dataset(
        dataset_name, split='train', end=num_train_examples)
    dev_dataset = tfds_builder.get_dataset(
        dataset_name, split='train', start=num_train_examples)
    train_and_dev_dataset = tfds_builder.get_dataset(
        dataset_name, split='train')

    dev_test_dataset = tfds_builder.get_dataset(
        dataset_name, split='validation')
    test_dataset = tfds_builder.get_dataset(dataset_name, split='test')
  elif dataset_name == 'coil100':
    train_dataset = coil100.get_dataset(split='train')
    dev_dataset = coil100.get_dataset(split='dev')
    dev_test_dataset = coil100.get_dataset(split='dev_test')
    test_dataset = coil100.get_dataset(split='test')
    train_and_dev_dataset = coil100.get_dataset(split='train_and_dev')
  elif dataset_name == 'domainnet':
    train_dataset = domainnet.get_dataset(split='train')
    dev_dataset = domainnet.get_dataset(split='dev')
    dev_test_dataset = domainnet.get_dataset(split='dev_test')
    test_dataset = domainnet.get_dataset(split='test')
    train_and_dev_dataset = domainnet.get_dataset(split='train_and_dev')
  elif dataset_name == 'smallnorb':
    train_dataset = smallnorb.get_dataset(split='train')
    dev_dataset = smallnorb.get_dataset(split='dev')
    dev_test_dataset = smallnorb.get_dataset(split='dev_test')
    test_dataset = smallnorb.get_dataset(split='test')
    train_and_dev_dataset = smallnorb.get_dataset(split='train_and_dev')
  elif dataset_name == 'imagenet2012':
    train_fraction = 0.9
    dev_fraction = 0.05
    num_examples = dataset_info.splits['train'].num_examples
    num_train_examples = int(num_examples * train_fraction)
    num_dev_examples = int(num_examples * dev_fraction)  # 64_058 images
    train_dataset = tfds_builder.get_dataset(
        dataset_name, split='train', end=num_train_examples)
    dev_dataset = tfds_builder.get_dataset(
        dataset_name,
        split='train',
        start=num_train_examples,
        end=num_train_examples + num_dev_examples)
    train_and_dev_dataset = tfds_builder.get_dataset(
        dataset_name, split='train', end=num_train_examples + num_dev_examples)
    dev_test_dataset = tfds_builder.get_dataset(
        dataset_name,
        split='train',
        start=num_train_examples + num_dev_examples)
    # Use provided validation split for actual testing.
    test_dataset = tfds_builder.get_dataset(dataset_name, split='validation')
  elif dataset_name == 'voc/2012':
    train_fraction = 0.8
    dev_test_fraction = 0.5
    num_examples = dataset_info.splits['train'].num_examples
    num_train_examples = int(num_examples * train_fraction)  # 4_574 images
    num_val_examples = dataset_info.splits['validation'].num_examples
    num_dev_test_examples = int(
        num_val_examples * dev_test_fraction)  # 2_911 images
    train_dataset = tfds_builder.get_dataset(
        dataset_name, split='train', end=num_train_examples)
    dev_dataset = tfds_builder.get_dataset(
        dataset_name,
        split='train',
        start=num_train_examples)
    train_and_dev_dataset = tfds_builder.get_dataset(
        dataset_name, split='train')
    dev_test_dataset = tfds_builder.get_dataset(
        dataset_name,
        split='validation',
        end=num_dev_test_examples)
    # Use provided validation split for actual testing.
    test_dataset = tfds_builder.get_dataset(
        dataset_name,
        split='validation',
        start=num_dev_test_examples)
  else:
    raise NotImplementedError(f'TFDS dataset {dataset_name} not available')

  return DatasetSplits(
      train=KeyAndDataset(train_key, train_dataset),
      dev=KeyAndDataset(dev_key, dev_dataset),
      train_and_dev=KeyAndDataset(train_and_dev_key, train_and_dev_dataset),
      dev_test=KeyAndDataset(dev_test_key, dev_test_dataset),
      test=KeyAndDataset(test_key, test_dataset),
  )


def _dataset_splits_for_nevis(source: NevisSource) -> DatasetSplits:
  """Constructs key and dataset for nevis dataset."""

  dataset_key_prefix = _canonicalize_name(source.name)

  train_key = f'{dataset_key_prefix}_train'
  test_key = f'{dataset_key_prefix}_test'
  dev_test_key = f'{dataset_key_prefix}_dev_test'
  train_and_dev_key = f'{dataset_key_prefix}_train_and_dev'
  dev_key = f'{dataset_key_prefix}_dev'

  train_dataset = nevis_dataset_loader.get_dataset(
      source.name, 'train', root_dir=LSCL_DATA_DIR)
  dev_test_dataset = nevis_dataset_loader.get_dataset(
      source.name, 'dev-test', root_dir=LSCL_DATA_DIR)
  test_dataset = nevis_dataset_loader.get_dataset(
      source.name, 'test', root_dir=LSCL_DATA_DIR)
  dev_dataset = nevis_dataset_loader.get_dataset(
      source.name, 'dev', root_dir=LSCL_DATA_DIR)
  train_and_dev_dataset = nevis_dataset_loader.get_dataset(
      source.name, 'train_and_dev', root_dir=LSCL_DATA_DIR)

  return DatasetSplits(
      train=KeyAndDataset(train_key, train_dataset),
      dev=KeyAndDataset(dev_key, dev_dataset),
      train_and_dev=KeyAndDataset(train_and_dev_key, train_and_dev_dataset),
      dev_test=KeyAndDataset(dev_test_key, dev_test_dataset),
      test=KeyAndDataset(test_key, test_dataset),
  )


def _build_lookup(
    dataset_names: Sequence[str]) -> Mapping[str, Optional[DatasetSplits]]:
  """Creates a lookup for given dataset names."""

  with futures.ThreadPoolExecutor(
      max_workers=DEFAULT_THREADPOOL_WORKERS) as executor:
    result = list(executor.map(_get_splits_for_dataset_name, dataset_names))

  return dict(zip(dataset_names, result))


def _canonicalize_name(s: str) -> str:
  """Translates special characters in datasets names to underscores."""
  return s.translate(str.maketrans(' -/~', '____'))
