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

"""Handler for Brodatz dataset."""

import functools
import os
from typing import List
from dm_nevis.datasets_storage.handlers import types
import numpy as np
from PIL import Image

from tensorflow.io import gfile


def _get_class_names(filenames: List[str]) -> List[str]:
  return sorted([os.path.splitext(f)[0] for f in filenames])


def brodatz_handler(dataset_path: str,
                    rng_seed: int = 0) -> types.HandlerOutput:
  """Imports Brodatz texture dataset.

  We import the 111 original texture images. Each of these images represent one
  class. Following the procedure in:

  A Training-free Classification Framework for Textures, Writers, and Materials-
  R. Timofte1 and L. Van Gool, 2012

  we extract from each image 9 non-overlapping regions.
  In order to ensure all the classes are presented in all the splits, we
  randomly select out of these 9 samples:
  - 5 samples for train
  - 1 sample for dev
  - 1 sample for dev-test
  - 2 samples for test
  These sets are non-overlapping by construction.


  Link: https://www.ux.uis.no/~tranden/brodatz.html

  Args:
    dataset_path: Path with downloaded datafiles.
    rng_seed: Seed for random number generator.

  Returns:
    Metadata and generator functions.
  """

  filenames = gfile.listdir(dataset_path)
  class_names = _get_class_names(filenames)

  metadata = types.DatasetMetaData(
      num_channels=1,
      num_classes=len(filenames),
      image_shape=(),  # Ignored for now.
      preprocessing='random_crop',  # select random crops in the images
      additional_metadata=dict(
          labels=class_names,
          task_type='classification',
          image_type='texture'
      ))

  def gen(rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    for idx, f in enumerate(filenames):
      splits_idx = rng.permutation(range(9))
      im = Image.open(os.path.join(dataset_path, f))
      im.load()
      w, h = im.size
      k = -1
      for i in range(0, w, int(w/3)):
        for j in range(0, h, int(h/3)):
          # TODO: Write a function for computing the box coordinates
          # and test it.
          box = (i, j, i + int(w/3), j + int(h/3))
          if i + int(w/3) < w and j + int(h/3) < h:
            k += 1
            image = im.crop(box)
            if k in splits_idx[:5]:
              yield image, idx, 'train'
            if k == splits_idx[5]:
              yield image, idx, 'dev'
            if k == splits_idx[6]:
              yield image, idx, 'dev-test'
            if k in splits_idx[7:]:
              yield image, idx, 'test'

  def select_subset(gen, subsets):
    for image, label, split in gen:
      if split in subsets:
        yield image, label

  per_split_gen = dict()
  per_split_gen['train'] = select_subset(gen(rng_seed), ['train',])
  per_split_gen['dev'] = select_subset(gen(rng_seed), ['dev',])
  per_split_gen['train_and_dev'] = select_subset(gen(rng_seed),
                                                 ['train', 'dev'])
  per_split_gen['dev-test'] = select_subset(gen(rng_seed), [
      'dev-test',
  ])
  per_split_gen['test'] = select_subset(gen(rng_seed), ['test',])

  return (metadata, per_split_gen)

brodatz_dataset = types.DownloadableDataset(
    name='brodatz',
    download_urls=[
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D1.gif',
            checksum='d5b7a11b6c2e21d4869626e2e43a6c76'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D2.gif',
            checksum='1de256b931c57e40c7bc9c3f64c6a77a'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D3.gif',
            checksum='ec3927c4f532f88f069700f8d6adfddd'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D4.gif',
            checksum='1ef331b11c9a3b85c34f4be1852b69e5'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D5.gif',
            checksum='8c6b91aee71dfcd66aee551e6a0609e0'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D6.gif',
            checksum='e6bb6971f81d319f623615d5694b3209'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D7.gif',
            checksum='c2fcd13fc32c2b631343445c0e230020'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D8.gif',
            checksum='801dade42334cac045e04a235f2986da'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D9.gif',
            checksum='dc65d59935048475ad4867d84ebbfa54'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D10.gif',
            checksum='75778b4707eb295101464113d78bec6e'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D11.gif',
            checksum='f659294380d33fa8752ac6b070d9c55b'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D12.gif',
            checksum='58a68e7fcdb1b0c32b6bb085ed3fe464'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D13.gif',
            checksum='74762bcca81edf3683d91871d4863898'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D15.gif',
            checksum='468611c6987f098b984bee1ef5feece5'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D16.gif',
            checksum='bd81a2680d168ed1bd156b7d840a7e0e'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D17.gif',
            checksum='4fa54c530e545ea9f6f3a9572e2197c7'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D18.gif',
            checksum='1649654b700b0ec8dea92d937db90e07'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D19.gif',
            checksum='c49ee81d5ac0241cc42765fbb9367140'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D20.gif',
            checksum='166d44aa57308f1044d5d6009d85964e'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D21.gif',
            checksum='070b869c8f38f6005595c062c09dd29e'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D22.gif',
            checksum='d5fc9a65b2a66afa641375e005b1c3a8'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D23.gif',
            checksum='68f3c724340a17cc9b71ccbbef2c625a'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D24.gif',
            checksum='553f108617063d3bae59dbc0842d40a6'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D25.gif',
            checksum='983966e908c0bb871d0c7eeb87d842eb'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D26.gif',
            checksum='25cb81107a1344bb0df5bb700ea0d545'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D27.gif',
            checksum='b4ad552c8153121f54311e0bf4d71742'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D28.gif',
            checksum='bcb1f90c91e63232fc482e861ad2a5ef'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D29.gif',
            checksum='97c091cf6bd85df9953fbacf4c26e653'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D30.gif',
            checksum='f640ea4d19451070ab7521d01fe0443c'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D31.gif',
            checksum='f2b021904c5657adff2f0ccd3c174da2'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D32.gif',
            checksum='2c106006421fd382c8bb7d0dde4a7757'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D33.gif',
            checksum='b87d02748fc35987ad31848eaa018309'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D34.gif',
            checksum='bfa73bb2478c5197a4841b26bbee319a'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D35.gif',
            checksum='708271b6fb9eff6ddb04ecd8144df2a1'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D36.gif',
            checksum='7e61234cc1f705872c7157197950f473'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D37.gif',
            checksum='efa7f55b325e0ec0adddfe2f13feb59f'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D38.gif',
            checksum='41a7f446618790981a7126ec4f11a3dc'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D39.gif',
            checksum='bf7c79d4bebfa5e39833e3d19722f906'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D40.gif',
            checksum='00916ab1171c486b2aaa97dff111b767'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D41.gif',
            checksum='09df782a9078662fad344fc00ebf15ef'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D42.gif',
            checksum='5c0c9878f3404e9f827e5b36e9e4bd78'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D43.gif',
            checksum='a3bb8e0a94e7bdf50bb202db4bbfd7cd'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D44.gif',
            checksum='fe9671ef6e3a847cacc014b4e34aed3a'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D45.gif',
            checksum='983b7bc79ce0510cce14ec1b1e16fa11'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D46.gif',
            checksum='bd9ff64e7e4d49f213d8ee68e4c96a73'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D47.gif',
            checksum='8153b39e1b9d535d7b5617f8af824770'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D48.gif',
            checksum='91fc6fc1df6984f2ee4caa0878f79e61'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D49.gif',
            checksum='9bf59efe485d20cefe403e22981bdf5f'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D50.gif',
            checksum='3f51cb54e0812916aab4dd7a3ff1d53f'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D51.gif',
            checksum='845012e87e736e7c086d860c3438395e'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D52.gif',
            checksum='b3e50f0ec2fd1a6fedb01633e45e723c'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D53.gif',
            checksum='fea16eb7f88bef7b8f59cb773c109a1e'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D54.gif',
            checksum='7ce97a6514196ec55b5641ca6da128e4'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D55.gif',
            checksum='83a53d7f3ed452d84bd24a5feb16ca82'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D56.gif',
            checksum='e05b6a4d4118a69e8c1dc21f367a2837'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D57.gif',
            checksum='ecf251b805b618c92f66eeaa42f3a555'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D58.gif',
            checksum='ed0ed6bf2f9d76f7763dfd2a59ade2d7'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D59.gif',
            checksum='ceabd5df9baeb447be7206e8f40b59c9'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D60.gif',
            checksum='18a2dea676bc12c9dfce54859a194132'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D61.gif',
            checksum='1f85e3af72d92fd24b8518f70f235150'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D62.gif',
            checksum='6f79c57fa556d4b449583787e2dcad28'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D63.gif',
            checksum='eb7ee131280bffc2b2416e719c848402'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D64.gif',
            checksum='80e74de3008d706d95097ea1e9f0c47c'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D65.gif',
            checksum='5a3d6019a57aab602339b7ce28b185da'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D66.gif',
            checksum='ad7eb2e9504c2f6aa420e902bf281e8b'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D67.gif',
            checksum='04b67bcc065507358a1cd139f6330386'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D68.gif',
            checksum='2de16cc286ca056234c9d961db6faf29'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D69.gif',
            checksum='847e8b08e204e51b9f6b37b27eb665e2'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D70.gif',
            checksum='e71ea4d910079d8076282a808447b092'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D71.gif',
            checksum='748bfdec8178eb759283b20f8f18c7b7'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D72.gif',
            checksum='aab06dd699291cb98bc9bf3858c5c8e2'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D73.gif',
            checksum='0fb12645d29c83e397bad8a4c2a02641'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D74.gif',
            checksum='84bc434804e1af9e0ebebe35e268fe63'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D75.gif',
            checksum='1925755a7bbb8c63eb84b1f764633040'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D76.gif',
            checksum='f1ad969319f6fc6bd7282c3a665580f1'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D77.gif',
            checksum='4a0bcc6bdb82e5a2021d1fd674143309'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D78.gif',
            checksum='84891367765a6645da6bf344960d3367'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D79.gif',
            checksum='23fe0d9309572a9398ab76585dfed68c'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D80.gif',
            checksum='cda3fa8f9beb4ebd7b1214ae52d63007'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D81.gif',
            checksum='03f9a63e224777b8fc6a831e294eb191'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D82.gif',
            checksum='71ff43df59c976685594c2699a7ca285'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D83.gif',
            checksum='72b08177d945df0f19fd7dee6d7d3199'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D84.gif',
            checksum='71cfd495fe5697ba2c97584362c763d7'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D85.gif',
            checksum='ec10e406c98376ef453e8ff84cd17ab7'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D86.gif',
            checksum='c8ae9a9b08c34f10c98e10b8fbe3faa4'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D87.gif',
            checksum='efd18bd1b96786cd0c1154d3b6607112'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D88.gif',
            checksum='7388e9b96303c330363a127b5f86de9a'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D89.gif',
            checksum='3fbfb4fcacd97bd8ff4d16c865e4d1c5'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D90.gif',
            checksum='9799b578783825b275a016d3f73f5ee9'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D91.gif',
            checksum='5ce405b9a67949c358b8425ad0eb043d'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D92.gif',
            checksum='0e1ad08968c216ec63989cea2ed97591'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D93.gif',
            checksum='51651d6a16ffac5eada215a9828b47dd'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D94.gif',
            checksum='7e665d45c5d145b9677501699ccc6ef9'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D95.gif',
            checksum='78d6a78e47f05bb0ae28a926310a3869'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D96.gif',
            checksum='40633ecff095460e126aa30e55e2b914'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D97.gif',
            checksum='4dbd7162f540bf106e8287b585798341'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D98.gif',
            checksum='5f11c141eb653f7401f9dd28e88cb73c'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D99.gif',
            checksum='ff60fd7aa813f72b8ef0cac840db6761'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D100.gif',
            checksum='8e5e62d263ce3bad21d8c01ac7c0faa5'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D101.gif',
            checksum='c8b21ce148aafb82635cb18966b0eac4'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D102.gif',
            checksum='fa6ac2cf66fe51318209ac74d9a08dee'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D103.gif',
            checksum='5976a960557eca00559042b0041921dd'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D104.gif',
            checksum='f0565aeebc36cad137af950125001082'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D105.gif',
            checksum='6f27031fed8269dd0fb9d36572eb84de'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D106.gif',
            checksum='b338b00b68eec8d35b14b82d5eef2ba8'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D107.gif',
            checksum='3d5c5fe771dab76f041cf58b4b7f95e8'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D108.gif',
            checksum='98e8c0881b909259cc812d0ee1a7f700'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D109.gif',
            checksum='3b0b3b050c09f5505f6c5079a702d87e'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D110.gif',
            checksum='4df933730394c919130a051ef1b5cd53'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D111.gif',
            checksum='459692fc135ea149fe40470d75c2f8ca'),
        types.DownloadableArtefact(
            url='http://www.ux.uis.no/~tranden/brodatz/D112.gif',
            checksum='6c4cedeb6915d76742fb224a44293dd6')
    ],
    handler=functools.partial(brodatz_handler, rng_seed=0))
