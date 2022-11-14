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

"""Animal WEB handler."""

import os
import subprocess
import tarfile
from dm_nevis.datasets_storage.handlers import extraction_utils as eu
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
from PIL import Image

from tensorflow.io import gfile


_RAR_FNAME = 'animal_dataset_v1_c.rar'
_TAR_FNAME = 'animal_dataset_v1_c.tar.gz'

_LABELS = [
    'saluki', 'kultarr', 'bighornsheep', 'bandedmongoose', 'fox', 'goat',
    'tasmaniandevil', 'wallaroo', 'whiptailwallaby', 'feralhorse', 'binturong',
    'geoffroyscat', 'klipspringer', 'vervetmonkey', 'possum', 'arcticwolf',
    'rustyspottedgenet', 'capegraymongoose', 'elk', 'commonbrownlemur', 'panda',
    'amurtiger', 'quokka', 'indri', 'debrazzasmonkey', 'collaredpeccary',
    'cloudedleopard', 'emperorpenguin', 'zebu', 'wildcat', 'lionpd', 'walrus',
    'irishwolfhound', 'hoodedseal', 'camel', 'patasmonkey', 'commongenet',
    'italiangreyhound', 'viverratangalungamalayancivet', 'bullmastif',
    'blackbackedjackal', 'pekingesedog', 'lumholtzstreekangaroo', 'komondor',
    'toquemacaque', 'longnosedmongoose', 'matschiestreekangaroo', 'woodchuck',
    'slendermongoose', 'gundi', 'animal', 'feralcat', 'chamois',
    'borneanslowloris', 'dachshund', 'yelloweyedpenguin', 'harpseal', 'bharal',
    'blueeyedblacklemur', 'treeshrew', 'dallsheep', 'brushtailedrockwallaby',
    'marmoset', 'goldenbamboolemur', 'greaterbamboolemur', 'balinesecat',
    'californiansealion', 'fallowdeer', 'adeliepenguin', 'waterbuck',
    'mareebarockwallaby', 'horse', 'zebra', 'blackrhino', 'australianterrier',
    'wildebeest', 'monte', 'oncilla', 'armadillo', 'frenchbulldog',
    'swamprabbit', 'cheetah', 'gentoopenguin', 'greylangur', 'mouflon',
    'alaskanmalamute', 'amurleopard', 'dhole', 'baikalseal', 'brownrat',
    'kingpenguin', 'redpanda', 'hamster', 'echidna', 'bushbaby', 'fishingcat',
    'westernlesserbamboolemur', 'beardedseal', 'colo', 'roanantelope',
    'harbourseal', 'chinstrappenguin', 'giantschnauzer', 'collaredbrownlemur',
    'stripedhyena', 'opossum', 'guanaco', 'wisent', 'visayanwartypig',
    'barbarymacaque', 'onager', 'caiman', 'feralgoat', 'commonwarthog',
    'hartebeest', 'arcticfox', 'whiteheadedlemur', 'spottedseal', 'capebuffalo',
    'medraneanmonkseal', 'jaguar', 'wildass', 'barbarysheep', 'gibbons',
    'spottedhyena', 'leopardcat', 'hedgehog', 'uc', 'topi', 'commonchimpanzee',
    'dunnart', 'agilewallaby', 'sundaslowloris', 'wombat', 'chinesegoral',
    'caribou', 'weddellseal', 'canadianlynx', 'husky', 'liger',
    'sharpesgrysbok', 'graywolf', 'hare', 'caracal', 'hyrax', 'platypus',
    'capefox', 'eurasianlynx', 'oribi', 'northernelephantseal',
    'centralchimpanzee', 'dormouse', 'gerbil', 'cougar', 'capybara', 'ferret',
    'przewalskihorse', 'crestedpenguin', 'oryx', 'steinbucksteenbok', 'nilgai',
    'mangabey', 'australianshepherd', 'spidermonkey', 'monkey', 'brownhyena',
    'roedeer', 'bull', 'pardinegenet', 'anoa', 'leopard', 'swampwallaby',
    'bonobo', 'cottonrat', 'vole', 'humboldtpenguin', 'africanpenguin',
    'goldenjackal', 'gorilla', 'commonkusimanse', 'redtailmonkey', 'aardwolf',
    'suni', 'blackandwhiteruffedlemar', 'chowchow', 'raccoon', 'bolognesedog',
    'kangaroo', 'dalmatian', 'gharial', 'australiancattledog', 'ruddymongoose',
    'nightmonkey', 'swiftfox', 'weasel', 'easternchimpanzee', 'bison', 'yak',
    'chital', 'titi', 'woollymonkey', 'reedbuck', 'pallascat',
    'smallasianmongoose', 'grizzlybear', 'rustyspottedcat', 'coatis',
    'redruffedlemur', 'kinkajou', 'parmawallaby', 'coypu', 'westernchimpanzee',
    'asianpalmcivet', 'domesticcat', 'giantotter', 'pygmyrabbit', 'grayfox',
    'kiang', 'pademelon', 'lemur', 'wapiti', 'asiangoldencat', 'agouti',
    'bandedpalmcivet', 'fieldmouse', 'junglecat', 'anteater', 'mexicanwolf',
    'largespottedgenet', 'beatingmongoose', 'goodfellowstreekangaroo',
    'flyingsquirrel', 'wolverine', 'guineapig', 'dassie', 'orangutan',
    'greyseal', 'ocelot', 'howler', 'germanpinscher', 'koala', 'bilby',
    'goldenretriever', 'galagos', 'leopardseal', 'spottedneckedotter',
    'crownedlemur', 'owstonspalmcivet', 'donkey', 'duiker', 'pygmyslowloris',
    'cservalserval', 'hippopotamus', 'tamarin', 'alaskanhare', 'badger',
    'dingo', 'boar', 'goldenlangur', 'greatdane', 'jackrabbit', 'uakari',
    'colobus', 'fennecfox', 'sandcat', 'bamboolemur', 'bengalslowloris',
    'dugong', 'rhesusmonkey', 'marshmongoose', 'littlebluepenguin', 'hogdeer',
    'redbelliedsquirrel', 'commondwarfmongoose', 'corsacfox', 'whitewolf',
    'addax', 'stripeneckedmongoose', 'deermouse', 'japanesemacaque', 'giraffe',
    'babirusa', 'hamadryasbaboon', 'douclangur', 'anatolianshepherddog',
    'bluemonkey', 'muskox', 'yellowfootedrockwallaby', 'gerenuk', 'doberman',
    'hawaiianmonkseal', 'magellanicpenguin', 'crabeaterseal', 'bobcat',
    'feralcattle', 'jaguarundi', 'potoroo', 'muntjacdeer', 'geladababoon',
    'harvestmouse', 'rhinoceros', 'olivebaboon', 'buffalo', 'patagonianmara',
    'bushbuck', 'blackbuck', 'beaver', 'zonkey', 'bordercollie',
    'southernelephantseal', 'tammarwallaby', 'olingos', 'quoll',
    'easternlesserbamboolemur', 'bandicoot', 'alpineibex', 'redneckedwallaby',
    'bear', 'japaneseserow', 'galapagossealion', 'muriqui', 'blackfootedcat',
    'cacomistle', 'ringtail', 'germanshepherddog', 'ribbonseal', 'domesticdog',
    'lutung', 'tarsiers', 'margay', 'bongo', 'francoislangur', 'potto',
    'whitetaileddeer', 'australiansealion', 'capuchinmonkey', 'dikdik',
    'aardvark', 'snowleopard', 'banteng', 'chihuahua', 'proboscismonkey',
    'ayeaye', 'pantanalcat', 'ethiopianwolf', 'africanwilddog', 'deer',
    'alpaca', 'servalinegenet', 'chipmunk', 'degu', 'urial'
]


def animal_web_handler(
    dataset_path: str,
    apply_unrar: bool = True
) -> types.HandlerOutput:
  """Handler for AnimalWeb dataset."""

  label_to_id = dict(
      ((label, label_id) for label_id, label in enumerate(_LABELS)))

  metadata = types.DatasetMetaData(
      num_classes=len(label_to_id),
      num_channels=3,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_to_id=label_to_id,
          task_type='classification',
          image_type='object',
      ))

  # Unarachive the images.
  if apply_unrar:
    # Normally, we assume that we have access to unrar utility and use
    # rar archive.
    subprocess.call(['unrar', 'e', '-y', '-idq', _RAR_FNAME], cwd=dataset_path)
  else:
    # In this case, we assume that we pre-archived a file in a tar format.
    tarfile.open(os.path.join(dataset_path, _TAR_FNAME),
                 'r|gz').extractall(path=dataset_path)

  def make_gen():
    for fname in gfile.listdir(dataset_path):
      if not fname.endswith('jpg'):
        continue
      image_fname = os.path.splitext(fname)[0]
      label_name = image_fname.split('_')[0]
      label = label_to_id[label_name]
      image = Image.open(os.path.join(dataset_path, fname)).convert('RGB')
      yield (image, label)

  deduplicated_data_gen = eu.deduplicate_data_generator(make_gen())

  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      deduplicated_data_gen, splits.SPLIT_WITH_FRACTIONS_FOR_ALL_DATA,
      splits.MERGED_TRAIN_AND_DEV)

  return metadata, per_split_gen


animal_web_dataset = types.DownloadableDataset(
    name='animal_web',
    download_urls=[
        types.DownloadableArtefact(
            url='https://drive.google.com/uc?export=download&id=13PbHxUofhdJLZzql3TyqL22bQJ3HwDK4&confirm=y',
        checksum='d2d7e0a584ee4bd9badc74a9f2ef3b82')
    ],
    website_url='https://fdmaproject.wordpress.com/author/fdmaproject/',
    handler=animal_web_handler)
