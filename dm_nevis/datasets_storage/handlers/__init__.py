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

"""All handlers."""

from typing import Iterable, List, Optional, Union, Tuple

from dm_nevis.datasets_storage.handlers import aberdeen
from dm_nevis.datasets_storage.handlers import aloi
from dm_nevis.datasets_storage.handlers import alot
from dm_nevis.datasets_storage.handlers import animal
from dm_nevis.datasets_storage.handlers import animal_web
from dm_nevis.datasets_storage.handlers import awa2
from dm_nevis.datasets_storage.handlers import belgium_tsc
from dm_nevis.datasets_storage.handlers import biwi
from dm_nevis.datasets_storage.handlers import brodatz
from dm_nevis.datasets_storage.handlers import butterflies
from dm_nevis.datasets_storage.handlers import caltech256
from dm_nevis.datasets_storage.handlers import caltech_camera_traps
from dm_nevis.datasets_storage.handlers import caltech_categories
from dm_nevis.datasets_storage.handlers import casia_hwdb
from dm_nevis.datasets_storage.handlers import chars74k
from dm_nevis.datasets_storage.handlers import cmu_amp_expression
from dm_nevis.datasets_storage.handlers import coco
from dm_nevis.datasets_storage.handlers import coil20
from dm_nevis.datasets_storage.handlers import covid_19_xray
from dm_nevis.datasets_storage.handlers import cvc_muscima
from dm_nevis.datasets_storage.handlers import ddsm
from dm_nevis.datasets_storage.handlers import extended_yaleb
from dm_nevis.datasets_storage.handlers import fgvc_aircraft
from dm_nevis.datasets_storage.handlers import flickr_material_database as fmd
from dm_nevis.datasets_storage.handlers import food101
from dm_nevis.datasets_storage.handlers import food101n
from dm_nevis.datasets_storage.handlers import german_tsr
from dm_nevis.datasets_storage.handlers import iaprtc12
from dm_nevis.datasets_storage.handlers import ig02
from dm_nevis.datasets_storage.handlers import interact
from dm_nevis.datasets_storage.handlers import kth_tips
from dm_nevis.datasets_storage.handlers import landsat
from dm_nevis.datasets_storage.handlers import lfw
from dm_nevis.datasets_storage.handlers import lfwa
from dm_nevis.datasets_storage.handlers import magellan_venus_volcanoes
from dm_nevis.datasets_storage.handlers import mall
from dm_nevis.datasets_storage.handlers import melanoma
from dm_nevis.datasets_storage.handlers import mit_scenes
from dm_nevis.datasets_storage.handlers import mnist_m
from dm_nevis.datasets_storage.handlers import mnist_rotation
from dm_nevis.datasets_storage.handlers import mpeg7
from dm_nevis.datasets_storage.handlers import nih_chest_xray
from dm_nevis.datasets_storage.handlers import not_mnist
from dm_nevis.datasets_storage.handlers import office31
from dm_nevis.datasets_storage.handlers import office_caltech_10
from dm_nevis.datasets_storage.handlers import olivetti_face
from dm_nevis.datasets_storage.handlers import oxford_flowers_17
from dm_nevis.datasets_storage.handlers import pacs
from dm_nevis.datasets_storage.handlers import pascal_voc2005
from dm_nevis.datasets_storage.handlers import pascal_voc2006
from dm_nevis.datasets_storage.handlers import pascal_voc2007
from dm_nevis.datasets_storage.handlers import path_mnist
from dm_nevis.datasets_storage.handlers import pneumonia_chest_xray
from dm_nevis.datasets_storage.handlers import ppmi
from dm_nevis.datasets_storage.handlers import scenes15
from dm_nevis.datasets_storage.handlers import scenes8
from dm_nevis.datasets_storage.handlers import semeion
from dm_nevis.datasets_storage.handlers import shanghai_tech
from dm_nevis.datasets_storage.handlers import silhouettes
from dm_nevis.datasets_storage.handlers import sketch
from dm_nevis.datasets_storage.handlers import stanford_cars
from dm_nevis.datasets_storage.handlers import sun_attributes
from dm_nevis.datasets_storage.handlers import synthetic_covid19_xray
from dm_nevis.datasets_storage.handlers import tid
from dm_nevis.datasets_storage.handlers import tiny_imagenet
from dm_nevis.datasets_storage.handlers import trancos
from dm_nevis.datasets_storage.handlers import tubercolosis
from dm_nevis.datasets_storage.handlers import types
from dm_nevis.datasets_storage.handlers import uiuc_cars
from dm_nevis.datasets_storage.handlers import uiuc_texture
from dm_nevis.datasets_storage.handlers import umd
from dm_nevis.datasets_storage.handlers import umist
from dm_nevis.datasets_storage.handlers import usps
from dm_nevis.datasets_storage.handlers import vistex
from dm_nevis.datasets_storage.handlers import voc_actions
from dm_nevis.datasets_storage.handlers import wiki_paintings

_DATASETS_TO_HANDLERS = {
    'sun_attributes':
        sun_attributes.sun_attributes_dataset,
    'animal_web':
        animal_web.animal_web_dataset,
    'aberdeen':
        aberdeen.aberdeen_dataset,
    'animal':
        animal.animal_dataset,
    'aloi':
        aloi.aloi_dataset,
    'aloi_grey':
        aloi.aloi_grey_dataset,
    'alot':
        alot.alot_dataset,  # NOTYPO
    'alot_grey':
        alot.alot_grey_dataset,  # NOTYPO
    'awa2':
        awa2.awa2_dataset,
    'belgium_tsc':
        belgium_tsc.belgium_tsc_dataset,
    'biwi':
        biwi.biwi_dataset,
    'brodatz':
        brodatz.brodatz_dataset,
    'butterflies':
        butterflies.butterflies_dataset,
    'chars74k':
        chars74k.chars74k_dataset,
    'caltech256':
        caltech256.caltech256_dataset,
    'caltech_categories':
        caltech_categories.caltech_categories_dataset,
    'casia_hwdb':
        casia_hwdb.casia_hwdb_dataset,
    'cmu_amp_expression':
        cmu_amp_expression.cmu_amp_expression_dataset,
    'coco_single_label':
        coco.coco_single_label_dataset,
    'coco_multi_label':
        coco.coco_multi_label_dataset,
    'coil20':
        coil20.coil_20_dataset,
    'coil20_unproc':
        coil20.coil_20_unproc_dataset,
    'extended_yaleb':
        extended_yaleb.extended_yaleb_dataset,
    'fgvc_aircraft_family':
        fgvc_aircraft.fgvc_aircraft_family_dataset,
    'fgvc_aircraft_manufacturer':
        fgvc_aircraft.fgvc_aircraft_manufacturer_dataset,
    'fgvc_aircraft_variant':
        fgvc_aircraft.fgvc_aircraft_variant_dataset,
    'covid_19_xray':
        covid_19_xray.covid_19_xray_dataset,
    'ddsm':
        ddsm.ddsm_dataset,
    'flickr_material_database':
        fmd.flickr_material_database_dataset,
    'german_tsr':
        german_tsr.german_tsr_dataset,
    'iaprtc12':
        iaprtc12.iaprtc12_dataset,
    'ig02':
        ig02.ig02_dataset,
    'interact':
        interact.interact_dataset,
    'kth_tips':
        kth_tips.kth_tips_dataset,
    'kth_tips_grey':
        kth_tips.kth_tips_grey_dataset,
    'kth_tips_2a':
        kth_tips.kth_tips_2a_dataset,
    'kth_tips_2b':
        kth_tips.kth_tips_2b_dataset,
    'landsat':
        landsat.landsat_dataset,
    'lfw':
        lfw.lfw_dataset,
    'lfwa':
        lfwa.lfwa_dataset,
    'magellan_venus_volcanoes':
        magellan_venus_volcanoes.magellan_venus_volcanoes_dataset,
    'mall':
        mall.mall_dataset,
    'melanoma':
        melanoma.melanoma_dataset,
    'mit_scenes':
        mit_scenes.mit_scenes_dataset,
    'mnist_m':
        mnist_m.mnist_m_dataset,
    'nih_chest_xray':
        nih_chest_xray.nih_chest_xray_dataset,
    'office31':
        office31.office31_dataset,
    'office_caltech_10':
        office_caltech_10.office_caltech_10_dataset,
    'pacs':
        pacs.pacs_dataset,
    'pascal_voc2005':
        pascal_voc2005.pascal_voc2005_dataset,
    'pascal_voc2006':
        pascal_voc2006.pascal_voc2006_dataset,
    'pascal_voc2007':
        pascal_voc2007.pascal_voc2007_dataset,
    'ppmi':
        ppmi.ppmi_dataset,
    'scenes8':
        scenes8.scenes8_dataset,
    'scenes15':
        scenes15.scenes15_dataset,
    'shanghai_tech':
        shanghai_tech.shanghai_tech_dataset,
    'silhouettes_16':
        silhouettes.silhouettes_16_dataset,
    'silhouettes_28':
        silhouettes.silhouettes_28_dataset,
    'sketch':
        sketch.sketch_dataset,
    'not_mnist':
        not_mnist.not_mnist_dataset,
    'oxford_flowers_17':
        oxford_flowers_17.oxford_flowers_17_dataset,
    'trancos':
        trancos.trancos_dataset,
    'synthetic_covid19_xray':
        synthetic_covid19_xray.synthetic_covid19_xray_dataset,
    'stanford_cars':
        stanford_cars.stanford_cars_dataset,
    'tid2008':
        tid.tid2008_dataset,
    'tid2013':
        tid.tid2013_dataset,
    'olivetti_face':
        olivetti_face.olivetti_face_dataset,
    'path_mnist':
        path_mnist.path_mnist_dataset,
    'pneumonia_chest_xray':
        pneumonia_chest_xray.pneumonia_chest_xray_dataset,
    'tubercolosis':
        tubercolosis.tubercolosis_dataset,
    'uiuc_cars':
        uiuc_cars.uiuc_cars_dataset,
    'uiuc_texture':
        uiuc_texture.uiuc_texture_dataset,
    'umist':
        umist.umist_dataset,
    'usps':
        usps.usps_dataset,
    'semeion':
        semeion.semeion_dataset,
    'food101':
        food101.food101_dataset,
    'food101n':
        food101n.food101n_dataset,
    'caltech_camera_traps':
        caltech_camera_traps.caltech_camera_traps_dataset,
    'cvc_muscima':
        cvc_muscima.cvc_muscima_dataset,
    'mpeg7':
        mpeg7.mpeg7_dataset,
    'tiny_imagenet':
        tiny_imagenet.tiny_imagenet_dataset,
    'mnist_rotation':
        mnist_rotation.mnist_rotation_dataset,
    'umd':
        umd.umd_dataset,
    'vistex':
        vistex.vistex_dataset,
    'voc_actions':
        voc_actions.voc_actions_dataset,
    'wiki_paintings_artist':
        wiki_paintings.wiki_paintings_dataset_artist,
    'wiki_paintings_genre':
        wiki_paintings.wiki_paintings_dataset_genre,
    'wiki_paintings_style':
        wiki_paintings.wiki_paintings_dataset_style,
}


def get_links_for_dataset(dataset: str) -> List[types.Artefact]:
  return _DATASETS_TO_HANDLERS[dataset].download_urls


def get_handler_for_dataset(dataset: str) -> types.Handler:
  return _DATASETS_TO_HANDLERS[dataset].handler


def is_dataset_available(dataset_name: str) -> bool:
  return dataset_name in _DATASETS_TO_HANDLERS


def dataset_names() -> Iterable[str]:
  return _DATASETS_TO_HANDLERS.keys()


def get_dataset(dataset_name: str) -> types.DownloadableDataset:
  return _DATASETS_TO_HANDLERS[dataset_name]
