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

"""Information about LSCL datasets for the benchmark."""

# Datasets mapping from canonical name to tfds_name, available in tfds.
TFDS_DATASETS_MAPPING = {
    "Caltech 101": "caltech101",
    "CUB 200": "caltech_birds2011",
    "CIFAR 10": "cifar10",
    "CIFAR 100": "cifar100",
    "COIL 100": "coil100",
    "CUB 200 2011": "caltech_birds2011",
    "DomainNet-Real": "domainnet",
    "EMNIST Balanced": "emnist/balanced",
    "FashionMNIST": "fashion_mnist",
    "Food 101 N": "food101",
    "ImageNet": "imagenet2012",
    "iNaturalist": "i_naturalist2017",
    "Oxford Flowers": "oxford_flowers102",
    "Oxford Pets": "oxford_iiit_pet",
    "Stanford Dogs": "stanford_dogs",
    "SUN": "sun397",
}
