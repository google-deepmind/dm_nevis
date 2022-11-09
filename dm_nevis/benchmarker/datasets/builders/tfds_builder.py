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

"""Dataset builders for TFDS."""

from typing import Optional

from dm_nevis.benchmarker.datasets import dataset_builders
from dm_nevis.benchmarker.datasets import datasets
from dm_nevis.benchmarker.datasets import tasks
import tensorflow as tf
import tensorflow_datasets as tfds


DEFAULT_SHUFFLE_BUFFER_SIZE = 5_000

SINGLE_LABEL_DATASETS = [
    "caltech101",
    "cifar10",
    "cifar100",
    "caltech_birds2011",
    "dtd",
    "emnist/balanced",
    "fashion_mnist",
    "food101",
    "imagenet2012",
    "mnist",
    "oxford_flowers102",
    "oxford_iiit_pet",
    "patch_camelyon",
    "stanford_dogs",
    "stl10",
    "sun397",
    "svhn_cropped",
    "smallnorb",
    "domainnet/sketch",
]

MULTI_LABEL_DATASETS = [
    "voc/2012",
    "voc/2007",
    "celeb_a",
    # "lfw",
    # "coco",  # Add support for this multi-label dataset.
]

SUPPORTED_DATASETS = SINGLE_LABEL_DATASETS + MULTI_LABEL_DATASETS


_CELEB_A_ATTRIBUTES = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]


def get_single_label_dataset(dataset_name: str, split: str, start: int,
                             end: int) -> datasets.Dataset:
  """Gets single class tfds dataset."""
  dataset_info = tfds.builder(dataset_name).info

  label_key = "label"
  if dataset_name == "smallnorb":
    label_key = "label_category"

  num_classes = dataset_info.features[label_key].num_classes

  task_name = dataset_name.translate(str.maketrans(" -/", "___"))
  task_key = tasks.TaskKey(
      task_name, tasks.TaskKind.CLASSIFICATION,
      tasks.ClassificationMetadata(num_classes=num_classes))

  builder_fn, metadata = dataset_builders.tfds_dataset_builder_fn(
      tfds_name=dataset_name,
      shuffle_buffer_size=DEFAULT_SHUFFLE_BUFFER_SIZE,
      split=split,
      start=start,
      end=end,
      to_minibatch_fn=lambda x: _to_minibatch_single_label(x, label_key))

  return datasets.Dataset(
      builder_fn=builder_fn,
      task_key=task_key,
      num_examples=metadata.num_examples)


def _to_minibatch_single_label(data, label_key) -> datasets.MiniBatch:
  return datasets.MiniBatch(
      image=data["image"],
      label=data[label_key],
      multi_label_one_hot=None,
  )


def _to_minibatch_multi_label(data, multi_label_key,
                              num_classes) -> datasets.MiniBatch:
  image = data["image"]
  multi_label = data[multi_label_key]
  multi_label_one_hot = tf.reduce_sum(
      tf.one_hot(multi_label, num_classes), axis=0)
  return datasets.MiniBatch(
      image=image,
      label=None,
      multi_label_one_hot=multi_label_one_hot,
  )


def _to_minibatch_celeb_a(data) -> datasets.MiniBatch:
  image = data["image"]
  attributes = []
  for attr in _CELEB_A_ATTRIBUTES:
    attributes.append(tf.cast(data["attributes"][attr], tf.float32))
  attributes = tf.stack(attributes, axis=0)
  return datasets.MiniBatch(
      image=image,
      label=None,
      multi_label_one_hot=attributes,
  )


def get_multi_label_dataset(dataset_name: str, split: str, start: int,
                            end: int) -> datasets.Dataset:
  """Gets multi label tfds dataset."""
  dataset_info = tfds.builder(dataset_name).info
  task_name = dataset_name.translate(str.maketrans(" -/", "___"))

  if dataset_name == "celeb_a":
    num_classes = len(dataset_info.features["attributes"])
    to_minibatch_fn = _to_minibatch_celeb_a
  elif dataset_name == "voc/2012" or dataset_name == "voc/2007":
    multi_labels_key = "labels"
    # pylint: disable=g-long-lambda
    num_classes = dataset_info.features[multi_labels_key].num_classes
    to_minibatch_fn = lambda x: _to_minibatch_multi_label(
        x, multi_labels_key, num_classes)
  else:
    raise ValueError(f"Unsupported dataset: {dataset_name}")

  task_key = tasks.TaskKey(
      task_name, tasks.TaskKind.MULTI_LABEL_CLASSIFICATION,
      tasks.MultiLabelClassificationMetadata(num_classes=num_classes))

  builder_fn, metadata = dataset_builders.tfds_dataset_builder_fn(
      tfds_name=dataset_name,
      shuffle_buffer_size=DEFAULT_SHUFFLE_BUFFER_SIZE,
      split=split,
      start=start,
      end=end,
      to_minibatch_fn=to_minibatch_fn)

  return datasets.Dataset(
      builder_fn=builder_fn,
      task_key=task_key,
      num_examples=metadata.num_examples)


def get_dataset(dataset_name: str,
                split: str,
                *,
                start: Optional[int] = None,
                end: Optional[int] = None) -> datasets.Dataset:
  """Gets tensorflow dataset."""

  if dataset_name not in SUPPORTED_DATASETS:
    raise ValueError(f"Unsupported dataset: {dataset_name}")

  if dataset_name in SINGLE_LABEL_DATASETS:
    return get_single_label_dataset(dataset_name, split, start, end)
  elif dataset_name in MULTI_LABEL_DATASETS:
    return get_multi_label_dataset(dataset_name, split, start, end)
  else:
    raise ValueError(f"Unsupported dataset: {dataset_name}")
