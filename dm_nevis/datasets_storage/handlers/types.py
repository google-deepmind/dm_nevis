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

"""Types required for handlers."""

import dataclasses
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from PIL import Image as pil_image
import tensorflow as tf
import tensorflow_datasets as tfds

ImageLike = Union[pil_image.Image, tf.Tensor]
ScalarLike = Union[np.ndarray, tf.Tensor, int, float]


class Example(NamedTuple):
  image: Optional[ImageLike]
  label: Optional[ScalarLike]
  multi_label: Optional[List[ScalarLike]]


@dataclasses.dataclass
class DatasetMetaData:
  num_classes: int
  num_channels: int
  image_shape: Tuple[int]
  additional_metadata: Dict[str, Any]
  features: Optional[tfds.features.FeaturesDict] = None
  preprocessing: str = ""


Image = pil_image.Image
Label = int
# We consider union for backwards compatibility for tuples (image, label)
DataGenerator = Iterable[Union[Tuple[Image, Label], Example]]
HandlerOutput = Tuple[DatasetMetaData, Dict[str, DataGenerator]]
Handler = Callable[[str], HandlerOutput]
FixtureWriterFn = Callable[[str], None]


@dataclasses.dataclass(frozen=True)
class KaggleCompetition:
  """Static metadata for kaggle competitions."""
  competition_name: str
  checksum: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class KaggleDataset:
  """Static metadata for kaggle Dataset.

  Dataset name is of the format `user/dataset`.
  """
  dataset_name: str
  checksum: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class DownloadableArtefact:
  """Static metadata for any artefact that can be downloaded."""
  url: str
  checksum: Optional[str] = None


Artefact = Union[DownloadableArtefact, KaggleCompetition, KaggleDataset]


@dataclasses.dataclass(frozen=True)
class DownloadableDataset:
  """Static metadata for downloadabe datasets.

  Attributes:
    name: The name of the dataset.
    handler: A callable that generates iterators over the dataset features,
      given a path contaiing the downloaded dataset artifacts.
    download_urls: URLs for all of the dataset artifacts or kaggle
      competition/dataset.
    manual_download: If the data artifacts must be manually downloaded.
    website_url: The (optional) dataset homepage.
    paper_url: The (optional) URL for the dataset.
    paper_title: The (optional) title of the paper where the dataset is defined.
    authors: The (optional) authors of the dataset's paper.
    year: The (optional) year the dataset's paper was published.
    papers_with_code_url: If the dataset has a page on papers with code, it may
      be stored here.
    fixture_writer: An (optional) function to write a dataset fixture for the
      dataset. This is a callable that writes fixture versions of the dataset
      artifact to a given path, and is intended for use in tests.
  """
  name: str
  handler: Handler
  download_urls: List[Artefact]
  manual_download: bool = False
  website_url: Optional[str] = None
  paper_url: Optional[str] = None
  paper_title: Optional[str] = None
  authors: Optional[str] = None
  year: Optional[int] = None
  papers_with_code_url: Optional[str] = None
  fixture_writer: Optional[FixtureWriterFn] = None
