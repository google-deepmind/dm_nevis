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

"""Casia HWDB 1.1 dataset handler."""

import codecs
import io
import os
import struct
import zipfile
from dm_nevis.datasets_storage.handlers import splits
from dm_nevis.datasets_storage.handlers import types
import numpy as np
from PIL import Image

from tensorflow.io import gfile


def casia_hwdb_handler(artifacts_path: str) -> types.HandlerOutput:
  """Imports Casia HWDB 1.1 dataset.

  The dataset home page is at
  http://www.nlpr.ia.ac.cn/databases/handwriting/Offline_database.html
  This dataset is an OCR dataset of recognition of handwriting Chinese
  characters.
  The dataset comes as a set of 3 zip files each containing several binary files
  with the actual images in a binary format (see website for details). Images
  have variable spatial resolution, and the background is set to 255.
  There are a total of 1,172,907 images and 3926 labels (Chinese and other
  characters).
  The data is pre-split into training and test sets, as specified by the
  filenames of the zip files.

  Args:
    artifacts_path: Path with downloaded artifacts.

  Returns:
    Metadata and generator functions.
  """
  files = gfile.listdir(artifacts_path)
  assert files

  def extract_data(stream, keep_ima):
    # keep_ima is a boolean flag. If it is true then we return also a list
    # with the images present in the binary stream. If it is false we return an
    # empty list (and save time and memory).
    images = []
    labels = []
    # Get total number of bytes, useful to know when to stop reading.
    stream.seek(0, 2)
    eof = stream.tell()
    stream.seek(0, 0)
    while stream.tell() < eof:
      packed_length = stream.read(4)
      length = struct.unpack("<I", packed_length)[0]
      raw_label = struct.unpack(">cc", stream.read(2))
      try:
        label = codecs.decode(raw_label[0] + raw_label[1], encoding="gb2312-80")
      except Exception:  # pylint: disable=broad-except
        # In few cases decoding fails, we then store the raw byte label.
        label = raw_label[0] + raw_label[1]
      width = struct.unpack("<H", stream.read(2))[0]
      height = struct.unpack("<H", stream.read(2))[0]
      assert length == width * height + 10
      raw_image = struct.unpack("{}B".format(height * width),
                                stream.read(height * width))
      if keep_ima:
        image = np.array(raw_image, dtype=np.uint8).reshape(height, width)
        image = 255 - image  # Set background to 0.
        images.append(Image.fromarray(image))
      labels.append(label)

    return images, labels

  # Do a first pass over the data to figure out the labels.
  print("Extracting labels")
  label_str_to_int = dict()
  all_labels = []
  for zip_fname in files:
    with zipfile.ZipFile(os.path.join(artifacts_path, zip_fname), "r") as zf:
      for name in sorted(zf.namelist()):
        f = zf.getinfo(name)
        if f.is_dir():
          continue
        _, labels = extract_data(io.BytesIO(zf.read(f)), False)
        all_labels += labels
  num_examples = len(all_labels)
  all_labels = list(set(all_labels))  # Get unique labels.
  num_classes = len(all_labels)
  for i, label in enumerate(all_labels):
    label_str_to_int[label] = i
  print("There are a total of " + str(num_examples) + " examples and " +
        str(len(all_labels)) + " labels.")

  metadata = types.DatasetMetaData(
      num_classes=num_classes,  # 3926
      num_channels=1,
      image_shape=(),  # Ignored for now.
      additional_metadata=dict(
          label_to_id=label_str_to_int,
          task_type="classification",
          image_type="ocr"))

  def make_gen(curr_files):
    for zip_fname in curr_files:
      with zipfile.ZipFile(os.path.join(artifacts_path, zip_fname), "r") as zf:
        for name in sorted(zf.namelist()):
          f = zf.getinfo(name)
          if f.is_dir():
            continue
          images, labels = extract_data(io.BytesIO(zf.read(f)), True)
          for image, label in zip(images, labels):
            yield (image, label_str_to_int[label])

  training_files = ["Gnt1.1TrainPart1.zip", "Gnt1.1TrainPart2.zip"]
  make_gen_fn = lambda: make_gen(training_files)
  per_split_gen = splits.random_split_generator_into_splits_with_fractions(
      make_gen_fn, splits.SPLIT_WITH_FRACTIONS_FOR_TRAIN,
      splits.MERGED_TRAIN_AND_DEV)
  per_split_gen["test"] = make_gen(["Gnt1.1Test.zip"])

  return metadata, per_split_gen

# pylint: disable=line-too-long
# Unofficial links (but still from the authors) for the three files are:
# - https://mail.cstnet.cn/coremail/downloadFile.jsp?key=1U31SsvkjqFd-VIyT8jLTejXo9vqo90yo9FLUu2kTDawSstETnvC3ZJdantET9GJfgKm3cKq3XKyTgGka1AGF4fdTuxEOnECzVFA3sa9wZEXaWCLTejXo9vqo90yo9FLwujXTWCyfu2LaV7L3ZtmjeIk-s71UUUUjU2l39EtjqanVW8XFWDZFW3Zr4UKw45yVZGmSsFdUZCmaUtsU18USUjgUnkU7DjBUnrU4UjjUnAU4DjKU0xUx7jFU88UcUj2U88U77jWU0fj-7wfvpGQ&code=cv62nuho
# - https://gzc-download.ftn.qq.com/ftn_handler/f2fdd9604fbe398e97ed995547e4c31ef0460bab2a0afae99aa1526566f5251d819ba1210e5df5ed0df45ff8ad04d1a1e0ce689d26c00ec724bc97425a87e46d/?fname=Gnt1.1TrainPart2.zip&xffz=966309492&k=9c9b0132e9731ec9aee84a323734346417d2fc0b35343464464b5c0b530506505a56490b02550149575c5054180d0306054900575700035451055051570c2064240a10031b056016020d0a62544640564d1e0d42213f39deeb7da2617ba5ce637d2a8788a0cd38e4ab&code=cdd2544d
# - https://njc-download.ftn.qq.com/ftn_handler/7eedd59d4d7a6f3ccbef7057e36f92abd09aa682a0f325491ead9454ea469ad0c68165f3683402a77b59e8085a8950a650aca0848436379699601bad022b5d20/?fname=Gnt1.1Test.zip&xffz=479659120&k=cb9c6b61e9234f98f9ef1a61376465354467a37d35646535114c0d000c505d040555195757015c1800525057185c560d044e0202505c065605510d0251556b35730d40501b55315047171a1b5c14719be76aeaae89656afaf0d85f83d48a3a643e51f0&code=4c4a5de5
# In the event that the official server times out, the above links can be used
# as a fall back.

casia_hwdb_dataset = types.DownloadableDataset(
    name="casia_hwdb",
    download_urls=[
        types.DownloadableArtefact(
            url="http://www.nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.1TrainPart1.zip",
            checksum="72bac7b6a5ce37f184f277421adfacfd"),
        types.DownloadableArtefact(
            url="http://www.nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.1TrainPart2.zip",
            checksum="a8b76e4eccfb1fd8d56c448f6a096c27"),
        types.DownloadableArtefact(
            url="http://www.nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.1Test.zip",
            checksum="e74f9a4863e73d8b80ed402452c97448")
    ],
    handler=casia_hwdb_handler,
    paper_title="CASIA Online and Offline Chinese Handwriting Databases",
    authors="Cheng-Lin Liu, Fei Yin, Da-Han Wang, Qiu-Feng Wang",
    year="2011")
