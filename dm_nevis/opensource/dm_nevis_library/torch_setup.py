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
"""Setup script to install the experiments_torch library."""
import setuptools

with open("requirements.txt", "r") as f:
  dependencies = list(map(lambda x: x.strip(), f.readlines()))

setuptools.setup(
    name="experiments_torch",
    version="0.1",
    author="DeepMind LSCL Team",
    author_email="nevis@deepmind.com",
    description="Nevis experiments in PyTorch.",
    long_description="Nevis experiments in PyTorch.",
    long_description_content_type="text/markdown",
    url="https://github.com/deepmind/dm_nevis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
    install_requires=dependencies)
