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

"""File containing all the paths."""
# pylint: disable=line-too-long
import os
LSCL_DATA_DIR = os.environ.get('LSCL_DATA_DIR', '/tmp/nevis_data_dir')
LSCL_RAW_DATA_DIR = os.environ.get('LSCL_RAW_DATA_DIR', '/tmp/nevis_raw_data_dir')
METADATA_FNAME = 'metadata'
STATUS_FNAME = 'status'
STABLE_DIR_NAME = 'stable'
