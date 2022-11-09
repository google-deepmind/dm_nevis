#!/bin/bash
set -euo pipefail

# A script to create a virtualenv, install dependencies, and run all tests.

# TODO: Don't create the env from scratch in the future.
# We do this at the moment due to (what appears to be) a bug in pip that causes
# our packages to become corrupt when re-installing the dependencies.
rm -rf env

python3 -m venv env
source env/bin/activate

pip install \
  -r requirements.txt \
  -r experiments_jax/requirements.txt \
  -r experiments_torch/requirements.txt \
  -e . \
  pytest

pytest dm_nevis/benchmarker
pytest dm_nevis/streams
pytest experiments_jax
pytest experiments_torch

echo ✅ All tests passed
