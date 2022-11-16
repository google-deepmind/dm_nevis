#!/bin/bash
set -euo pipefail

usage() {
  echo "Usage:"
  echo -e "\t-d <DATASET_NAME> | Dataset name"
  echo -e "\t-s <STREAM_NAME>  | Stream variant name (FULL|SHORT|TINY|DEBUG|...)"
  echo -e "\t-b                | Build docker before running"
  echo -e "\t-e                | Develop mode where code is mounted"
  echo -e "\t-h                | Help message"
  exit $1
}

DATASET_NAME=''
STREAM_NAME=''
SHOULD_BUILD_DOCKER=false
DEVELOP_MODE=false

while getopts ":hd:s:be" arg; do
  case $arg in
    d)
      DATASET_NAME=${OPTARG}
      ;;
    s) 
      STREAM_NAME=${OPTARG}
      ;;
    b)
      SHOULD_BUILD_DOCKER=true
      ;;
    e)
      DEVELOP_MODE=true
      ;;
    h | *) # Display help.
      usage 0
      ;;
  esac
done

eval LOCAL_DIR="~/nevis"
eval KAGGLE_CRED_DIR="~/.kaggle"
NEVIS_CODE_DIR=`realpath ./dm_nevis`
NEVIS_DATA_DIR="/tmp/nevis_data_dir"
NEVIS_RAW_DATA_DIR="/tmp/nevis_raw_data_dir"

if $DEVELOP_MODE ; then
  MOUNT_MAPPING="-v ${NEVIS_CODE_DIR}:/root/dm_nevis"
  IMG="nevis-data-dev"
else
  MOUNT_MAPPING=""
  IMG="nevis-data"
fi

if $SHOULD_BUILD_DOCKER ; then
  docker build \
    -f dockerfile.data . -t "$IMG" --network=host
fi


docker run \
  -v ${LOCAL_DIR}:/tmp ${MOUNT_MAPPING} \
  -v ${KAGGLE_CRED_DIR}:/root/.kaggle \
  --env NEVIS_DATA_DIR=${NEVIS_DATA_DIR} \
  --env NEVIS_RAW_DATA_DIR=${NEVIS_RAW_DATA_DIR} \
  -it "$IMG" \
  --try_download_artifacts_from_urls \
  --write_stable_version \
  --local_download_dir=/tmp/datasets \
  --dataset=$DATASET_NAME --stream=$STREAM_NAME
