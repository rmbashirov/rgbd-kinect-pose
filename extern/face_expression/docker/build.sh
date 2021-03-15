#!/usr/bin/env bash

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ${CURRENT_DIR}/source.sh

docker build -t $NAME -f ${CURRENT_DIR}/Dockerfile ${CURRENT_DIR}/..
