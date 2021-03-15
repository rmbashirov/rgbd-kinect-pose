#!/usr/bin/env bash

WANDB_API_KEY="e090e0521b859c1a50c9514e509ea5f12c8351df"
WORKDIR="/Vol0/user/k.iskakov/dev/face_expression"
PORT=8087

PARAMS="-p ${PORT}:${PORT} -w ${WORKDIR} -e WANDB_API_KEY=${WANDB_API_KEY} --net=host --ipc=host -u $(id -u ${USER}):$(id -g ${USER})"
NAME="face_expression-karfly"
HEAD_NAME="airuhead01:5000/${NAME}"
VOLUMES="-v /Vol0:/Vol0 -v /Vol1:/Vol1"
