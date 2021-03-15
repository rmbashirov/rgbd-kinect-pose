#!/usr/bin/env bash

source ../source.sh
docker build -t $NAME_01 -f Dockerfile .
CONTAINER_ID=$(docker run -ti -d $NAME_01)
docker exec -ti $CONTAINER_ID sh -c "./install_k4abt.sh"
docker commit $CONTAINER_ID $NAME_02
docker rm -f $CONTAINER_ID
