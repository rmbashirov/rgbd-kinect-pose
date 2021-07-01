#!/usr/bin/env bash

source source.sh

VOLUMES="-v /home/renat:/home/renat -v $PWD/../src:/src"

# ensure nvidia is your default runtime
docker run -ti $PARAMS $VOLUMES $NAME_03 $@
