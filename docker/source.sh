#!/usr/bin/env bash

PARAMS="--net=host --ipc=host -u $(id -u ${USER}):$(id -g ${USER}) --privileged -e DISPLAY=$DISPLAY"
NAME_01="azure_kinect:01_base"
NAME_02="azure_kinect:02_eula"
NAME_03="azure_kinect:03_final"
