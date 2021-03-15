#!/usr/bin/env bash

# $1: path to log dir
python -m multiprocessing_pipeline.view_log \
  --log_dirpath $1
