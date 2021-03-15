#!/usr/bin/env bash

python -m pyk4a.viewer \
  --dump_filepath ~/Desktop/dump.pickle \
  --dump_frames 300 \
  --fps 30 \
  --no_depth \
  --parallel_bt
