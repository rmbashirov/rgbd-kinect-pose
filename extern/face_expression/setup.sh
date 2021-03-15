#!/usr/bin/env bash

pip uninstall -y face_expression
rm -rf build dist face_expression.egg-info
find . -name '*.pyc' -delete
python setup.py install --force