#!/usr/bin/env bash

python3.10 -m pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet