#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2

export PYTHONPATH=`pwd`

python ./tools/deployment/pytorch2onnx.py $CONFIG $CHECKPOINT --verify
