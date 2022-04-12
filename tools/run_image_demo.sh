#!/usr/bin/env bash
CONFIG=$1
CHECKPOINT=$2

export PYTHONPATH=`pwd`
echo $PYTHONPATH
python ./demo/inference_by_jiangtao_face.py $CONFIG $CHECKPOINT