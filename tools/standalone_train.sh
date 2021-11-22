#!/usr/bin/env bash
###
 # @Author       : jiangtao
 # @Date         : 2021-11-22 10:30:59
 # @Email        : jiangtaoo2333@163.com
 # @LastEditTime : 2021-11-22 15:26:14
 # @Description  : 
### 

set -x

CONFIG=$1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

python -u tools/train.py ${CONFIG} --launcher="none" 
