# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

export LADDER_HOME=$(pwd)/../..
export LADDER_TVM_HOME=$LADDER_HOME/3rdparty/tvm
export LADDER_CUTLASS_HOME=$LADDER_HOME/3rdparty/cutlass
export PYTHONPATH=$LADDER_HOME/python
export PYTHONPATH=$LADDER_TVM_HOME/python:$PYTHONPATH
export CPLUS_INCLUDE_PATH=$LADDER_CUTLASS_HOME/include

echo "[LADDER] Using checkpoint path: $CHECKPOINT_PATH"
LADDER_LOG_PATH="$CHECKPOINT_PATH/ladder/logs"

MODEL_PATH=$(pwd)/../models

mkdir -p logs/resnet

python -u ./ladder_resnet.py --cublas --async_propagation --prefix resnet18 --batch 1 --fake_quant -1 2>&1 | tee logs/resnet/resnet_b1_fp16.log
python -u ./ladder_resnet.py --cublas --async_propagation --prefix resnet18 --batch 128 --fake_quant -1 2>&1 | tee logs/resnet/resnet_b128_fp16.log