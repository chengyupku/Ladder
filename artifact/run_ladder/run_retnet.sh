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

mkdir -p logs/retnet

python -u ./ladder_retnet.py --cublas --async_propagation --prefix retnet_3b --batch 1 --seqlen 4096 --fake_quant -1 2>&1 | tee logs/retnet/retnet_3b_b1_seq4096_fp16.log
python -u ./ladder_retnet.py --cublas --async_propagation --prefix retnet_65b --batch 1 --seqlen 4096 --fake_quant -1 2>&1 | tee logs/retnet/retnet_65b_b1_seq4096_fp16.log