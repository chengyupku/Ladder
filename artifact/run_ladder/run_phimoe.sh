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

mkdir -p logs/phimoe

python -u ./ladder_phimoe.py --cublas --async_propagation --prefix phimoe_3.5b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant -1 2>&1 | tee logs/phimoe/phimoe3.5b_b1_s4096_kv4096_fp16.log