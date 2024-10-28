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

mkdir -p logs/llama3

# llama3_8b, batch 1, seqlen_q 1, seqlen_kv 8192
# python -u ./ladder_llama3.py --cublas --async_propagation --batch 1 --seqlen_q 1 --seqlen_kv 8192 --fake_quant -1 2>&1 | tee logs/llama3/llama3_8b_b1_s1_kv8192_q-1.log
# python -u ./ladder_llama3.py --cublas --async_propagation --batch 1 --seqlen_q 1 --seqlen_kv 8192 --no_attn --fake_quant -1 2>&1 | tee logs/llama3/llama3_8b_b1_s1_kv8192_q-1_no_attn.log
# python -u ./ladder_llama3.py --cublas --async_propagation --batch 1 --seqlen_q 1 --seqlen_kv 8192 --single_attn --fake_quant -1 2>&1 | tee logs/llama3/llama3_8b_b1_s1_kv8192_q-1_single_attn.log

# llama3_8b, batch 64, seqlen_q 1, seqlen_kv 8192
# python -u ./ladder_llama3.py --cublas --async_propagation --batch 64 --seqlen_q 1 --seqlen_kv 8192 --fake_quant -1 2>&1 | tee logs/llama3/llama3_8b_b64_s1_kv8192_q-1.log
# python -u ./ladder_llama3.py --cublas --async_propagation --batch 1 --seqlen_q 1 --seqlen_kv 8192 --no_attn --fake_quant -1 2>&1 | tee logs/llama3/llama3_8b_b64_s1_kv8192_q-1_no_attn.log
# python -u ./ladder_llama3.py --cublas --async_propagation --batch 1 --seqlen_q 1 --seqlen_kv 8192 --single_attn --fake_quant -1 2>&1 | tee logs/llama3/llama3_8b_b64_s1_kv8192_q-1_single_attn.log

# llama3_70b, batch 1, seqlen_q 1, seqlen_kv 8192
# python -u ./ladder_llama3.py --cublas --async_propagation --batch 1 --seqlen_q 1 --seqlen_kv 8192 --fake_quant -1 2>&1 | tee logs/llama3/llama3_70b_b1_s1_kv8192_q-1.log
# python -u ./ladder_llama3.py --cublas --async_propagation --batch 1 --seqlen_q 1 --seqlen_kv 8192 --no_attn --fake_quant -1 2>&1 | tee logs/llama3/llama3_70b_b1_s1_kv8192_q-1_no_attn.log
# python -u ./ladder_llama3.py --cublas --async_propagation --batch 1 --seqlen_q 1 --seqlen_kv 8192 --single_attn --fake_quant -1 2>&1 | tee logs/llama3/llama3_70b_b1_s1_kv8192_q-1_single_attn.log

# llama3_70b, batch 16, seqlen_q 1, seqlen_kv 8192
python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 16 --seqlen_q 1 --seqlen_kv 8192 --fake_quant -1 2>&1 | tee logs/llama3/llama3_70b_b16_s1_kv8192_q-1.log
# python -u ./ladder_llama3.py --cublas --async_propagation --batch 16 --seqlen_q 1 --seqlen_kv 8192 --no_attn --fake_quant -1 2>&1 | tee logs/llama3/llama3_70b_b16_s1_kv8192_q-1_no_attn.log
# python -u ./ladder_llama3.py --cublas --async_propagation --batch 16 --seqlen_q 1 --seqlen_kv 8192 --single_attn --fake_quant -1 2>&1 | tee logs/llama3/llama3_70b_b16_s1_kv8192_q-1_single_attn.log