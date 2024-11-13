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

# llama3_8b, batch 1, seqlen_q 4096, seqlen_kv 4096, fp16
python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant -1 2>&1 | tee logs/llama3/llama3_8b_b1_s4096_kv4096_fp16.log
python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant -1 --no_attn 2>&1 | tee logs/llama3/llama3_8b_b1_s4096_kv4096_fp16_no_attn.log

# llama3_8b, batch 1, seqlen_q 1, seqlen_kv 8192, fp16
python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 1 --seqlen_q 1 --seqlen_kv 8192 --fake_quant -1 2>&1 | tee logs/llama3/llama3_8b_b1_s1_kv8192_fp16.log
python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 1 --seqlen_q 1 --seqlen_kv 8192 --fake_quant -1 --no_attn 2>&1 | tee logs/llama3/llama3_8b_b1_s1_kv8192_fp16_no_attn.log

# llama3_8b, batch 16, seqlen_q 1, seqlen_kv 8192, fp16
python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 16 --seqlen_q 1 --seqlen_kv 8192 --fake_quant -1 2>&1 | tee logs/llama3/llama3_8b_b16_s1_kv8192_fp16.log
python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 16 --seqlen_q 1 --seqlen_kv 8192 --fake_quant -1 --no_attn 2>&1 | tee logs/llama3/llama3_8b_b16_s1_kv8192_fp16_no_attn.log

# llama3_70b, batch 1, seqlen_q 4096, seqlen_kv 4096, fp16
python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant -1 2>&1 | tee logs/llama3/llama3_70b_b1_s4096_kv4096_fp16.log
python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant -1 --no_attn 2>&1 | tee logs/llama3/llama3_70b_b1_s4096_kv4096_fp16_no_attn.log

# llama3_70b, batch 1, seqlen_q 1, seqlen_kv 8192, fp16
python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 1 --seqlen_q 1 --seqlen_kv 8192 --fake_quant -1 2>&1 | tee logs/llama3/llama3_70b_b1_s1_kv8192_fp16.log
python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 1 --seqlen_q 1 --seqlen_kv 8192 --fake_quant -1 --no_attn 2>&1 | tee logs/llama3/llama3_70b_b1_s1_kv8192_fp16_no_attn.log

# llama3_70b, batch 16, seqlen_q 1, seqlen_kv 8192, fp16
python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 16 --seqlen_q 1 --seqlen_kv 8192 --fake_quant -1 2>&1 | tee logs/llama3/llama3_70b_b16_s1_kv8192_fp16.log
python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 16 --seqlen_q 1 --seqlen_kv 8192 --fake_quant -1 --no_attn 2>&1 | tee logs/llama3/llama3_70b_b16_s1_kv8192_fp16_no_attn.log

# llama3_8b, batch 1, seqlen_q 4096, seqlen_kv 4096, fp4
python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/llama3/llama3_8b_b1_s4096_kv4096_fp4.log
python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant 0 --bits 4 --format nf --no_attn 2>&1 | tee logs/llama3/llama3_8b_b1_s4096_kv4096_fp4_no_attn.log

# llama3_8b, batch 1, seqlen_q 1, seqlen_kv 8192, fp4
python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 1 --seqlen_q 1 --seqlen_kv 8192 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/llama3/llama3_8b_b1_s1_kv8192_fp4.log
python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 1 --seqlen_q 1 --seqlen_kv 8192 --fake_quant 0 --bits 4 --format nf --no_attn 2>&1 | tee logs/llama3/llama3_8b_b1_s1_kv8192_fp4_no_attn.log

# llama3_8b, batch 16, seqlen_q 1, seqlen_kv 8192, fp4
python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 16 --seqlen_q 1 --seqlen_kv 8192 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/llama3/llama3_8b_b16_s1_kv8192_fp4.log
python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 16 --seqlen_q 1 --seqlen_kv 8192 --fake_quant 0 --bits 4 --format nf --no_attn 2>&1 | tee logs/llama3/llama3_8b_b16_s1_kv8192_fp4_no_attn.log

# llama3_70b, batch 1, seqlen_q 4096, seqlen_kv 4096, fp4
python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/llama3/llama3_70b_b1_s4096_kv4096_fp4.log
python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant 0 --bits 4 --format nf --no_attn 2>&1 | tee logs/llama3/llama3_70b_b1_s4096_kv4096_fp4_no_attn.log

# llama3_70b, batch 1, seqlen_q 1, seqlen_kv 8192, fp4
python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 1 --seqlen_q 1 --seqlen_kv 8192 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/llama3/llama3_70b_b1_s1_kv8192_fp4.log
python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 1 --seqlen_q 1 --seqlen_kv 8192 --fake_quant 0 --bits 4 --format nf --no_attn 2>&1 | tee logs/llama3/llama3_70b_b1_s1_kv8192_fp4_no_attn.log

# llama3_70b, batch 16, seqlen_q 1, seqlen_kv 8192, fp4
python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 16 --seqlen_q 1 --seqlen_kv 8192 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/llama3/llama3_70b_b16_s1_kv8192_fp4.log
python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 16 --seqlen_q 1 --seqlen_kv 8192 --fake_quant 0 --bits 4 --format nf --no_attn 2>&1 | tee logs/llama3/llama3_70b_b16_s1_kv8192_fp4_no_attn.log

# llama3_8b, batch 1, seqlen_q 4096, seqlen_kv 4096, fp16
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant -1 2>&1 | tee logs/llama3/single_attention.log
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant -1 2>&1 | tee logs/llama3/llama3_8b_b1_s4096_kv4096_fp16.log
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant -1 --no_attn 2>&1 | tee logs/llama3/llama3_8b_b1_s4096_kv4096_fp16_no_attn.log
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant -1 --single_attn 2>&1 | tee logs/llama3/llama3_8b_b1_s4096_kv4096_fp16_single_attn.log

# # llama3_8b, batch 1, seqlen_q 1, seqlen_kv 8192, fp16
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 1 --seqlen_q 1 --seqlen_kv 8192 --fake_quant -1 2>&1 | tee logs/llama3/llama3_8b_b1_s1_kv8192_fp16.log

# # llama3_8b, batch 16, seqlen_q 1, seqlen_kv 8192, fp16
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 16 --seqlen_q 1 --seqlen_kv 8192 --fake_quant -1 2>&1 | tee logs/llama3/llama3_8b_b16_s1_kv8192_fp16.log

# # llama3_70b, batch 1, seqlen_q 4096, seqlen_kv 4096, fp16
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant -1 2>&1 | tee logs/llama3/llama3_70b_b1_s4096_kv4096_fp16.log

# # llama3_70b, batch 1, seqlen_q 1, seqlen_kv 8192, fp16
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 1 --seqlen_q 1 --seqlen_kv 8192 --fake_quant -1 2>&1 | tee logs/llama3/llama3_70b_b1_s1_kv8192_fp16.log

# # # llama3_70b, batch 16, seqlen_q 1, seqlen_kv 8192, fp16
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 16 --seqlen_q 1 --seqlen_kv 8192 --fake_quant -1 2>&1 | tee logs/llama3/llama3_70b_b16_s1_kv8192_fp16.log

# # llama3_8b, batch 1, seqlen_q 4096, seqlen_kv 4096, fp4
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/llama3/llama3_8b_b1_s4096_kv4096_fp4.log
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant 0 --bits 4 --format nf --use_prebuilt 2>&1 | tee logs/llama3/llama3_8b_b1_s4096_kv4096_fp4.log

# # llama3_8b, batch 1, seqlen_q 1, seqlen_kv 8192, fp4
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 1 --seqlen_q 1 --seqlen_kv 8192 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/llama3/llama3_8b_b1_s1_kv8192_fp4.log
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 1 --seqlen_q 1 --seqlen_kv 8192 --fake_quant 0 --bits 4 --format nf --use_prebuilt 2>&1 | tee logs/llama3/llama3_8b_b1_s1_kv8192_fp4.log

# # llama3_8b, batch 16, seqlen_q 1, seqlen_kv 8192, fp4
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 16 --seqlen_q 1 --seqlen_kv 8192 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/llama3/llama3_8b_b16_s1_kv8192_fp4.log
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 16 --seqlen_q 1 --seqlen_kv 8192 --fake_quant 0 --bits 4 --format nf --use_prebuilt 2>&1 | tee logs/llama3/llama3_8b_b16_s1_kv8192_fp4.log

# # llama3_70b, batch 1, seqlen_q 4096, seqlen_kv 4096, fp4
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/llama3/llama3_70b_b1_s4096_kv4096_fp4.log
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant 0 --bits 4 --format nf --use_prebuilt 2>&1 | tee logs/llama3/llama3_70b_b1_s4096_kv4096_fp4.log

# # llama3_70b, batch 1, seqlen_q 1, seqlen_kv 8192, fp4
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 1 --seqlen_q 1 --seqlen_kv 8192 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/llama3/llama3_70b_b1_s1_kv8192_fp4.log
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 1 --seqlen_q 1 --seqlen_kv 8192 --fake_quant 0 --bits 4 --format nf --use_prebuilt 2>&1 | tee logs/llama3/llama3_70b_b1_s1_kv8192_fp4.log

# # # llama3_70b, batch 16, seqlen_q 1, seqlen_kv 8192, fp4
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 16 --seqlen_q 1 --seqlen_kv 8192 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/llama3/llama3_70b_b16_s1_kv8192_fp4.log
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 16 --seqlen_q 1 --seqlen_kv 8192 --fake_quant 0 --bits 4 --format nf --use_prebuilt  2>&1 | tee logs/llama3/llama3_70b_b16_s1_kv8192_fp4.log

# llama3_8b, batch 1, seqlen_q 4096, seqlen_kv 4096, fp4
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/llama3/llama3_8b_b1_s4096_kv4096_fp4_no_fuse_ladder_gemm.log
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant 0 --bits 4 --format nf --use_prebuilt 2>&1 | tee logs/llama3/llama3_8b_b1_s4096_kv4096_fp4_no_fuse_ladder_gemm.log

# # llama3_8b, batch 1, seqlen_q 1, seqlen_kv 8192, fp4
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 1 --seqlen_q 1 --seqlen_kv 8192 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/llama3/llama3_8b_b1_s1_kv8192_fp4_no_fuse_ladder_gemm.log
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 1 --seqlen_q 1 --seqlen_kv 8192 --fake_quant 0 --bits 4 --format nf --use_prebuilt 2>&1 | tee logs/llama3/llama3_8b_b1_s1_kv8192_fp4_no_fuse_ladder_gemm.log

# # llama3_8b, batch 16, seqlen_q 1, seqlen_kv 8192, fp4
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 16 --seqlen_q 1 --seqlen_kv 8192 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/llama3/llama3_8b_b16_s1_kv8192_fp4_no_fuse_ladder_gemm.log
# # python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 16 --seqlen_q 1 --seqlen_kv 8192 --fake_quant 0 --bits 4 --format nf --use_prebuilt 2>&1 | tee logs/llama3/llama3_8b_b16_s1_kv8192_fp4_no_fuse_ladder_gemm.log

# # llama3_70b, batch 1, seqlen_q 4096, seqlen_kv 4096, fp4
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/llama3/llama3_70b_b1_s4096_kv4096_fp4_no_fuse_ladder_gemm.log
# # python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant 0 --bits 4 --format nf --use_prebuilt 2>&1 | tee logs/llama3/llama3_70b_b1_s4096_kv4096_fp4_no_fuse_ladder_gemm.log

# # llama3_70b, batch 1, seqlen_q 1, seqlen_kv 8192, fp4
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 1 --seqlen_q 1 --seqlen_kv 8192 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/llama3/llama3_70b_b1_s1_kv8192_fp4_no_fuse_ladder_gemm.log
# # python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 1 --seqlen_q 1 --seqlen_kv 8192 --fake_quant 0 --bits 4 --format nf --use_prebuilt 2>&1 | tee logs/llama3/llama3_70b_b1_s1_kv8192_fp4_no_fuse_ladder_gemm.log

# # # llama3_70b, batch 16, seqlen_q 1, seqlen_kv 8192, fp4
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 16 --seqlen_q 1 --seqlen_kv 8192 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/llama3/llama3_70b_b16_s1_kv8192_fp4_no_fuse_ladder_gemm.log
# # python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 16 --seqlen_q 1 --seqlen_kv 8192 --fake_quant 0 --bits 4 --format nf --use_prebuilt 2>&1 | tee logs/llama3/llama3_70b_b16_s1_kv8192_fp4_no_fuse_ladder_gemm.log

# llama3_8b, batch 1, seqlen_q 4096, seqlen_kv 4096
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant 0 --bits 4 --format nf --single_attn --use_prebuilt 2>&1 | tee logs/llama3/llama3_8b_b1_s4096_kv4096_single_attn.log
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant -1 --single_attn --use_prebuilt 2>&1 | tee logs/llama3/llama3_8b_b1_s4096_kv4096_single_attn.log

# # llama3_8b, batch 1, seqlen_q 1, seqlen_kv 8192
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 1 --seqlen_q 1 --seqlen_kv 8192 --fake_quant -1 --single_attn 2>&1 | tee logs/llama3/llama3_8b_b1_s1_kv8192_single_attn.log

# # llama3_8b, batch 16, seqlen_q 1, seqlen_kv 8192
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 16 --seqlen_q 1 --seqlen_kv 8192 --fake_quant -1 --single_attn 2>&1 | tee logs/llama3/llama3_8b_b16_s1_kv8192_single_attn.log

# # llama3_70b, batch 1, seqlen_q 4096, seqlen_kv 4096
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant -1 --single_attn 2>&1 | tee logs/llama3/llama3_70b_b1_s4096_kv4096_single_attn.log

# # llama3_70b, batch 1, seqlen_q 1, seqlen_kv 8192
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 1 --seqlen_q 1 --seqlen_kv 8192 --fake_quant -1 --single_attn 2>&1 | tee logs/llama3/llama3_70b_b1_s1_kv8192_single_attn.log

# # # llama3_70b, batch 16, seqlen_q 1, seqlen_kv 8192
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_70b --batch 16 --seqlen_q 1 --seqlen_kv 8192 --fake_quant -1 --single_attn 2>&1 | tee logs/llama3/llama3_70b_b16_s1_kv8192_single_attn.log

# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant 0 --bits 4 --format nf --no_attn 2>&1 | tee logs/llama3/llama3_8b_b1_s4096_kv4096_no_attn.log
# python -u ./ladder_llama3.py --cublas --async_propagation --prefix llama3_8b --batch 1 --seqlen_q 4096 --seqlen_kv 4096 --fake_quant 0 --bits 4 --format nf --no_attn --use_prebuilt 2>&1 | tee logs/llama3/llama3_8b_b1_s4096_kv4096_no_attn.log