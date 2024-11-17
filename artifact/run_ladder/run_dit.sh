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

mkdir -p logs/dit

# DiT-XL/2, n=16, image_size=256x256, fp16
python -u ./ladder_dit.py --cublas --async_propagation --prefix dit_XL2 --n 16 --img 256 --fake_quant -1 2>&1 | tee logs/dit/dit_XL2_n16_img256_fp16.log
python -u ./ladder_dit.py --cublas --async_propagation --prefix dit_XL2 --n 16 --img 256 --fake_quant -1 --no_attn 2>&1 | tee logs/dit/dit_XL2_n16_img256_fp16_no_attn.log

# DiT-XL/2, n=16, image_size=512x512, fp16
python -u ./ladder_dit.py --cublas --async_propagation --prefix dit_XL2 --n 16 --img 512 --fake_quant -1 2>&1 | tee logs/dit/dit_XL2_n16_img512_fp16.log
python -u ./ladder_dit.py --cublas --async_propagation --prefix dit_XL2 --n 16 --img 512 --fake_quant -1 --no_attn 2>&1 | tee logs/dit/dit_XL2_n16_img512_fp16_no_attn.log

# DiT-XL/2, n=64, image_size=256x256, fp16
python -u ./ladder_dit.py --cublas --async_propagation --prefix dit_XL2 --n 64 --img 256 --fake_quant -1 2>&1 | tee logs/dit/dit_XL2_n64_img256_fp16.log
python -u ./ladder_dit.py --cublas --async_propagation --prefix dit_XL2 --n 64 --img 256 --fake_quant -1 --no_attn 2>&1 | tee logs/dit/dit_XL2_n64_img256_fp16_no_attn.log

# DiT-XL/2, n=64, image_size=512x512, fp16
python -u ./ladder_dit.py --cublas --async_propagation --prefix dit_XL2 --n 64 --img 512 --fake_quant -1 2>&1 | tee logs/dit/dit_XL2_n64_img512_fp16.log
python -u ./ladder_dit.py --cublas --async_propagation --prefix dit_XL2 --n 64 --img 512 --fake_quant -1 --no_attn 2>&1 | tee logs/dit/dit_XL2_n64_img512_fp16_no_attn.log

# # DiT-XL/2, n=16, image_size=256x256, fp4
# python -u ./ladder_dit.py --cublas --async_propagation --prefix dit_XL2 --n 16 --img 256 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/dit/dit_XL2_n16_img256_fp4.log

# # DiT-XL/2, n=16, image_size=512x512, fp4
# python -u ./ladder_dit.py --cublas --async_propagation --prefix dit_XL2 --n 16 --img 512 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/dit/dit_XL2_n16_img512_fp4.log

# # DiT-XL/2, n=64, image_size=256x256, fp4
# python -u ./ladder_dit.py --cublas --async_propagation --prefix dit_XL2 --n 64 --img 256 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/dit/dit_XL2_n64_img256_fp4.log

# # DiT-XL/2, n=64, image_size=512x512, fp4
# python -u ./ladder_dit.py --cublas --async_propagation --prefix dit_XL2 --n 64 --img 512 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/dit/dit_XL2_n64_img512_fp4.log