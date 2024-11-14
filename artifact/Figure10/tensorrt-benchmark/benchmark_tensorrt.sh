# !/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

MODEL_PATH=/home/aiscuser/cy/Ladder/artifact/models
TRT_EXEC_PATH=$(pwd)/../../baseline_framework/TensorRT-9.0.1.4/bin
export LD_LIBRARY_PATH=$TRT_EXEC_PATH/../lib:$LD_LIBRARY_PATH
echo "[TENSORRT] Using checkpoint path: $CHECKPOINT_PATH"
LADDER_LOG_PATH="$CHECKPOINT_PATH/onnxruntime/logs"


mkdir -p logs

$TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/llama3/llama3_8b_layer1_seq4096_bs1_kv4096/model.onnx --saveEngine=$MODEL_PATH/llama3/llama3_8b_layer1_seq4096_bs1_kv4096/model.trt --fp16 --workspace=8192 | tee logs/llama3-8b-layer1-seq4096-bs1-kv4096.log

$TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/llama3/llama3_8b_layer1_seq1_bs1_kv8192/model.onnx --saveEngine=$MODEL_PATH/llama3/llama3_8b_layer1_seq1_bs1_kv8192/model.trt --fp16 --workspace=8192 | tee logs/llama3-8b-layer1-seq1-bs1-kv8192.log

$TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/llama3/llama3_8b_layer1_seq1_bs16_kv8192/model.onnx --saveEngine=$MODEL_PATH/llama3/llama3_8b_layer1_seq1_bs16_kv8192/model.trt --fp16 --workspace=8192 | tee logs/llama3-8b-layer1-seq1-bs16-kv8192.log

$TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/llama3/llama3_70b_layer1_seq4096_bs1_kv4096/model.onnx --saveEngine=$MODEL_PATH/llama3/llama3_70b_layer1_seq4096_bs1_kv4096/model.trt --fp16 --workspace=8192 | tee logs/llama3-70b-layer1-seq4096-bs1-kv4096.log

$TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/llama3/llama3_70b_layer1_seq1_bs1_kv8192/model.onnx --saveEngine=$MODEL_PATH/llama3/llama3_70b_layer1_seq1_bs1_kv8192/model.trt --fp16 --workspace=8192 | tee logs/llama3-70b-layer1-seq1-bs1-kv8192.log

$TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/llama3/llama3_70b_layer1_seq1_bs16_kv8192/model.onnx --saveEngine=$MODEL_PATH/llama3/llama3_70b_layer1_seq1_bs16_kv8192/model.trt --fp16 --workspace=8192 | tee logs/llama3-70b-layer1-seq1-bs16-kv8192.log


# $TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/resnet-50-b1/model.onnx --saveEngine=$MODEL_PATH/resnet-50-b1/model.trt --fp16 --workspace=8192 | tee logs/resnet-50-b1.log

# $TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/shufflenet-b1/model.onnx --saveEngine=$MODEL_PATH/shufflenet-b1/model.trt --fp16 --workspace=8192 | tee logs/shufflenet-b1.log

# $TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/Conformer-b1/model.onnx --saveEngine=$MODEL_PATH/Conformer-b1/model.trt --fp16 --workspace=8192 | tee logs/Conformer-b1.log

# $TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/vit-b1/model.onnx --saveEngine=$MODEL_PATH/vit-b1/model.trt --fp16 --workspace=8192 | tee logs/vit-b1.log

# $TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/resnet-50-b128/model.onnx --saveEngine=$MODEL_PATH/resnet-50-b128/model.trt --fp16 --workspace=8192 | tee logs/resnet-50-b128.log

# $TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/shufflenet-b128/model.onnx --saveEngine=$MODEL_PATH/shufflenet-b128/model.trt --fp16 --workspace=8192 | tee logs/shufflenet-b128.log

# $TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/Conformer-b128/model.onnx --saveEngine=$MODEL_PATH/Conformer-b128/model.trt --fp16 --workspace=8192 | tee logs/Conformer-b128.log

# $TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/vit-b128/model.onnx --saveEngine=$MODEL_PATH/vit-b128/model.trt --fp16 --workspace=8192 | tee logs/vit-b128.log

# # large languange models

# $TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/bloom_176b/bloom-176b_layer1_seq1_bs1/model.onnx --saveEngine=$MODEL_PATH/bloom_176b/bloom-176b_layer1_seq1_bs1/model.trt --fp16 --workspace=8192 | tee logs/bloom-176b-layer1-seq1-bs1.log

# $TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/bloom_176b/bloom-176b_layer1_seq1_bs32/model.onnx --saveEngine=$MODEL_PATH/bloom_176b/bloom-176b_layer1_seq1_bs32/model.trt --fp16 --workspace=8192 | tee logs/bloom-176b-layer1-seq1-bs32.log

# $TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/bloom_176b/bloom-176b_layer1_seq4096_bs1/model.onnx --saveEngine=$MODEL_PATH/bloom_176b/bloom-176b_layer1_seq4096_bs1/model.trt --fp16 --workspace=8192 | tee logs/bloom-176b-layer1-seq4096-bs1.log

# $TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/llama_70b/llama2_70b_layer1_seq1_bs1/model.onnx --saveEngine=$MODEL_PATH/llama_70b/llama2_70b_layer1_seq1_bs1/model.trt --fp16 --workspace=8192 | tee logs/llama-70b-layer1-seq1-bs1.log

# $TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/llama_70b/llama2_70b_layer1_seq1_bs32/model.onnx --saveEngine=$MODEL_PATH/llama_70b/llama2_70b_layer1_seq1_bs32/model.trt --fp16 --workspace=8192 | tee logs/llama-70b-layer1-seq1-bs32.log

# $TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/llama_70b/llama2_70b_layer1_seq4096_bs1/model.onnx --saveEngine=$MODEL_PATH/llama_70b/llama2_70b_layer1_seq4096_bs1/model.trt --fp16 --workspace=8192 | tee logs/llama-70b-layer1-seq4096-bs1.log
