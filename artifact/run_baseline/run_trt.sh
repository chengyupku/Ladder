# !/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# install tensorrt
apt install -y wget
mkdir -p ./baseline_framework
# Download and install the NVIDIA repository key
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/9.0.1/tars/TensorRT-9.0.1.4.Linux.x86_64-gnu.cuda-12.2.tar.gz -O ./baseline_framework/TensorRT-9.0.1.4.Linux.x86_64-gnu.cuda-12.2.tar.gz 
# Tar the file
cd ./baseline_framework
tar -xvzf TensorRT-9.0.1.4.Linux.x86_64-gnu.cuda-12.2.tar.gz 
cd ..



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