# !/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# # install tensorrt
# apt install -y wget
# mkdir -p ./baseline_framework
# # Download and install the NVIDIA repository key
# wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/9.0.1/tars/TensorRT-9.0.1.4.Linux.x86_64-gnu.cuda-12.2.tar.gz -O ./baseline_framework/TensorRT-9.0.1.4.Linux.x86_64-gnu.cuda-12.2.tar.gz 
# wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.1.0/tars/TensorRT-10.1.0.27.Linux.x86_64-gnu.cuda-12.4.tar.gz -O ./baseline_framework/TensorRT-10.1.0.27.Linux.x86_64-gnu.cuda-12.4.tar.gz
# # Tar the file
# cd ./baseline_framework
# tar -xvzf TensorRT-9.0.1.4.Linux.x86_64-gnu.cuda-12.2.tar.gz 
# tar -xvzf TensorRT-10.1.0.27.Linux.x86_64-gnu.cuda-12.4.tar.gz
# cd ..



MODEL_PATH=/home/aiscuser/cy/Ladder/artifact/models
# TRT_EXEC_PATH=$(pwd)/baseline_framework/TensorRT-9.0.1.4/bin
TRT_EXEC_PATH=$(pwd)/baseline_framework/TensorRT-10.1.0.27/bin
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

$TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/dit/dit_XL2_layer1_n16_img256/model.onnx --saveEngine=$MODEL_PATH/dit/dit_XL2_layer1_n16_img256/model.trt --fp16 | tee logs/dit-xl2-layer1-n16-img256.log
$TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/dit/dit_XL2_layer1_n16_img512/model.onnx --saveEngine=$MODEL_PATH/dit/dit_XL2_layer1_n16_img512/model.trt --fp16 | tee logs/dit-xl2-layer1-n16-img512.log
$TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/dit/dit_XL2_layer1_n64_img256/model.onnx --saveEngine=$MODEL_PATH/dit/dit_XL2_layer1_n64_img256/model.trt --fp16 | tee logs/dit-xl2-layer1-n64-img256.log
$TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/dit/dit_XL2_layer1_n64_img512/model.onnx --saveEngine=$MODEL_PATH/dit/dit_XL2_layer1_n64_img512/model.trt --fp16 | tee logs/dit-xl2-layer1-n64-img512.log

$TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/phimoe/phimoe_3.5b_layer1_seq4096_bs1_kv4096/model.onnx --saveEngine=$MODEL_PATH/phimoe/phimoe_3.5b_layer1_seq4096_bs1_kv4096/model.trt --fp16 --workspace=8192 | tee logs/phimoe-3.5b-layer1-seq4096-bs1-kv4096.log
$TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/phimoe/phimoe_3.5b_layer1_seq1_bs1_kv8192/model.onnx --saveEngine=$MODEL_PATH/phimoe/phimoe_3.5b_layer1_seq1_bs1_kv8192/model.trt --fp16 --workspace=8192 | tee logs/phimoe-3.5b-layer1-seq1-bs1-kv8192.log
$TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/phimoe/phimoe_3.5b_layer1_seq1_bs16_kv8192/model.onnx --saveEngine=$MODEL_PATH/phimoe/phimoe_3.5b_layer1_seq1_bs16_kv8192/model.trt --fp16 --workspace=8192 | tee logs/phimoe-3.5b-layer1-seq1-bs16-kv8192.log

$TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/retnet/retnet_3b_layer1_seq4096_bs1/model.onnx --saveEngine=$MODEL_PATH/retnet/retnet_3b_layer1_seq4096_bs1/model.trt --fp16 | tee logs/retnet-3b-layer1-seq4096-bs1.log
$TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/retnet/retnet_3b_layer1_seq1_bs1/model.onnx --saveEngine=$MODEL_PATH/retnet/retnet_3b_layer1_seq1_bs1/model.trt --fp16 | tee logs/retnet-3b-layer1-seq1-bs1.log
$TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/retnet/retnet_3b_layer1_seq1_bs16/model.onnx --saveEngine=$MODEL_PATH/retnet/retnet_3b_layer1_seq1_bs16/model.trt --fp16 | tee logs/retnet-3b-layer1-seq1-bs16.log
$TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/retnet/retnet_65b_layer1_seq4096_bs1/model.onnx --saveEngine=$MODEL_PATH/retnet/retnet_65b_layer1_seq4096_bs1/model.trt --fp16 | tee logs/retnet-65b-layer1-seq4096-bs1.log
$TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/retnet/retnet_65b_layer1_seq1_bs1/model.onnx --saveEngine=$MODEL_PATH/retnet/retnet_65b_layer1_seq1_bs1/model.trt --fp16 | tee logs/retnet-65b-layer1-seq1-bs1.log
$TRT_EXEC_PATH/trtexec --onnx=$MODEL_PATH/retnet/retnet_65b_layer1_seq1_bs16/model.onnx --saveEngine=$MODEL_PATH/retnet/retnet_65b_layer1_seq1_bs16/model.trt --fp16 | tee logs/retnet-65b-layer1-seq1-bs16.log