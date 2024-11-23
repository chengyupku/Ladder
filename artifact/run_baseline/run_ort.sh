# #!/bin/bash

CUDNN_PATH="/home/aiscuser/miniconda3/envs/cybb/lib/python3.9/site-packages/nvidia/cudnn/lib"
if [[ ":$LD_LIBRARY_PATH:" != *":$CUDNN_PATH:"* ]]; then
    export LD_LIBRARY_PATH="$CUDNN_PATH:$LD_LIBRARY_PATH"
fi

MODEL_PREFIX="/home/aiscuser/cy/Ladder/artifact/models/llama3"

python run_onnxrt.py --prefix ${MODEL_PREFIX}/llama3_8b_layer1_seq4096_bs1_kv4096
python run_onnxrt.py --prefix ${MODEL_PREFIX}/llama3_8b_layer1_seq1_bs1_kv8192
python run_onnxrt.py --prefix ${MODEL_PREFIX}/llama3_8b_layer1_seq1_bs16_kv8192
python run_onnxrt.py --prefix ${MODEL_PREFIX}/llama3_70b_layer1_seq4096_bs1_kv4096
python run_onnxrt.py --prefix ${MODEL_PREFIX}/llama3_70b_layer1_seq1_bs1_kv8192
python run_onnxrt.py --prefix ${MODEL_PREFIX}/llama3_70b_layer1_seq1_bs16_kv8192

MODEL_PREFIX="/home/aiscuser/cy/Ladder/artifact/models/dit"

python run_onnxrt.py --prefix ${MODEL_PREFIX}/dit_XL2_layer1_n16_img256
python run_onnxrt.py --prefix ${MODEL_PREFIX}/dit_XL2_layer1_n16_img512
python run_onnxrt.py --prefix ${MODEL_PREFIX}/dit_XL2_layer1_n64_img256
python run_onnxrt.py --prefix ${MODEL_PREFIX}/dit_XL2_layer1_n64_img512

MODEL_PREFIX="/home/aiscuser/cy/Ladder/artifact/models/mamba2"

python run_onnxrt.py --prefix ${MODEL_PREFIX}/mamba2_1.3b_layer1_seq1024_bs1

MODEL_PREFIX="/home/aiscuser/cy/Ladder/artifact/models/phimoe"

python run_onnxrt.py --prefix ${MODEL_PREFIX}/phimoe_3.5b_layer1_seq4096_bs1_kv4096
python run_onnxrt.py --prefix ${MODEL_PREFIX}/phimoe_3.5b_layer1_seq1_bs1_kv8192
python run_onnxrt.py --prefix ${MODEL_PREFIX}/phimoe_3.5b_layer1_seq1_bs16_kv8192

MODEL_PREFIX="/home/aiscuser/cy/Ladder/artifact/models/retnet"

python run_onnxrt.py --prefix ${MODEL_PREFIX}/retnet_3b_layer1_seq4096_bs1
python run_onnxrt.py --prefix ${MODEL_PREFIX}/retnet_3b_layer1_seq1_bs1
python run_onnxrt.py --prefix ${MODEL_PREFIX}/retnet_3b_layer1_seq1_bs16
python run_onnxrt.py --prefix ${MODEL_PREFIX}/retnet_65b_layer1_seq4096_bs1
python run_onnxrt.py --prefix ${MODEL_PREFIX}/retnet_65b_layer1_seq1_bs1
python run_onnxrt.py --prefix ${MODEL_PREFIX}/retnet_65b_layer1_seq1_bs16

MODEL_PREFIX="/home/aiscuser/cy/Ladder/artifact/models/resnet"

python run_onnxrt.py --prefix ${MODEL_PREFIX}/resnet18_bs1
python run_onnxrt.py --prefix ${MODEL_PREFIX}/resnet18_bs128