# #!/bin/bash

MODEL_PREFIX="/home/aiscuser/cy/Ladder/artifact/models/llama3"

python run_onnxrt.py --prefix ${MODEL_PREFIX}/llama3_8b_layer1_seq4096_bs1_kv4096
python run_onnxrt.py --prefix ${MODEL_PREFIX}/llama3_8b_layer1_seq1_bs1_kv8192
python run_onnxrt.py --prefix ${MODEL_PREFIX}/llama3_8b_layer1_seq1_bs16_kv8192
python run_onnxrt.py --prefix ${MODEL_PREFIX}/llama3_70b_layer1_seq4096_bs1_kv4096
python run_onnxrt.py --prefix ${MODEL_PREFIX}/llama3_70b_layer1_seq1_bs1_kv8192
python run_onnxrt.py --prefix ${MODEL_PREFIX}/llama3_70b_layer1_seq1_bs16_kv8192