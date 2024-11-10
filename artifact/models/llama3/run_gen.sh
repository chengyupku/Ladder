# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# llama3_8b, batch 1, seqlen_q 4096, seqlen_kv 4096
echo "Running llama3_8b, batch 1, seqlen_q 4096, seqlen_kv 4096"
python -u ./gen.py --config 8b --batch_size 1 --seq_length_q 4096 --seq_length_kv 4096

# llama3_8b, batch 1, seqlen_q 1, seqlen_kv 8192
echo "Running llama3_8b, batch 1, seqlen_q 1, seqlen_kv 8192"
python -u ./gen.py --config 8b --batch_size 1 --seq_length_q 1 --seq_length_kv 8192 --is_decode

# llama3_8b, batch 16, seqlen_q 1, seqlen_kv 8192
echo "Running llama3_8b, batch 16, seqlen_q 1, seqlen_kv 8192"
python -u ./gen.py --config 8b --batch_size 16 --seq_length_q 1 --seq_length_kv 8192 --is_decode

# llama3_70b, batch 1, seqlen_q 4096, seqlen_kv 4096
echo "Running llama3_70b, batch 1, seqlen_q 4096, seqlen_kv 4096"
python -u ./gen.py --config 70b --batch_size 1 --seq_length_q 4096 --seq_length_kv 4096

# llama3_70b, batch 1, seqlen_q 1, seqlen_kv 8192
echo "Running llama3_70b, batch 1, seqlen_q 1, seqlen_kv 8192"
python -u ./gen.py --config 70b --batch_size 1 --seq_length_q 1 --seq_length_kv 8192 --is_decode

# llama3_70b, batch 16, seqlen_q 1, seqlen_kv 8192
echo "Running llama3_70b, batch 16, seqlen_q 1, seqlen_kv 8192"
python -u ./gen.py --config 70b --batch_size 16 --seq_length_q 1 --seq_length_kv 8192 --is_decode