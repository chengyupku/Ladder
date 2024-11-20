# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

echo "Running phimoe_3.5b, batch 1, seqlen_q 4096, seqlen_kv 4096"
python -u ./gen.py --config 3.5b --batch_size 1 --seq_length_q 4096 --seq_length_kv 4096 --profile

echo "Running phimoe_3.5b, batch 1, seqlen_q 1, seqlen_kv 8192"
python -u ./gen.py --config 3.5b --batch_size 1 --seq_length_q 1 --seq_length_kv 8192 --is_decode --profile

echo "Running phimoe_3.5b, batch 16, seqlen_q 1, seqlen_kv 8192"
python -u ./gen.py --config 3.5b --batch_size 16 --seq_length_q 1 --seq_length_kv 8192 --is_decode --profile

echo "Running phimoe_3.5b, batch 64, seqlen_q 1, seqlen_kv 8192"
python -u ./gen.py --config 3.5b --batch_size 64 --seq_length_q 1 --seq_length_kv 8192 --is_decode --profile