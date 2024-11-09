# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# llama3_8b, batch 1, seqlen_q 1, seqlen_kv 8192
python -u ./gen.py --config 8b --batch_size 1 --seq_length_q 8192 --seq_length_kv 8192