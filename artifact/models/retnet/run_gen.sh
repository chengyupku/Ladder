# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

python -u ./gen.py --config 3b --batch_size 1 --seq_length 4096
python -u ./gen.py --config 3b --batch_size 1 --seq_length 1 --is_decode
python -u ./gen.py --config 3b --batch_size 16 --seq_length 1 --is_decode

python -u ./gen.py --config 65b --batch_size 1 --seq_length 4096
python -u ./gen.py --config 65b --batch_size 1 --seq_length 1 --is_decode
python -u ./gen.py --config 65b --batch_size 16 --seq_length 1 --is_decode