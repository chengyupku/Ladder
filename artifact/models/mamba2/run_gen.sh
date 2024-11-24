# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

python -u ./gen.py --config 1.3b --batch_size 1 --seq_length 1 --is_decode --profile
python -u ./gen.py --config 1.3b --batch_size 32 --seq_length 1 --is_decode --profile
python -u ./gen.py --config 1.3b --batch_size 128 --seq_length 1 --is_decode --profile
