# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

python -u ./gen.py --config XL2 --n 16 --input_size 256
python -u ./gen.py --config XL2 --n 16 --input_size 512
python -u ./gen.py --config XL2 --n 64 --input_size 256
python -u ./gen.py --config XL2 --n 64 --input_size 512