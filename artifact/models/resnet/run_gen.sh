# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

python -u ./gen.py --batch_size 1 --profile
python -u ./gen.py --batch_size 128  --profile