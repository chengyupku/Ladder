import torch
import os
import argparse
from transformers import Mamba2Config, Mamba2Model

export_single_layer = True
mamba2_config = {}
mamba2_config["1.3b"] = Mamba2Config(
    num_heads=64,
    head_dim=64,
    hidden_size=2048, # check: d_model
    state_size=128, # check: d_state
    num_hidden_layers=1,
    expand=2,
    conv_kernel=4, # check: d_conv
    n_groups=1,
    use_bias=False,
    use_conv_bias=True,
    use_cache=True,
    rms_norm=True,
    chunk_size=256,
)

mamba2_config["2.7b"] = Mamba2Config(
    num_heads=80,
    head_dim=64,
    hidden_size=2560, # check: d_model
    state_size=128, # check: d_state
    num_hidden_layers=1,
    expand=2,
    conv_kernel=4, # check: d_conv
    n_groups=1,
    use_bias=False,
    use_conv_bias=True,
    use_cache=True,
    rms_norm=True,
    chunk_size=256,
)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='1.3b')
parser.add_argument("--seq_length", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()

batch_size = args.batch_size
seq_length = args.seq_length
assert args.config in ["1.3b", "2.7b"], "Invalid config"
config = mamba2_config[args.config]

model = Mamba2Model(config).half().cuda()
if export_single_layer:
    model = model.layers[0]
model.eval()
print(model)

if export_single_layer:
    input_ids = torch.ones(batch_size, seq_length, config.hidden_size, device="cuda", dtype=torch.float16)
else:
    input_ids = torch.ones(args.batch_size, args.seq_length, device="cuda", dtype=torch.int64)
out = model(input_ids)
# print(out)

# make a directory to save the model
dir_name = f"mamba2_{args.config}_layer1_seq{seq_length}_bs{batch_size}"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# Save model into ONNX
torch.onnx.export(
    model,
    input_ids,
    f"{dir_name}/model.onnx",
    export_params=True,
    opset_version=14
)