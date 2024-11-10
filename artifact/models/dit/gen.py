import torch
import torch.nn as nn
import os
import argparse
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from DiT.models import DiT_models, TimestepEmbedder, LabelEmbedder

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='XL2')
parser.add_argument("--n", type=int, default=16)
parser.add_argument("--input_size", type=int, default=256)
parser.add_argument("--num_classes", type=int, default=1000)
args = parser.parse_args()
n = args.n
input_size = args.input_size // 8
num_classes = args.num_classes

assert "DiT-"+args.config in DiT_models, "Invalid config"

model = DiT_models["DiT-"+args.config](
    input_size=input_size,
    num_classes=num_classes
)

patch_size = model.patch_size
in_channels = model.in_channels
if "XL" in args.config:
    hidden_size = 1152
elif "L" in args.config:
    hidden_size = 1024
elif "B" in args.config:
    hidden_size = 768
elif "S" in args.config:
    hidden_size = 384

model = model.blocks[0].half().cuda()
model.eval()
print(model)

x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True).half().cuda()
t_embedder = TimestepEmbedder(hidden_size).half().cuda()

# Create sampling noise:
x = torch.randn(n, 4, input_size, input_size, dtype=torch.float16, device="cuda")
x = x_embedder(x)  # (N, T, D), where T = H * W / patch_size ** 2
t = torch.randn(n, hidden_size, dtype=torch.float16, device="cuda")

# out = model(x, t)                      # (N, T, D)

# make a directory to save the model
dir_name = f"dit_{args.config}_layer1_n{n}_input{input_size}"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# Save model into ONNX
torch.onnx.export(
    model,
    (x, t),
    f"{dir_name}/model.onnx",
    export_params=True,
    opset_version=14
)