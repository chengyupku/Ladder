import torch
import os
import sys
import argparse
from unet import UNet

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--profile", action="store_true")
args = parser.parse_args()

model =UNet(3, 64).half().cuda()
model.eval()

print(model)
input_args = (torch.randn(args.batch_size, 3, 4096, 4096, device="cuda", dtype=torch.float16), )

if args.profile:
    def measure_time(model, input_args, num_warmup=10, num_runs=10):
        for _ in range(num_warmup):
            output = model(*input_args)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        timings = []
        for _ in range(num_runs):
            start_event.record()
            output = model(*input_args)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            timings.append(elapsed_time)
        avg_time = sum(timings) / num_runs
        return avg_time

    avg_time = measure_time(model, input_args)
    print(f"Average execution time (no torch.compile): {avg_time:.6f} ms")
    model = torch.compile(model)
    avg_time = measure_time(model, input_args)
    print(f"Average execution time (torch.compile): {avg_time:.6f} ms")
    exit()

# make a directory to save the model
dir_name = f"unet_bs{args.batch_size}"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# Save model into ONNX
torch.onnx.export(
    model,
    input_args,
    f"{dir_name}/model.onnx",
    export_params=True,
    opset_version=14
)