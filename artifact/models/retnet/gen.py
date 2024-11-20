import torch
import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
retnet_src_path = os.path.join(current_dir, "RetNet", "src")
if retnet_src_path not in sys.path:
    sys.path.append(retnet_src_path)

from RetNet.src import retnet

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='1.3b')
parser.add_argument("--seq_length", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--profile", action="store_true")
args = parser.parse_args()

# batch_size = args.batch_size
# seq_length = args.seq_length
# assert args.config in ["1.3b", "2.7b"], "Invalid config"
# config = mamba2_config[args.config]

# model = Mamba2Model(config).half().cuda()
# if export_single_layer:
#     model = model.layers[0]
# model.eval()
# print(model)

# if export_single_layer:
#     input_ids = torch.ones(batch_size, seq_length, config.hidden_size, device="cuda", dtype=torch.float16)
# else:
#     input_ids = torch.ones(args.batch_size, args.seq_length, device="cuda", dtype=torch.int64)
# out = model(input_ids)
# # print(out)
# if args.profile:
#     def measure_time(model, input_args, num_warmup=10, num_runs=10):
#         for _ in range(num_warmup):
#             output = model(input_args)
#         start_event = torch.cuda.Event(enable_timing=True)
#         end_event = torch.cuda.Event(enable_timing=True)

#         timings = []
#         for _ in range(num_runs):
#             start_event.record()
#             output = model(input_args)
#             end_event.record()
#             torch.cuda.synchronize()
#             elapsed_time = start_event.elapsed_time(end_event)
#             timings.append(elapsed_time)
#         avg_time = sum(timings) / num_runs
#         return avg_time

#     avg_time = measure_time(model, input_ids)
#     print(f"Average execution time (no torch.compile): {avg_time:.6f} ms")
#     model = torch.compile(model)
#     avg_time = measure_time(model, input_ids)
#     print(f"Average execution time (torch.compile): {avg_time:.6f} ms")
#     exit()

# 1.3B model
layers = 1
hidden_dim = 2048
ffn_size = 4096
heads = 16

model = retnet.RetNet(layers, hidden_dim, ffn_size, heads, double_v_dim=True).half().cuda()
model.eval()
print(model)

# # make a directory to save the model
# dir_name = f"mamba2_{args.config}_layer1_seq{seq_length}_bs{batch_size}"
# if not os.path.exists(dir_name):
#     os.makedirs(dir_name)

# # Save model into ONNX
# torch.onnx.export(
#     model,
#     input_ids,
#     f"{dir_name}/model.onnx",
#     export_params=True,
#     opset_version=14
# )