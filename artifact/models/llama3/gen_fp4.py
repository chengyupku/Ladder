import torch
from transformers import LlamaModel, LlamaConfig
import os
import argparse
from typing import List, Optional, Tuple, Union
from wrapper import LlamaWrapper
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers import BitsAndBytesConfig


nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="fp4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.float16
)

config_llama3_8b = LlamaConfig(
  attention_bias=False,
  attention_dropout=0.0,
  bos_token_id=128000,
  eos_token_id=128001,
  hidden_act="silu",
  hidden_size=4096,
  initializer_range=0.02,
  intermediate_size=14336,
  max_position_embeddings=8192,
  mlp_bias=False,
  num_attention_heads=32,
  num_hidden_layers=1,
  num_key_value_heads=8,
  pad_token_id=128255,
  pretraining_tp=1,
  rms_norm_eps=1e-05,
  rope_scaling=None,
  rope_theta=500000.0,
  tie_word_embeddings=False,
  use_cache=True,
  vocab_size=128256,
  quantization_config=nf4_config
)

config_llama3_70b = LlamaConfig(
  attention_bias=False,
  attention_dropout=0.0,
  bos_token_id=128000,
  eos_token_id=128001,
  hidden_act="silu",
  hidden_size=8192,
  initializer_range=0.02,
  intermediate_size=28672,
  max_position_embeddings=8192,
  mlp_bias=False,
  num_attention_heads=64,
  num_hidden_layers=1,
  num_key_value_heads=8,
  pad_token_id=128255,
  pretraining_tp=1,
  rms_norm_eps=1e-05,
  rope_scaling=None,
  rope_theta=500000.0,
  tie_word_embeddings=False,
  use_cache=True,
  vocab_size=128256,
  quantization_config=nf4_config
)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='8b')
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--seq_length_q", type=int, default=1)
parser.add_argument("--seq_length_kv", type=int, default=8192)
parser.add_argument("--is_decode", action="store_true")
parser.add_argument("--profile", action="store_true")
args = parser.parse_args()
batch_size = args.batch_size
seq_length_q = args.seq_length_q
seq_length_kv = args.seq_length_kv
is_decode = args.is_decode
if (args.config == '8b'):
    config = config_llama3_8b
elif (args.config == '70b'):
    config = config_llama3_70b
else:
    raise ValueError("Invalid config")

if (args.config == '8b'):
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
else:
    model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
model = LlamaModel.from_pretrained("./llama3_70b", quantization_config=nf4_config).layers[0]
# model = LlamaModel(config).half().cuda().layers[0]
# We use LlamaWrapper to wrap the model so that it can accept the same inputs as the LlamaModel
model = LlamaWrapper(config, model)
model.eval()
# print(model)

kv_shape = [
    batch_size, 
    config.num_key_value_heads, 
    seq_length_kv - seq_length_q, 
    config.hidden_size // config.num_attention_heads
]

rotary_emb = LlamaRotaryEmbedding(config=config).half().cuda()
rotary_emb.eval()

inputs_embeds = torch.ones(batch_size, seq_length_q, config.hidden_size, device="cuda", dtype=torch.float16)
position_ids = torch.arange(seq_length_kv - seq_length_q, seq_length_kv, device="cuda").unsqueeze(0)
position_embeddings = rotary_emb(inputs_embeds, position_ids)

k = torch.ones(kv_shape, device="cuda", dtype=torch.float16)
v = torch.ones(kv_shape, device="cuda", dtype=torch.float16)
past_key_values = [[k, v]]

if is_decode:
    input_args = (inputs_embeds, position_ids, position_embeddings, None, past_key_values, True)
else:
    input_args = (inputs_embeds, position_ids, position_embeddings)

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

    # model = torch.compile(model)
    with torch.no_grad():
        avg_time = measure_time(model, input_args)
        print(f"Average execution time: {avg_time:.6f} ms")
        model = torch.compile(model)
        avg_time = measure_time(model, input_args)
        print(f"Average execution time (torch.compile): {avg_time:.6f} ms")
    exit()
output = model(*input_args)

# make a directory to save the model
dir_name = f"llama3_{args.config}_layer1_seq{seq_length_q}_bs{batch_size}_kv{seq_length_kv}_no_attn"
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

# # For simple tests
# from single_attention import SingleAttention
# model = SingleAttention().half().cuda()
# model.eval()

# q_shape = [batch_size, config.num_attention_heads, seq_length_q, config.hidden_size // config.num_attention_heads]
# kv_shape = [batch_size, config.num_attention_heads, seq_length_kv, config.hidden_size // config.num_attention_heads]
# q = torch.ones(q_shape, device="cuda", dtype=torch.float16)
# k = torch.ones(kv_shape, device="cuda", dtype=torch.float16)
# v = torch.ones(kv_shape, device="cuda", dtype=torch.float16)

# input_args = (q, k, v, None)
# output = model(*input_args)

# # make a directory to save the model
# dir_name = f"llama3_{args.config}_layer1_seq{seq_length_q}_bs{batch_size}_kv{seq_length_kv}_single_attn"
# if not os.path.exists(dir_name):
#     os.makedirs(dir_name)

# # Save model into ONNX
# torch.onnx.export(
#     model,
#     input_args,
#     f"{dir_name}/model.onnx",
#     export_params=True,
#     opset_version=14
# )

# from ..model.pytorch.single_attention import SimpleGemm
# model = SimpleGemm().half().cuda()
# model.eval()
# m, n, k = 64, 64, 64
# A = torch.ones(m, k, device="cuda", dtype=torch.float16)
# B = torch.ones(k, n, device="cuda", dtype=torch.float16)

# dir_name = f"simple_gemm_test"
# if not os.path.exists(dir_name):
#     os.makedirs(dir_name)

# # Save model into ONNX
# torch.onnx.export(
#     model,
#     (A, B),
#     f"{dir_name}/model.onnx",
#     export_params=True,
#     opset_version=14
# )

# from ..model.pytorch.single_attention import SimpleAdd
# model = SimpleAdd().half().cuda()
# model.eval()
# m, n = 64, 64
# A = torch.ones(m, n, device="cuda", dtype=torch.float16)

# dir_name = f"simple_add_test"
# if not os.path.exists(dir_name):
#     os.makedirs(dir_name)

# # Save model into ONNX
# torch.onnx.export(
#     model,
#     (A),
#     f"{dir_name}/model.onnx",
#     export_params=True,
#     opset_version=14
# )
