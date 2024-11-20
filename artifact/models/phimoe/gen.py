import torch
from transformers import PhimoeConfig, PhimoeModel, PhimoeForCausalLM
from transformers.models.phimoe.modeling_phimoe import PhimoeRotaryEmbedding
import os
import argparse
from wrapper import PhimoeWrapper
from typing import List, Optional, Tuple, Union

config_phimoe_3_5b = PhimoeConfig(
    vocab_size=32064,
    hidden_size=4096,
    intermediate_size=6400,
    num_hidden_layers=1,
    num_attention_heads=32,
    num_key_value_heads=8,
    hidden_act="silu",
    max_position_embeddings=131072,
    initializer_range=0.02,
    rms_norm_eps=1e-5,
    use_cache=True,
    bos_token_id=1,
    eos_token_id=32000,
    tie_word_embeddings=False,
    rope_theta=10000.0,
    sliding_window=131072,
    attention_dropout=0.0,
    num_experts_per_tok=2,
    num_local_experts=16,
    output_router_logits=False,
    router_aux_loss_coef=0.0,
    router_jitter_noise=0.01,
    input_jitter_noise=0.01,
    attention_bias=True,
    lm_head_bias=True,
)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='3.5b')
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
if (args.config == '3.5b'):
    config = config_phimoe_3_5b
else:
    raise ValueError("Invalid config")


# input_ids = torch.tensor([[0,0,0,0,0,0,0,0]], device="cuda", dtype=torch.int64)
# model = PhimoeForCausalLM(config).half().cuda()
# # Generate
# generate_ids = model.generate(input_ids, max_length=30)
# print(generate_ids)

model = PhimoeModel(config).half().cuda().layers[0]
model = PhimoeWrapper(config, model)
model.eval()
print(model)

kv_shape = [
    batch_size, 
    config.num_key_value_heads, 
    seq_length_kv - seq_length_q, 
    config.hidden_size // config.num_attention_heads
]

rotary_emb = PhimoeRotaryEmbedding(config=config).half().cuda()
rotary_emb.eval()

inputs_embeds = torch.ones(batch_size, seq_length_q, config.hidden_size, device="cuda", dtype=torch.float16)
position_ids = torch.arange(0, seq_length_q, device="cuda").unsqueeze(0)
position_embeddings = rotary_emb(inputs_embeds, seq_length_q)

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

    avg_time = measure_time(model, input_args)
    print(f"Average execution time (no torch.compile): {avg_time:.6f} ms")
    model = torch.compile(model)
    avg_time = measure_time(model, input_args)
    print(f"Average execution time (torch.compile): {avg_time:.6f} ms")
    exit()
output = model(*input_args)

# make a directory to save the model
dir_name = f"phimoe_{args.config}_layer1_seq{seq_length_q}_bs{batch_size}_kv{seq_length_kv}"
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