import torch
from transformers import LlamaModel, LlamaConfig
import os
import argparse

export_single_layer = False
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
  vocab_size=128256
)

model = LlamaModel(config_llama3_70b).half().cuda()
print(model.dtype)
if export_single_layer:
    model = model.layers[0]
model.eval()
print(model)
parser = argparse.ArgumentParser()
parser.add_argument("--seq_length_q", type=int, default=1)
parser.add_argument("--seq_length_kv", type=int, default=64)
parser.add_argument("--batch_size", type=int, default=1)
args = parser.parse_args()

batch_size = args.batch_size
seq_length_q = args.seq_length_q
seq_length_kv = args.seq_length_kv

if export_single_layer:
    input_ids = torch.ones(batch_size, seq_length_q, config_llama3_70b.hidden_size, device="cuda", dtype=torch.float16)
else:
    input_ids = torch.ones(batch_size, seq_length_q, device="cuda", dtype=torch.int64)
    kv_shape = [batch_size, config_llama3_70b.num_key_value_heads, seq_length_kv, config_llama3_70b.hidden_size // config_llama3_70b.num_attention_heads]
    k = torch.ones(kv_shape, device="cuda", dtype=torch.float16)
    v = torch.ones(kv_shape, device="cuda", dtype=torch.float16)
    past_key_values = [[k, v]]
    args = (input_ids, None, None, past_key_values, None, True)
    # output = model(input_ids, past_key_values=past_key_values, use_cache=True)

    # make a directory to save the model -> {llama2_70b_layer1_seq1_bs16/model.onnx}
    dir_name = f"llama3_70b_layer1_seq{seq_length_q}_bs{batch_size}_kv{seq_length_kv}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Save model into ONNX
    torch.onnx.export(
        model,
        args,
        f"{dir_name}/model.onnx",
        export_params=True,
        opset_version=14
    )