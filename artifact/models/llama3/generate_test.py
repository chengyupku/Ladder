import torch
from transformers import LlamaModel, LlamaConfig, LlamaForCausalLM
import os
import argparse
from typing import List, Optional, Tuple, Union
from artifact.models.llama3.wrapper import LlamaWrapper
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

export_single_layer = False
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
  vocab_size=128256
)
input_ids = torch.tensor([[0,0,0,0,0,0,0,0]], device="cuda", dtype=torch.int64)
model = LlamaForCausalLM(config_llama3_8b).half().cuda()
# Generate
generate_ids = model.generate(input_ids, max_length=30)
print(generate_ids)

# import torch
# import math

# query_states = torch.rand((2, 64), device="cuda", dtype=torch.float16)
# key_states = torch.rand((16, 64), device="cuda", dtype=torch.float16)
# value_states = torch.rand((16, 64), device="cuda", dtype=torch.float16)

# def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
#         is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
#     L, S = query.size(-2), key.size(-2)
#     scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
#     attn_bias = torch.zeros(L, S, dtype=query.dtype, device="cuda")
#     if is_causal:
#         assert attn_mask is None
#         temp_mask = torch.ones(L, S, dtype=torch.bool, device="cuda").tril(diagonal=0)
#         attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
#         attn_bias.to(query.dtype)
#         print("attn_bias", attn_bias)

#     if attn_mask is not None:
#         if attn_mask.dtype == torch.bool:
#             attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
#         else:
#             attn_bias += attn_mask

#     if enable_gqa:
#         key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
#         value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

#     attn_weight = query @ key.transpose(-2, -1) * scale_factor
#     attn_weight += attn_bias
#     attn_weight = torch.softmax(attn_weight, dim=-1)
#     attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
#     return attn_weight @ value

# ref_out = scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=None, dropout_p=0.0, is_causal=True)
# attn_output = torch.nn.functional.scaled_dot_product_attention(
#     query_states,
#     key_states,
#     value_states,
#     attn_mask=None,
#     dropout_p=0.0,
#     is_causal=True,
# )
# print(attn_output)
# assert torch.allclose(ref_out, attn_output, atol=1e-3)