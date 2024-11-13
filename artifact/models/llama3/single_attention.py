import torch
from transformers import LlamaModel, LlamaConfig
import os
import argparse
import math

class SingleAttention(torch.nn.Module):
    def __init__(self):
        super(SingleAttention, self).__init__()

    def forward(self, query_states, key_states, value_states, causal_mask):
        # head_dim = query_states.size(-1)
        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        # # if attention_mask is not None:  # no matter the length, we just slice it
        # #     causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        # #     attn_weights = attn_weights + causal_mask
        # # upcast attention to fp32
        # attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_weights = torch.nn.functional.dropout(attn_weights, p=0.0, training=False)
        # attn_output = torch.matmul(attn_weights, value_states)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=0.0,
            is_causal=True,
        )
        return attn_output

class SimpleGemm(torch.nn.Module):
    def __init__(self):
        super(SimpleGemm, self).__init__()

    def forward(self, A, B):
        return torch.matmul(A, B)

class SimpleAdd(torch.nn.Module):
    def __init__(self):
        super(SimpleAdd, self).__init__()

    def forward(self, A):
        return A + 1