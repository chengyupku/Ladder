import torch
from transformers import LlamaModel, LlamaConfig
import os
import argparse

class SingleAttention(torch.nn.Module):
    def __init__(self):
        super(SingleAttention, self).__init__()

    def forward(self, query_states, key_states, value_states, causal_mask):
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=0.0,
            is_causal=False,
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