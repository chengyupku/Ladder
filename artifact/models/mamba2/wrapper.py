import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache

class Mamba2Wrapper(nn.Module):
    def __init__(self, config, block, is_decode, batch_size):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([block])
        self.is_decode = is_decode

    def forward(
        self,
        inputs_embeds: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        hidden_states = inputs_embeds
        batch_size = hidden_states.shape[0]
        if self.is_decode:
            cache = Mamba2Cache(self.config, batch_size, torch.float16, "cuda")
            cache.seqlen_offset = 1
        else:
            cache = None
        for mixer_block in self.layers:
            hidden_states = mixer_block(
                hidden_states,
                cache_params=cache,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )

        return hidden_states