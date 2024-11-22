import torch
from transformers import LlamaModel, LlamaConfig, LlamaTokenizer
from conftest import VllmRunner
from vllm import SamplingParams
import time
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--seq_length", type=int, default=1)
parser.add_argument("--int4", action="store_true", help="use int4 quantization", default=False)

args = parser.parse_args()
batch_size = args.batch_size
seq_length = args.seq_length

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

linear_method = None

run_single = True
enable_awq = args.int4

batch_size = 1
in_seq_len = 32
out_seq_len = 128

with VllmRunner(
    "./llama3_8b",
    dtype="half",
    # quantization="bitblas",
    enforce_eager=True,
) as model:

    sampling_params = SamplingParams(max_tokens=out_seq_len)
    bitbnet_outputs = model.generate(
        ["Hello"], sampling_params=sampling_params
    )
    print(bitbnet_outputs)
    
    # def get_runtime():
    #     tic = time.time()
    #     _ = model(*args)
    #     return (time.time() - tic) * 1000

    # with torch.no_grad():
    #     st = time.time()
    #     while time.time() - st < 1.0:
    #         get_runtime() # warmup
    #     times = [get_runtime() for i in range(100)]
    #     print(f"vllm bloom run b{batch_size} s{seq_length} avg: {np.mean(times)} ms")


# if enable_awq:
#     quant_config = AWQConfig(
#         weight_bits=4,
#         group_size=32,
#         zero_point=False
#     )
#     linear_method = AWQLinearMethod(quant_config)
# model = LlamaForCausalLM(vllm_config=VllmConfig(model_config=config_llama3_8b))
# model = model.cuda().half()

# if run_single:
#     model = model.model.layers[0]

# if run_single:
#     hidden_states = torch.ones(batch_size, seq_length, config_70b.hidden_size, device="cuda", dtype=torch.float16)
#     position_ids = torch.ones(batch_size, seq_length, device="cuda", dtype=torch.long)
#     num_slots = 1 * 1
#     slot_mapping = random.sample(range(num_slots), 1)
#     slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device="cuda")
#     input_metadata = InputMetadata(
#             prompt_lens=[1],
#             slot_mapping=slot_mapping,
#             max_context_len=None,
#             context_lens=None,
#             block_tables=None,
#     )
#     kv_caches = (None, None)
#     cache_event = None
#     residual = None
#     args = (position_ids, hidden_states, kv_caches, input_metadata, cache_event, residual)
# else:
#     input_ids = torch.ones(batch_size, seq_length, device="cuda", dtype=torch.float16)
#     position_ids = torch.ones(batch_size, seq_length, device="cuda", dtype=torch.long)
#     input_metadata = InputMetadata(
#             prompt_lens=[1],
#             slot_mapping=1,
#             max_context_len=None,
#             context_lens=None,
#             block_tables=None,
#     )
#     kv_caches = [(None, None)] * config_70b.num_hidden_layers
#     cache_envents = None


# with torch.no_grad():
#     while True:
#         _ = model(*args)
