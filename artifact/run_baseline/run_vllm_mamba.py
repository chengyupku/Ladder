import argparse
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from vllm import LLM, SamplingParams
from vllm.attention.backends.placeholder_attn import PlaceholderAttentionMetadataBuilder, PlaceholderAttentionMetadata
from vllm.worker.model_runner import ModelInputForGPUBuilder
from vllm.forward_context import set_forward_context

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--seq_length", type=int, default=8192)
parser.add_argument("--model", type=str, default="mamba2_1.3b")
parser.add_argument("--quant", type=str, default="")
args = parser.parse_args()

model = args.model
batch_size = args.batch_size
seq_length = args.seq_length
prompts = [" ".join(["hello"] * (seq_length - 1))] * batch_size
sampling_params = SamplingParams(min_tokens=10)

llm = LLM(model=f"./{model}", enforce_eager=True)
outputs = llm.generate(prompts, sampling_params)


runner = llm.llm_engine.model_executor.driver_worker.model_runner
seq_lens = [seq_length]
query_lens = [seq_length]
input_builder = ModelInputForGPUBuilder(runner)
attn_metadata = PlaceholderAttentionMetadataBuilder(input_builder).build(
            seq_lens, query_lens, -1, batch_size)
attn_metadata.num_prefills = 1
attn_metadata.num_prefill_tokens = seq_length
attn_metadata.context_lens_tensor = torch.tensor([0], device="cuda", dtype=torch.int32)
model = llm.llm_engine.model_executor.driver_worker.model_runner.model.backbone.layers[0].cuda()

print("my metadata: ", attn_metadata)

hidden_size = 2048
hidden_states = torch.ones(batch_size * seq_length, hidden_size, device="cuda", dtype=torch.float16)
conv_state = torch.ones(batch_size, 4096, 3, device="cuda", dtype=torch.float16)
ssm_state = torch.ones(batch_size, 4096, 128, device="cuda", dtype=torch.float16)

input_args = (hidden_states, attn_metadata, None, conv_state, ssm_state)
with set_forward_context(attn_metadata):
    out = model(*input_args)

import time
repeats = 10
def profile_function():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.synchronize()
    else:
        device = torch.device("cpu")
    
    for _ in range(10):
        with set_forward_context(attn_metadata):
            out = model(*input_args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    start = time.time()
    for _ in range(repeats):
        with set_forward_context(attn_metadata):
            out = model(*input_args)
    if torch.cuda.is_available():
        torch.cuda.synchronize() 
    end = time.time()

    avg_time = (end - start) / repeats * 1000
    print(f"Average execution time over {repeats} runs: {avg_time:.4f} ms")

profile_function()