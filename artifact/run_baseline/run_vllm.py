import argparse
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from vllm import LLM, SamplingParams
from vllm.attention.backends.flash_attn import FlashAttentionMetadata, FlashAttentionMetadataBuilder
from vllm.worker.model_runner import ModelInputForGPUBuilder
from vllm.forward_context import set_forward_context

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--seq_length", type=int, default=128)
parser.add_argument("--model", type=str, default="llama3_8b")
parser.add_argument("--quant", type=str, default="")
args = parser.parse_args()

model = args.model
if args.quant:
    model = f"{model}_{args.quant}"
batch_size = args.batch_size
seq_length = args.seq_length
prompts = [" ".join(["hello"] * (seq_length - 1))] * batch_size
sampling_params = SamplingParams(min_tokens=10)

llm = LLM(model=f"./{model}", enforce_eager=True, gpu_memory_utilization=0.5, quantization="bitsandbytes", load_format="bitsandbytes")
outputs = llm.generate(prompts, sampling_params)

runner = llm.llm_engine.model_executor.driver_worker.model_runner
print("runner: ", runner)
print("runner type: ", type(runner))

seq_lens = [4096]
query_lens = [4096]
input_builder = ModelInputForGPUBuilder(runner)
attn_metadata = FlashAttentionMetadataBuilder(input_builder).build(
            seq_lens, query_lens, -1, batch_size)
attn_metadata.num_prefills = 1
attn_metadata.num_prefill_tokens = 4096
model = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[0].half().cuda()
model.eval()

batch_size = 1
seq_length_q = 4096
seq_length_kv = 4096
hidden_size = 4096
inputs_embeds = torch.ones(batch_size * seq_length_q, hidden_size, device="cuda", dtype=torch.float16)
position_ids = torch.arange(seq_length_kv - seq_length_q, seq_length_kv, device="cuda").unsqueeze(0)
kv_cache = torch.tensor([], device="cuda", dtype=torch.float16)

print("my metadata: ", attn_metadata)
input_args = (position_ids, inputs_embeds, kv_cache, attn_metadata, None)
with set_forward_context(attn_metadata):
    out = model(*input_args)
# print(model)
# print(type(model))


import time
repeats = 1000
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