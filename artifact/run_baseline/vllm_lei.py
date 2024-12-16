from vllm import LLM, SamplingParams
import numpy as np
from vllm.model_executor.models.llama import LlamaForCausalLM
import argparse
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="/home/aiscuser/cy/Ladder/artifact/run_baseline/llama3_8b")
parser.add_argument("--enforce_eager", type=bool, default=True)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--seq_len", type=int, default=1)

args = parser.parse_args() 

model = args.model

llm = LLM(model=model, enforce_eager=True, gpu_memory_utilization=0.5, quantization="bitsandbytes", load_format="bitsandbytes")
# llm = LLM(model=model, enforce_eager=True, gpu_memory_utilization=0.5)
sampling_params = SamplingParams(temperature=1.0,
                                 top_p=1.0,
                                 ignore_eos=True,
                                 max_tokens=8192,
                                 min_tokens=5120
                                 )

llm.llm_engine.model_executor.driver_worker.model_runner.model.kv_cache_length = 0

batch_size = args.batch_size
prompt_len = args.seq_len
dummy_prompt_token_ids = np.random.randint(10000, size=(batch_size, prompt_len)).tolist()

outputs = llm.generate(prompt_token_ids=dummy_prompt_token_ids,
                        sampling_params=sampling_params
                        )
print(outputs)