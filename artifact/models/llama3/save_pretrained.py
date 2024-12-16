import argparse
from transformers import LlamaConfig, LlamaForCausalLM, MambaConfig, MambaForCausalLM
from transformers import LlamaTokenizerFast

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama3_8b")
args = parser.parse_args()
name = args.model
assert name in ["llama3_8b", "llama3_70b", "mamba2_1.3b"]

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
    vocab_size=128256,
)

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

mamba2_config = MambaConfig(
    num_heads=64,
    head_dim=64,
    hidden_size=2048, # check: d_model
    state_size=128, # check: d_state
    num_hidden_layers=1,
    expand=2,
    conv_kernel=4, # check: d_conv
    n_groups=1,
    use_bias=False,
    use_conv_bias=True,
    use_cache=True,
    rms_norm=True,
    chunk_size=256,
)

if name == "llama3_8b":
    model = LlamaForCausalLM(config=config_llama3_8b)
    save_path = "./llama3_8b"
elif name == "llama3_70b":
    model = LlamaForCausalLM(config=config_llama3_70b)
    save_path = "./llama3_70b"
elif name == "mamba2_1.3b":
    model = MambaForCausalLM(config=mamba2_config)
    save_path = "./mamba2_1.3b"

model.save_pretrained(save_path)
tokenizer = LlamaTokenizerFast.from_pretrained("huggyllama/llama-7b", legacy=False, from_slow=True)
tokenizer.save_pretrained(save_path)