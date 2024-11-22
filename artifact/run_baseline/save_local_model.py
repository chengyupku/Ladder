from transformers import LlamaConfig, LlamaForCausalLM

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

model = LlamaForCausalLM(config=config_llama3_8b)

save_path = "./llama3_8b"
model.save_pretrained(save_path)

from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b")
tokenizer.save_pretrained("./llama3_8b")

# from transformers import LlamaForCausalLM
# model = LlamaForCausalLM.from_pretrained("./llama3_8b")
# print(model)