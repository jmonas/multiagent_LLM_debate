from transformers import AutoTokenizer, AutoModelForCausalLM, GPTJForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir="/scratch/network/jmonas/.cache/")
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16, cache_dir="/scratch/network/jmonas/.cache/")
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B",cache_dir="/scratch/network/jmonas/.cache/", revision="float16", low_cpu_mem_usage=True)