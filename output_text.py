from transformers import AutoTokenizer, AutoModelForCausalLM, GPTJForCausalLM
import torch
import time

print("start")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", cache_dir="/scratch/network/jmonas/.cache/")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto", torch_dtype=torch.float16,cache_dir="/scratch/network/jmonas/.cache/")

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(input_ids=input_ids, max_new_tokens=150)
print(tokenizer.decode(outputs[0]))
