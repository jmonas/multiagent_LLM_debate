from transformers import AutoTokenizer, AutoModelForCausalLM, GPTJForCausalLM
import torch
import time

print("start")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", cache_dir="/scratch/network/jmonas/.cache/")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto", torch_dtype=torch.float16,cache_dir="/scratch/network/jmonas/.cache/")

input_text = "Calculate and provide the exact numerical result for the following expression: 86+81+9."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=150)
print(tokenizer.decode(outputs[0]))
