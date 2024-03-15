from transformers import AutoTokenizer, AutoModelForCausalLM, GPTJForCausalLM
import torch
import time

def generate_round_query(answer):
    return f"These are the recent/updated opinions from other agents: {answer} Use these opinions carefully as additional advice, can you provide an updated answer?"

print("start")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", cache_dir="/scratch/network/jmonas/.cache/")
model_A = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto", torch_dtype=torch.float16,cache_dir="/scratch/network/jmonas/.cache/")

input_text = "Calculate exact numerical result for the following expression: 86+81+9*14-3"


input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

output_A_1 = model_A.generate(**input_ids, max_new_tokens=150, do_sample = True, temperature = .2)


decoded_output_A_1 = tokenizer.decode(output_A_1[0])

input_ids = tokenizer("What was my last question?", return_tensors="pt").to("cuda")
output_A_2 = model_A.generate(**input_ids, max_new_tokens=150, do_sample = True, temperature = .2)
