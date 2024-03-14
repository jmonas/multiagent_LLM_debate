from transformers import AutoTokenizer, AutoModelForCausalLM, GPTJForCausalLM
import torch
import time

def generate_round_query(answer):
    return f"These are the recent/updated opinions from other agents: {answer} Use these opinions carefully as additional advice, can you provide an updated answer?"

print("start")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", cache_dir="/scratch/network/jmonas/.cache/")
model_A = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto", torch_dtype=torch.float16,cache_dir="/scratch/network/jmonas/.cache/")
model_B = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto", torch_dtype=torch.float16,cache_dir="/scratch/network/jmonas/.cache/")

input_text = "Calculate exact numerical result for the following expression: 86+81+9*14-3"


input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

output_A_1 = model_A.generate(**input_ids, max_new_tokens=150, do_sample = True, temperature = .2)
output_B_1 = model_B.generate(**input_ids, max_new_tokens=150, do_sample = True, temperature = .2)


decoded_output_A_1 = tokenizer.decode(output_A_1[0])
decoded_output_B_1 = tokenizer.decode(output_B_1[0])

print("ROUND 1 RESULTS")
print("---------------------------")
print("A Results")
print(decoded_output_A_1)
print("B Results")
print(decoded_output_B_1)
print()
print()


input_id_A_2 = tokenizer(generate_round_query(decoded_output_B_1), return_tensors="pt").to("cuda")
input_id_B_2 = tokenizer(generate_round_query(decoded_output_A_1), return_tensors="pt").to("cuda")

print("token_length", len(input_id_A_2["input_ids"]))

output_A_2 = model_A.generate(**input_id_A_2, max_new_tokens=350, do_sample = True, temperature = .2)
output_B_2 = model_B.generate(**input_id_B_2, max_new_tokens=350, do_sample = True, temperature = .2)


decoded_output_A_2 = tokenizer.decode(output_A_2[0])
decoded_output_B_2 = tokenizer.decode(output_B_2[0])

print("ROUND 2 RESULTS")
print("---------------------------")
print("A Results")
print(decoded_output_A_2)
print("B Results")
print(decoded_output_B_2)
print()
print()