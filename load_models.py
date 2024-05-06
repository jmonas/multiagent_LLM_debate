from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("gg-hf/gemma-7b-it", cache_dir="/scratch/gpfs/jmonas/.cache/gemma-7b-it-8bit/")
model = AutoModelForCausalLM.from_pretrained("gg-hf/gemma-7b-it", device_map="auto", tcache_dir="/scratch/gpfs/jmonas/.cache/gemma-7b-it-8bit/")

# input_text = "Write me a poem about Machine Learning."
# input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# outputs = model.generate(**input_ids)
# print(tokenizer.decode(outputs[0]))
