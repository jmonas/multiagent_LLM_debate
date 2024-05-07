from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", cache_dir="/scratch/gpfs/jmonas/.cache/")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", torch_dtype=torch.float16,cache_dir="/scratch/gpfs/jmonas/.cache/")

# input_text = "Write me a poem about Machine Learning."
# input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# outputs = model.generate(**input_ids)
# print(tokenizer.decode(outputs[0]))
