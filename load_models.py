from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained("gg-hf/gemma-7b-it")
model = AutoModelForCausalLM.from_pretrained("gg-hf/gemma-7b-it", quantization_config=quantization_config)

# input_text = "Write me a poem about Machine Learning."
# input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# outputs = model.generate(**input_ids)
# print(tokenizer.decode(outputs[0]))
