from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=".cache")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=".cache")

# tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
# model_inputs = tokenizer(
#     ["A list of colors: red, blue", "Portugal is"], return_tensors="pt", padding=True
# ).to("cuda")
# generated_ids = model.generate(**model_inputs)
# generated_texts =tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

# with open('output.txt', 'w') as file:
#     for text in generated_texts:
#         file.write(text + '\n')