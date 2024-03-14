from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
print("start")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=".cache")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B",torch_dtype=torch.float16, cache_dir=".cache")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device selection")
# Move the model to the selected device
model.to(device)

prompt = (
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English."
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
print("tokenizer")

# Move the input IDs to the same device as the model
input_ids = input_ids.to(device)

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)
print("model generate")

gen_text = tokenizer.batch_decode(gen_tokens)[0]
print("decode")

# Move the generated tokens back to CPU for decoding if you're using CUDA
if device == torch.device("cuda"):
    gen_text = tokenizer.batch_decode(gen_tokens.cpu())[0]
else:
    gen_text = tokenizer.batch_decode(gen_tokens)[0]

print(gen_text)
