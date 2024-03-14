from transformers import AutoTokenizer, AutoModelForCausalLM, GPTJForCausalLM
import torch
import time

print("start")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir="/scratch/network/jmonas/.cache/")
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B",cache_dir="/scratch/network/jmonas/.cache/", revision="float16", low_cpu_mem_usage=True)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("done")
# print("device selection")
# # Move the model to the selected device
# model.to(device)

# print("Model Loaded..!")
 
# start_time = time.time()
 
# input_text = "Google was founded by"
 
# inputs = tokenizer(input_text, return_tensors="pt")
# input_ids = inputs["input_ids"].to("cuda")
      
# output = model.generate(
#    input_ids,
#    attention_mask=inputs["attention_mask"].to("cuda"),
#    do_sample=True,
#    max_length=150,
#    temperature=0.8,
#    use_cache=True,
#    top_p=0.9
# )
 
# end_time = time.time() - start_time
# print("Total Time => ",end_time)
# print(tokenizer.decode(output[0]))