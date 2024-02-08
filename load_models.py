from transformers import AutoTokenizer, AutoModelForCausalLM

AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=".cache")
AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=".cache")
