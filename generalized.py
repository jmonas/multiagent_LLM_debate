from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize the models and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gg-hf/gemma-2b-it", cache_dir="/scratch/network/jmonas/.cache/")
model_A = AutoModelForCausalLM.from_pretrained("gg-hf/gemma-2b-it", device_map="auto", torch_dtype=torch.float16, cache_dir="/scratch/network/jmonas/.cache/")
model_B = AutoModelForCausalLM.from_pretrained("gg-hf/gemma-2b-it", device_map="auto", torch_dtype=torch.float16, cache_dir="/scratch/network/jmonas/.cache/")

# Define a function to format the chat history with the chat template
def format_chat(chat_history):
    prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
    return inputs

def generate_round_query(answer):
    return f"These are the recent/updated opinions from other agents: {answer} Use these opinions carefully as additional advice, can you provide an updated answer?"

def clean_text(response):
    last_model_pos = response.rfind("<start_of_turn>model")
    extracted_text = response[last_model_pos:].split("\n", 1)[1] if last_model_pos != -1 else ""
    unwanted_phrase = "Sure, here's the calculation:\n"
    if extracted_text.startswith(unwanted_phrase):
        # Remove the unwanted phrase and any additional leading newlines
        extracted_text = extracted_text[len(unwanted_phrase):].lstrip("\n")
    return extracted_text

# Initial chat setup
chat_history_A = [
    {"role": "user", "content": "Calculate exact numerical result for the following expression: 86+81+9*14-3"},
]

chat_history_B = [
    {"role": "user", "content": "Calculate exact numerical result for the following expression: 86+81+9*14-3"},
]

# Function to run the debate for a specified number of rounds
def run_debate(number_of_rounds, chat_history_A, chat_history_B):
    for round_num in range(number_of_rounds):
        print(f"ROUND {round_num + 1} RESULTS")
        print("---------------------------")
        print("Query with history:")
        print(chat_history_A)
        print("---------------------------")
        print("---------------------------")
        
        # Generate a response from model_A
        inputs = format_chat(chat_history_A)
        outputs_A = model_A.generate(input_ids = inputs, max_new_tokens=150, do_sample = True, temperature = .2)
        response_A = tokenizer.decode(outputs_A[0], skip_special_tokens=True)
        print(response_A)
        response_A_cleaned = clean_text(response_A)

        print("Agent A Results:")
        print(response_A_cleaned)
        chat_history_A.append({"role": "model", "content": response_A_cleaned})
        
        # Generate a response from model_B
        inputs = format_chat(chat_history_B)
        outputs_B = model_B.generate(input_ids = inputs, max_new_tokens=150, do_sample = True, temperature = .2)
        response_B = tokenizer.decode(outputs_B[0], skip_special_tokens=True)
        response_B_cleaned = clean_text(response_B)

        print("Agent B Results:")
        print(response_B_cleaned)
        chat_history_B.append({"role": "model", "content": response_B_cleaned})
        
        print("\n")


        chat_history_A.append({"role": "user", "content": generate_round_query(response_B_cleaned)})
        chat_history_B.append({"role": "user", "content": generate_round_query(response_A_cleaned)})
        
    
    return chat_history_A, chat_history_B

# Run the debate for a specified number of rounds
final_chat_history_A, final_chat_history_B  = run_debate(2, chat_history_A, chat_history_B)
