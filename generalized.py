from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import re
# Initialize the models and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gg-hf/gemma-7b-it", cache_dir="/scratch/gpfs/jmonas/.cache/")
model_A = AutoModelForCausalLM.from_pretrained("gg-hf/gemma-7b-it", device_map="auto", torch_dtype=torch.float16, cache_dir="/scratch/gpfs/jmonas/.cache/")
# model_B = AutoModelForCausalLM.from_pretrained("gg-hf/gemma-7b-it", device_map="auto", torch_dtype=torch.float16, cache_dir="/scratch/network/jmonas/.cache/")

# Randomly generate three numbers between 1-30
numbers = random.sample(range(1, 20), 5)

# Randomly choose two different mathematical operators from the set
operators = random.choices(['+', '-', '*'], k=4)

# Format the mathematical expressions as strings
expression = f"{numbers[0]}{operators[0]}{numbers[1]}{operators[1]}{numbers[2]}{operators[2]}{numbers[3]}{operators[3]}{numbers[4]}"
expression_w_spaces = f"{numbers[0]} {operators[0]} {numbers[1]} {operators[1]} {numbers[2]} {operators[2]} {numbers[3]} {operators[3]} {numbers[4]} ="


# Define a function to format the chat history with the chat template
def format_chat(chat_history):
    prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
    return inputs

def generate_round_query(answer):
    return f"Another agent has provided this answer: {answer}. Examine their reasoning and calculation method. Based on their conclusion and your analysis, can you offer a revised or alternative answer? If you find discrepancies or disagree, please explain the reasoning behind your perspective. Place a '$' before your final answer."

def clean_text(response):
    last_model_pos = response.rfind("model")
    extracted_text = response[last_model_pos:].split("\n", 1)[1] if last_model_pos != -1 else ""
    unwanted_phrases = [
        "Sure, here's the calculation:\n",
        "Please note that these opinions are not guaranteed to be accurate and should be used with caution.",
        "Sure, here's the revised answer based on the updated context:",
        "Use these opinions carefully as additional advice, but be aware that the opinions may change in the future."
    ]
    for phrase in unwanted_phrases:
        if phrase in extracted_text:
            extracted_text = extracted_text.replace(phrase, "")
    
    # Strip leading and trailing whitespace and newlines
    return extracted_text.strip()

# def clean_extraction_text(query, response):
#     extracted_text = response.replace(query, "")
#     pattern = r"-?\d+"
#     # Find all matches of the pattern
#     numbers = re.findall(pattern, extracted_text)
#     # Convert found strings to integers
#     numbers = [int(number) for number in numbers]
#     return numbers, extracted_text

def parse_final_answer_correctly(text):
    text = text.replace(expression,"")
    text = text.replace(expression_w_spaces,"")
    text = text.replace(",","")
    pattern = r"-?\d+"
    
    # Find all matches of the pattern in the text
    numbers = re.findall(pattern, text)
    
    # Convert found strings to integers
    numbers = [int(number) for number in numbers]
    
    return numbers

    # pattern = r"""
    #     \$\s*'?\s*                # Matches the dollar sign, optional spaces, and optional single quote
    #     (?:                       # Non-capturing group for the whole expression
    #         (?:                   # Non-capturing group for arithmetic operations
    #             -?\d+\s*[\+\-\*\/]\s*   # Matches numbers and arithmetic operators
    #         )*                    # Zero or more repetitions of the arithmetic operations
    #         (-?\d+)               # Captures the numeric value (including negative numbers)
    #     )\s*'?\s*(?:=|(?!\+|\-|\*|\/)\D|$)   # Looks for an equals sign or non-arithmetic characters following the number
    # """
    # Compile the pattern with VERBOSE flag to allow whitespace and comments
    regex = re.compile(pattern, re.VERBOSE | re.MULTILINE | re.DOTALL)

    # Search for matches using the compiled regex pattern
    matches = regex.findall(text)

    # Extract the numeric values from the matches
    answers = [match for match in matches if match]

    if answers:
        # Return the last matched answer, assuming it's the final one in the text
        return answers[-1]
    else:
        return "No answer found"



print("EQUATION: ", expression)
print("CORRECT ANSWER: ", eval(expression))
# Initial chat setup 
chat_history_A = [
    {"role": "user", "content": f"What is the result of: {expression}? Show your steps and make sure to state your answer at the end of the response. Place a '$' before your final answer."},
]

chat_history_B = [
    {"role": "user", "content": f"What is the result of: {expression}? Show your steps and Make sure to state your answer at the end of the response. Place a '$' before your final answer."},
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
        outputs_A = model_A.generate(input_ids = inputs, max_new_tokens=100, do_sample = True, temperature = .8)
        response_A = tokenizer.decode(outputs_A[0], skip_special_tokens=True)
        response_A_cleaned = clean_text(response_A)

        print("Agent A Results:")
        print(response_A_cleaned)
        chat_history_A.append({"role": "model", "content": response_A_cleaned})
        
        # Generate a response from model_B
        inputs = format_chat(chat_history_B)
        outputs_B = model_A.generate(input_ids = inputs, max_new_tokens=100, do_sample = True, temperature = .8)
        response_B = tokenizer.decode(outputs_B[0], skip_special_tokens=True)
        response_B_cleaned = clean_text(response_B)
        print("Agent B Results:")
        print(response_B_cleaned)
        chat_history_B.append({"role": "model", "content": response_B_cleaned})

        print("\n")
        print("\n")

        pattern = r"(\*\*Additional Notes:\*\*.+|\*\*Alternative Answer:\*\*.+|\*\*Reasoning:\*\*.+|\*\*Explanation:\*\*.+)"
        response_A_cleaned = re.sub(pattern, "", response_A_cleaned, flags=re.DOTALL)
        response_B_cleaned = re.sub(pattern, "", response_B_cleaned, flags=re.DOTALL)
        

        extract_query_A = f" {response_A_cleaned}. Extract the final answer from the text. Only output the numerical answer."
        extract_query_B = f" {response_B_cleaned}. Extract the final answer from the text. Only output the numerical answer."
        extract_query_A_ids = tokenizer(extract_query_A, return_tensors="pt").to("cuda")
        extract_query_B_ids = tokenizer(extract_query_B, return_tensors="pt").to("cuda")

        extract_text_A_outputs = model_A.generate(**extract_query_A_ids, max_new_tokens=15)
        extract_text_B_outputs = model_A.generate(**extract_query_B_ids, max_new_tokens=15)
        
        extract_text_A_outputs = tokenizer.decode(extract_text_A_outputs[0], skip_special_tokens=True).replace(extract_query_A, "")
        extract_text_B_outputs = tokenizer.decode(extract_text_B_outputs[0], skip_special_tokens=True).replace(extract_query_B, "")
        cleaned_answer_A = parse_final_answer_correctly(extract_text_A_outputs)
        cleaned_answer_B = parse_final_answer_correctly(extract_text_B_outputs)

        print("EXTRACTED ANSWERS")
        print("ANSWER A", cleaned_answer_A)
        print("ANSWER B", cleaned_answer_B)
        print("\n")
        print("\n")
        print("WORK A")
        print(extract_text_A_outputs)
        print("WORK B")
        print(extract_text_B_outputs)
        print("\n")
        print("\n")
        print("\n")
        print("\n")
        print("\n")
        print("\n")
        print("\n")
        chat_history_A.append({"role": "user", "content": generate_round_query(response_B_cleaned)})
        chat_history_B.append({"role": "user", "content": generate_round_query(response_A_cleaned)})
        
    
    return chat_history_A, chat_history_B

# Run the debate for a specified number of rounds
final_chat_history_A, final_chat_history_B  = run_debate(3, chat_history_A, chat_history_B)
