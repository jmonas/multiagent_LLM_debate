from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import re
import time

# Initialize the models and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gg-hf/gemma-7b-it", cache_dir="/scratch/gpfs/jmonas/.cache/")
model = AutoModelForCausalLM.from_pretrained("gg-hf/gemma-7b-it", device_map="auto", torch_dtype=torch.float16, cache_dir="/scratch/gpfs/jmonas/.cache/")


# Define a function to format the chat history with the chat template
def format_chat(chat_history):
    prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
    return inputs

def generate_round_query(answer):
    return f"These are the solutions to the problem from other agents: {answer}. Using other agent's answers as additional advice, give an updated answer to the questio. If you find disagree with other agents, explain why. Put '!!!' before your final numerical answer."

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


def parse_final_answer_correctly(text, expression, expression_w_spaces):
    text = text.replace(expression,"")
    text = text.replace(expression_w_spaces,"")
    text = text.replace(",","")
    pattern = r"""
        !!!\s*'?\s*                 # Matches the '!!!', optional spaces, and optional single quote
        (-?\d+)                     # Captures the numeric value, including negative numbers
    """

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




def run_debate(number_of_rounds, number_of_agents):
    numbers = random.sample(range(1, 10), 4)
    operators = random.choices(['+','-'], k=3)
    expression = f"{numbers[0]}{operators[0]}{numbers[1]}{operators[1]}{numbers[2]}{operators[2]}{numbers[3]}"
    expression_w_spaces = f"{numbers[0]} {operators[0]} {numbers[1]} {operators[1]} {numbers[2]} {operators[2]} {numbers[3]} ="
    print("\n")
    print("\n")
    print(f"STARTING {number_of_agents}-AGENT DEBATE")
    print("EQUATION: ", expression)
    print("CORRECT ANSWER: ", eval(expression))
    print("---------------------------")
    print("\n")
    print("\n")
    print("\n")

    
    chat_histories = [[{"role": "user", "content": f"What is the answer to: {expression}? Show your steps and make sure to state your answer at the end of the response. Put '!!!' before your final numerical answer. EXAMPLE: Say the question is 1+1. After your reasoning, you would write:  !!! 2."}] for _ in range(number_of_agents)]
    final_answers = []
    for round_num in range(number_of_rounds):
        print(f"ROUND {round_num + 1} RESULTS")
        print("---------------------------")
        print("---------------------------")
        print("---------------------------")

        all_responses = []
        all_answers = []
        for i in range(number_of_agents):
            inputs = format_chat(chat_histories[i])
            outputs = model.generate(input_ids = inputs, max_new_tokens=125, do_sample = True, temperature = .25)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_cleaned = clean_text(response)
            chat_histories[i].append({"role": "model", "content": response_cleaned})

            all_responses.append(response_cleaned)
            print("\n")
            print("\n")
            print(f"Agent {i+1} Results:")
            print(response_cleaned)
            final_answer = parse_final_answer_correctly(response_cleaned, expression, expression_w_spaces)
            all_answers.append(final_answer)
        
        final_answers.append(all_answers)   
        print("EXTRACTED ANSWERS")
        for i, ans in enumerate(all_answers):
            print(f"ANSWER {i+1}: ", ans)
            aggregated_responses = ' '.join([f"AGENT {idx}: " + resp for idx, resp in enumerate(all_responses) if idx != i])
            chat_histories[i].append({"role": "user", "content": generate_round_query(aggregated_responses)})
        
        print("\n")
        print("\n")
        print("\n")
        print("\n")
        # print("\n")
        # print("\n")
    print(final_answers)
    return chat_histories

for _ in range(3):
    start_time = time.time()
    final_chat_history = run_debate(3, 2)
    stop_time = time.time()
    print("elapsed time: ", stop_time - start_time)
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")



