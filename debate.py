from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import re
import time
import numpy as np
import json
import os
from datetime import datetime
# Initialize the models and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gg-hf/gemma-7b-it", cache_dir="/scratch/gpfs/jmonas/.cache/")
model = AutoModelForCausalLM.from_pretrained("gg-hf/gemma-7b-it", device_map="auto", torch_dtype=torch.float16, cache_dir="/scratch/gpfs/jmonas/.cache/")



def append_to_json(file_path, new_data):
    # Check if file exists and read data if it does, otherwise start with an empty list
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
    else:
        data = []

    # Append new entry to data list
    data.append(new_data)

    # Write the updated list back to the file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    print("DUMPED", flush=True)

# Define a function to format the chat history with the chat template
def format_chat(chat_history):
    prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
    return inputs

def generate_round_query(answer):
    return f"These are the solutions to the problem from other agents(s): {answer}. Using your previous answer and other agent's answers as additional advice, give an updated answer to the question. If you disagree with other agents, explain why. Put '!!!' before your final numerical answer."

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




def run_debate(number_of_rounds, number_of_agents, temperature):
    numbers = random.sample(range(1, 10), 4)
    operators = random.choices(['+','-', '*'], k=3)
    expression = f"{numbers[0]}{operators[0]}{numbers[1]}{operators[1]}{numbers[2]}{operators[2]}{numbers[3]}"
    expression_w_spaces = f"{numbers[0]} {operators[0]} {numbers[1]} {operators[1]} {numbers[2]} {operators[2]} {numbers[3]} ="
    print("\n")
    print("\n")
    print(f"STARTING {number_of_agents}-AGENT DEBATE", flush=True)
    print("EQUATION: ", expression, flush=True)
    print("CORRECT ANSWER: ", eval(expression), flush=True)
    print("---------------------------", flush=True)
    print("\n")
    print("\n")
    print("\n")

    
    chat_histories = [[{"role": "user", "content": f"What is the answer to: {expression}? Show your steps and make sure to state your answer at the end of the response. Put '!!!' before your final numerical answer. EXAMPLE: !!! 294."}] for _ in range(number_of_agents)]
    final_answers = []
    for round_num in range(number_of_rounds):
        torch.cuda.empty_cache()
        print(f"ROUND {round_num + 1} RESULTS", flush=True)
        print("---------------------------", flush=True)
        print("---------------------------", flush=True)
        print("---------------------------", flush=True)

        all_responses = []
        all_answers = []
        for i in range(number_of_agents):
            inputs = format_chat(chat_histories[i])
            outputs = model.generate(input_ids = inputs, max_new_tokens=125, do_sample = True, temperature = temperature)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_cleaned = clean_text(response)
            chat_histories[i].append({"role": "model", "content": response_cleaned})

            all_responses.append(response_cleaned)
            print("\n", flush=True)
            print("\n", flush=True)
            print(f"Agent {i+1} Results:", flush=True)
            print(response_cleaned, flush=True)
            final_answer = parse_final_answer_correctly(response_cleaned, expression, expression_w_spaces)
            all_answers.append(final_answer)
        
        final_answers.append(all_answers)   
        print("EXTRACTED ANSWERS", flush=True)
        for i, ans in enumerate(all_answers):
            print(f"ANSWER {i+1}: ", ans)
            aggregated_responses = ' '.join([f"AGENT {idx}: " + resp for idx, resp in enumerate(all_responses) if idx != i])
            chat_histories[i].append({"role": "user", "content": generate_round_query(aggregated_responses)})
        
        print("\n", flush=True)
        print("\n", flush=True)
        print("\n", flush=True)
        print("\n", flush=True)
        # print("\n", flush=True)
        # print("\n", flush=True)
    print("CORRECT ANSWER: ", eval(expression), flush=True)
    print(final_answers, flush=True)

    return expression, eval(expression), final_answers








num_debates = 200
number_of_agents = 3
num_rounds = 3
agents_correct = [0] * number_of_agents
temperature = .1
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
json_file_path = f'results/debate_results_{number_of_agents}_{num_rounds}_{current_time}_{temperature}.json'

for debate_round in range(num_debates):
    print("START", flush=True)
    start_time = time.time()
    expression, truth, answers = run_debate(num_rounds, number_of_agents, temperature)

    stringified_truth = str(truth)

    wrong_to_right = False
    # right_to_wrong = False
    agents_flag = {}

    for i, ans in enumerate(answers[-1]):
        if ans.isdigit() or (ans.startswith('-') and ans[1:].isdigit()):
            if ans == stringified_truth:
                agents_correct[i] +=1
                agents_flag[i] =1
            else: 
                agents_flag[i] =0
        else:
            agents_flag[i] =0
    if  any(x != stringified_truth for x in answers[0]) and all(x == answers[-1][0] for x in answers[-1]) and answers[-1][0] == stringified_truth:
        print("SUCCESS, WRONG CHANGED RIGHT")

        wrong_to_right = True
    # if any(x == truth for x in answers[0]) and not all(x == answers[-1][0] for x in answers[-1]):
    #     print("FAILURE, RIGHT CHANGED WRONG ")
    storage_json = {
        "debate_num": debate_round,
        "problem" :expression,
        "truth" :  stringified_truth,
        "round_answers" : answers,
        "wrong_to_right": wrong_to_right,
        # "right_to_wrong": right_to_wrong,
        "final_agent_ans_flags": agents_flag,
        "all_agents_right" : all(x == answers[-1][0] for x in answers[-1]) and answers[-1][0] == stringified_truth
    }

    append_to_json(json_file_path, storage_json)
    stop_time = time.time()
    print("elapsed time: ", stop_time - start_time, flush=True)
    print("\n", flush=True)
    print("\n", flush=True)
    print("\n", flush=True)
    print("\n", flush=True)
    print("\n", flush=True)
    print("\n", flush=True)
    print("\n", flush=True)
    print("\n", flush=True)


print("ACCURACY: ", np.array(agents_correct)/num_debates, flush=True)


