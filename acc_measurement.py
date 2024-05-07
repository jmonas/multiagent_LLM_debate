import json

# Path to the JSON file
file_path = 'combined_2_agents_3_rounds_results.json'

# Function to read JSON data from a file
def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Load the JSON data from the file
data = read_json(file_path)

# Get the 'final_agent_ans_flags' dictionary
flags = data['final_agent_ans_flags']

# Calculate the total sum of the values
total_sum = sum(flags.values())

# Get the number of elements in the 'final_agent_ans_flags'
num_elements = len(flags)

# Calculate the accuracy
accuracy = total_sum / num_elements

print("Accuracy:", accuracy)
