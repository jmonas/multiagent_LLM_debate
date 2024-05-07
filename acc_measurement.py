import json

# Path to the JSON file
file_path = 'combined_2_agents_3_rounds_results.json'

# Function to read JSON data from a file
def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Load the JSON data from the file
data = read_json(file_path)

# Initialize a dictionary to keep track of the sums for "0" and "1"
flag_sums = {"0": 0, "1": 0}
total_counts = {"0": 0, "1": 0}

# Sum values across all entries
for entry in data:
    flags = entry.get('final_agent_ans_flags', {})
    for key in ["0", "1"]:
        if key in flags:
            flag_sums[key] += flags[key]
            total_counts[key] += 1

# Calculate the accuracy for each flag
for key in ["0", "1"]:
    if total_counts[key] > 0:
        accuracy = flag_sums[key] / total_counts[key]
        print(f"Accuracy for flag {key}: {accuracy:.2f}")
    else:
        print(f"No data found for flag {key}")
