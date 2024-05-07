import json

# Path to the JSON file
file_path = 'results/debate_results_5_3_2024-05-06_05-10-51_0.4.json'

# Function to read JSON data from a file
def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Load the JSON data from the file
data = read_json(file_path)

# Initialize a dictionary to keep track of the sums for "0" and "1"
right_to_wrong = 0
wrong_to_right = 0

# Sum values across all entries
for entry in data:
    wrong_to_right += 1 if entry.get('wrong_to_right', False) else 0

    round_answers = entry.get('round_answers', False)
    truth = entry.get('truth', False)

    local_right_to_wrong = 0
    for i, ans in enumerate(round_answers[0]):
        if ans == truth and round_answers[-1][i] != truth:
            local_right_to_wrong = 1

    right_to_wrong+=local_right_to_wrong



print("total items: ", len(data))
print(f"right to wrong: ", right_to_wrong)
print(f"wrong to right: ", wrong_to_right)
