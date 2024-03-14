import random

def generate_random_math_problem():
    # Define the operations
    operations = ['+', '-', '*']
    # Randomly choose the number of numbers involved (between 5 and 8)
    num_count = random.randint(4, 8)
    # Initialize the problem string with the first number
    problem = str(random.randint(1, 100))
    for _ in range(num_count - 1):
        # Choose a random operation
        op = random.choice(operations)
        # Choose a random number
        num = random.randint(1, 100)
        # Append the operation and number to the problem string
        problem += f"{op}{num}"
    return problem, eval(problem)

query, answer = generate_random_math_problem()

print(query, answer)

problem_statement = f"What is the result of {query}? Make sure to state your answer at the end of the response."

print(problem_statement)