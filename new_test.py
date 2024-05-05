import re

def parse_final_answers_refined(text):
    # Pattern to match arithmetic expressions and final answers, 
    # including cases where the answer is reiterated or explained further.
    pattern = r"""
        \$\s*'?\s*                # Matches the dollar sign, optional spaces, and optional single quote
        (?:                       # Non-capturing group for the whole expression
            (?:                   # Non-capturing group for arithmetic operations
                -?\d+\s*[\+\-\*\/]\s*   # Matches numbers and arithmetic operators
            )*                    # Zero or more repetitions of the arithmetic operations
            (-?\d+)               # Captures the numeric value (including negative numbers)
        )\s*'?\s*(?:=|(?!\+|\-|\*|\/)\D|$)   # Looks for an equals sign or non-arithmetic characters following the number
    """
    matches_found = []

    for pattern in patterns:
        # Find all matches and capture both arithmetic expressions and final numeric answers
        matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
        for match in matches:
            # Choosing the relevant numeric part, preferring the last piece as the final answer
            final_piece = match[-1]  # Last captured group presumed to be the most relevant
            if final_piece:
                matches_found.append(final_piece)

    if matches_found:
        # Returning unique matches to avoid duplication if the answer is reiterated
        return list(set(matches_found))
    else:
        return ["No answer found"]
    
def parse_accurate_final_answer2(text):
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

    

# Example texts with complex cases and reiterated answers
example_texts = [
    "**Final Answer:**\n\n$77\n\nROUND 2 RESULTS",
    "**Revised Answer:**\n\n$48\n\n**Reasoning:**",
    "Therefore, the final answer is $48.\n\n**Alternative Solution:**",
    "**Final Answer:**\n\n$-1\n\nTherefore, the revised answer is -1.",
    "**Final Answer:**\n\n$' -159 '"
]

a = "Agent A Results:**Reasoning:**1. Calculate the multiplication of 6 and 9, which is 54.2. Add 15 to the result, which is 69.3. Subtract 18 from the result, which is 51.**Final Answer:** !!! 51"
pattern = r"(\*\*Additional Notes:\*\*.+|\*\*Alternative Answer:\*\*.+|\*\*Reasoning:\*\*.+|\*\*Explanation:\*\*.+)"
response_A_cleaned = re.sub(pattern, "", a, flags=re.DOTALL)

print(response_A_cleaned)
# print(parse_accurate_final_answer2(example_texts))
for example in [response_A_cleaned]:
    print(parse_accurate_final_answer2(example))


    # "**Final Answer:**\n\n$1 + 70 - 88 = -17",
    # "**Revised Answer:**\n\n$1 + 10 * 7 - 8 * 11 = -1",