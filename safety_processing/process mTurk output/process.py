import csv


# Define a function to extract answers based on options being "true"
def get_answers(row, question_num):
    options = [
        f"Answer.text{question_num}_option_1A.checked",
        f"Answer.text{question_num}_option_1B.checked",
        f"Answer.text{question_num}_option_1C.checked",
        f"Answer.text{question_num}_option_1D.checked",
        f"Answer.text{question_num}_option_1E.checked",
    ]

    for idx, option in enumerate(options):
        if row[option].strip().lower() == "true":
            return str(idx)


# Initialize an empty dictionary to store questions and their corresponding answers
question_answers = {}

# Read the CSV file
with open("data.csv", "r", newline="") as csvfile:
    reader = csv.DictReader(csvfile)

    # Iterate through each row in the CSV
    for row in reader:
        # Iterate over questions 0-4
        for i in range(5):
            question_key = row[f"Input.question{i}"]

            # Get the answers for the given question
            answer = get_answers(row, i)

            # Update the dictionary
            if question_key not in question_answers:
                question_answers[question_key] = []
            if answer:
                question_answers[question_key].append(answer)

prune_ct = 0

# Write the results to an output CSV file
with open("output.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    # Write the header
    header = ["Question", "Answer"]
    writer.writerow(header)

    # Write each question, average of answers, and its corresponding answers to the CSV
    for question, answers in question_answers.items():
        average = sum(map(int, answers)) / len(answers) if answers else 0
        if average == 0:
            prune_ct += 1
        writer.writerow([question, average, "; ".join(answers)])
    print("Total pruned is", prune_ct)
