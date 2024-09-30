import json
import csv
import random


# Read the JSON data from the file
with open("data.json", "r") as file:
    data = json.load(file)

questions = []

# Iterate through each item and print the question
for item in data:
    questions.append(item["question"])

random.shuffle(questions)

# Group the dataset into batches of 5
batches = [questions[i : i + 5] for i in range(0, len(questions), 5)]

# Write the results into a CSV file
with open("output.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    # Write the header
    header = ["question1", "question2", "question3", "question4", "question5"]
    writer.writerow(header)

    # Write each batch to the CSV
    for batch in batches:
        row = [item for item in batch]

        # Fill in empty cells if the last batch has less than 5 questions
        while len(row) < 5:
            row.append("")

        writer.writerow(row)
