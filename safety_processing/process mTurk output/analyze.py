import statistics
import numpy as np
import pandas as pd

df = pd.read_csv("output.csv")
data = df["Answer"]


# Mean
mean = statistics.mean(data)

# Mode
try:
    mode = statistics.mode(data)
except statistics.StatisticsError:
    mode = "No unique mode found"

# Standard Deviation
stddev = statistics.stdev(data)

# 25th and 75th Percentiles
percentile_25 = np.percentile(data, 25)
percentile_75 = np.percentile(data, 75)

# Printing the results
print(f"Mean: {mean}")
print(f"Mode: {mode}")
print(f"Standard Deviation: {stddev}")
print(f"25th Percentile: {percentile_25}")
print(f"75th Percentile: {percentile_75}")
