import pandas as pd
import numpy as np

def entropy(y):
    p = y.value_counts() / len(y)  # no. of times each value appears/ length of y
    return -sum(p * np.log2(p))
data = pd.read_csv("ACTG175.csv")
y = data["treat"]
print("Entropy:", entropy(y))