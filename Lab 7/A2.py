import pandas as pd
import numpy as np

def gini(y):
    p = y.value_counts() / len(y)
    return 1 - sum(p ** 2)
data = pd.read_csv("ACTG175.csv")
y = data["treat"]
print("Gini Index:", gini(y))