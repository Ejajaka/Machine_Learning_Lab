import pandas as pd
import numpy as np

def entropy(y):
    p = y.value_counts() / len(y)
    return -sum(p * np.log2(p))
def information_gain(y, x):
    total = entropy(y)
    for v in x.unique(): # only choosing the unique values for the split and iterating through them
        part = y[x == v] # only considering the values that fall in after the split
        total -= len(part) / len(y) * entropy(part)
    return total

data = pd.read_csv("ACTG175.csv")
features = ["age", "wtkg", "hemo", "homo", "drugs", "karnof",
            "oprior", "z30", "zprior", "preanti", "race", "gender",
            "str2", "strat", "symptom", "cd40", "cd80"]
y = data["treat"]
best_feature = ""
best_gain = -1 #info gain is always 0 or greater then one
for col in features:
    x = data[col]
# Converting numerical data into 3 categories
    if x.nunique() > 5:   
        x = pd.cut(x, 3) 
    gain = information_gain(y, x)
    print(col, ":", round(gain, 3))
    if gain > best_gain:
        best_gain = gain
        best_feature = col
print("\nRoot node:", best_feature)