import pandas as pd
import numpy as np

def bin_feature(x, bins=3, method="width"):
    if method == "width":
        edges = np.linspace(x.min(), x.max(), bins + 1)
        return pd.cut(x, edges)
    elif method == "freq":
        return pd.qcut(x, bins)
def entropy(y):
    p = y.value_counts() / len(y)
    return -sum(p * np.log2(p))
def information_gain(y, x):
    total = entropy(y)
    for v in x.unique():
        part = y[x == v]
        total -= len(part) / len(y) * entropy(part)
    return total
def decision_tree(data, features, target):
    y = data[target]
    best_feature = ""
    best_gain = -1
    for col in features:
        x = data[col]
        if x.nunique() > 5:
            x = bin_feature(x, 3, "width")
        gain = information_gain(y, x)
        print(col, ":", round(gain, 3))
        if gain > best_gain:
            best_gain = gain
            best_feature = col
    print("\nRoot node:", best_feature)
    print("Information Gain:", round(best_gain, 3))

data = pd.read_csv("ACTG175.csv")
features = ["age", "wtkg", "hemo", "homo", "drugs", "karnof",
            "oprior", "z30", "zprior", "preanti", "race", "gender",
            "str2", "strat", "symptom", "cd40", "cd80"]
decision_tree(data, features, "treat")