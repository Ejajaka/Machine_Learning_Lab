import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

d = pd.read_csv("simulation_500.csv")
def mean(x):
    return np.sum(x, axis=0) / len(x)

def variance(x):
    m = mean(x)
    return np.sum((x - m) ** 2, axis=0) / len(x)

feature = d["accuracy"].values.astype(float)
counts, bin = np.histogram(feature, bins=10)
print("Bin edges:", bin)
print("Counts per bin:", counts)
plt.hist(feature, bins=10, edgecolor="black")
plt.xlabel("accuracy")
plt.ylabel("Frequency")
plt.title("Histogram of accuracy")
plt.show()
feat_mean = mean(feature)
feat_var = variance(feature)
print("Mean:", feat_mean)
print("Variance:", feat_var)
