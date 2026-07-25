import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

d = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="marketing_campaign")
def mean(x):
    return np.sum(x, axis=0) / len(x)

def variance(x):
    m = mean(x)
    return np.sum((x - m) ** 2, axis=0) / len(x)

feature = d["MntWines"].values.astype(float)
counts, bin = np.histogram(feature, bins=10)
print("Bin edges:", bin)
print("Counts per bin:", counts)
plt.hist(feature, bins=10, edgecolor="black")
plt.xlabel("MntWines")
plt.ylabel("Frequency")
plt.title("Histogram of MntWines")
plt.show()
feat_mean = mean(feature)
feat_var = variance(feature)
print("Mean:", feat_mean)
print("Variance:", feat_var)