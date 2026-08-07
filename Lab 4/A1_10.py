import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

d = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="marketing_campaign")
def mean(x):
    # Generated with Claude
    x = np.array(x)
    return np.sum(x) / len(x)
def variance(x):
    # Generated with Claude
    x = np.array(x)
    m = mean(x)
    return np.sum((x - m) ** 2) / len(x)

feature = d["MntWines"].values.astype(float)
counts, bin = np.histogram(feature, bins=10)
print("Bin edges:", bin)
print("Counts per bin:", counts)
plt.hist(feature, bins=10, edgecolor="black")
plt.xlabel("MntWines")
plt.ylabel("Frequency")
plt.title("Histogram of MntWines")
plt.show()
f_mean = mean(feature)
f_var = variance(feature)
print("Mean:", f_mean)
print("Variance:", f_var)