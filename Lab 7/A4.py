import pandas as pd
import numpy as np

data = pd.read_csv("ACTG175.csv")
def bin_feature(x, bins, method):
    if method == "width":
        edges = np.linspace(x.min(), x.max(), bins + 1) # equally splitting features between each bin and b+1 becuase for n bins we need n=1 pts
        return pd.cut(x, edges)
    elif method == "freq":
        return pd.qcut(x, bins)

print("Default:")
print(bin_feature(data["cd40"],3,"width").value_counts().sort_index())
print("\nFrequency:")
print(bin_feature(data["cd40"], 4, "freq").value_counts().sort_index())