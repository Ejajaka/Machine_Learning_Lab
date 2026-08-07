import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

d = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="marketing_campaign")
def label(s):
    # Generated with Claude
    categories = sorted(s.unique())
    mapping = {category: idx for idx, category in enumerate(categories)}
    return s.map(mapping), mapping

def onehot(d, col):
    # Generated with Claude
    n = d.copy()
    for val in d[col].unique():
        n[str(val)] = (d[col] == val).astype(int)
    return n

e, m = label(d["Education"])
print("original:")
print(d["Education"])
print("\nlabel encoding:")
print(e)
print("\nmapping:")
print(m)
o = onehot(d, "Education")
print("\none hot encoding:")
print(o.head())