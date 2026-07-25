import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

d = pd.read_csv("simulation_500.csv")
def label(s):
    c = sorted(s.dropna().unique())
    m = {}
    i = 0
    for x in c:
        m[x] = i
        i += 1
    return s.map(m), m

def onehot(d, col):
    n = d.copy()
    vals = d[col].dropna().unique()
    for x in vals:
        l = []
        for v in d[col]:
            if v == x:
                l.append(1)
            else:
                l.append(0)
        n[str(x)] = l
    n = n.drop(columns=[col])
    return n

e, m = label(d["method"])
print("original:")
print(d["method"])
print("\nlabel encoding:")
print(e)
print("\nmapping:")
print(m)
o = onehot(d, "method")
print("\none hot encoding:")
print(o.head())
