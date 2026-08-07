import pandas as pd
import numpy as np

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

w = d.drop(columns=["ID", "Dt_Customer"]).copy()
m = {}
m["Basic"] = 0
m["2n Cycle"] = 1
m["Graduation"] = 2
m["Master"] = 3
m["PhD"] = 4
e = w.copy()
e["Education"] = e["Education"].map(m)
e = onehot(e, "Marital_Status")
print("Before encoding:", len(w.columns), "columns")
print("After encoding:", len(e.columns), "columns")
print(e.head())