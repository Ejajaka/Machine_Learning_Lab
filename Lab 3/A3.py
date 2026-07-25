import pandas as pd
import numpy as np

d = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="marketing_campaign")
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