import numpy as nump
import pandas as exe
file = "Lab_Session_Data.xlsx"

def similarity(thy):
    row1 = thy.iloc[0].drop(['Record ID', 'Condition'])
    row2 = thy.iloc[1].drop(['Record ID', 'Condition'])
    A = exe.to_numeric(row1.replace({'f': 0, 't': 1, '?': nump.nan}), errors='coerce').fillna(0).values
    B = exe.to_numeric(row2.replace({'f': 0, 't': 1, '?': nump.nan}), errors='coerce').fillna(0).values
    cos_sim = nump.dot(A, B) / (nump.linalg.norm(A) * nump.linalg.norm(B))
    print("Cosine similarity:", cos_sim)
    return cos_sim

thy = exe.read_excel(file, sheet_name='thyroid0387_UCI')
cos_sim = similarity(thy)