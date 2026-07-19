import numpy as nump
import pandas as exe
import matplotlib.pyplot as plot
import seaborn 

col = ['on thyroxine', 'query on thyroxine', 'on antithyroid medication', 'sick',
    'pregnant', 'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hyperthyroid',
    'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'T3 measured',
    'TT4 measured', 'T4U measured', 'FTI measured', 'TBG measured']

def jc_smc(v1, v2):
    f11 = nump.sum((v1 == 1) & (v2 == 1))
    f00 = nump.sum((v1 == 0) & (v2 == 0))
    f10 = nump.sum((v1 == 1) & (v2 == 0))
    f01 = nump.sum((v1 == 0) & (v2 == 1))
    jc = f11 / (f11 + f10 + f01)
    smc = (f11 + f00) / (f11 + f10 + f01 + f00)
    return jc, smc

def cosine(v1, v2):
    return nump.dot(v1, v2) / (nump.linalg.norm(v1) * nump.linalg.norm(v2))
thy = exe.read_excel("Lab_Session_Data.xlsx", sheet_name='thyroid0387_UCI')  
n = 20
binary_vectors = thy[col].replace({'f': 0, 't': 1}).iloc[:n].values
full_vectors = thy.drop(columns=['Record ID', 'Condition']).replace({'f': 0, 't': 1, '?': nump.nan})
full_vectors = full_vectors.apply(exe.to_numeric, errors='coerce').fillna(0).iloc[:n].values
JC = nump.zeros((n, n))
SMC = nump.zeros((n, n))
COS = nump.zeros((n, n))
for i in range(n):
    for j in range(n):
        JC[i, j], SMC[i, j] = jc_smc(binary_vectors[i], binary_vectors[j])
        COS[i, j] = cosine(full_vectors[i], full_vectors[j])
fig, axes = plot.subplots(1, 3, figsize=(18, 5))
seaborn.heatmap(JC, ax=axes[0])
axes[0].set_title("JC")
seaborn.heatmap(SMC, ax=axes[1])
axes[1].set_title("SMC")
seaborn.heatmap(COS, ax=axes[2])
axes[2].set_title("Cosine")
plot.show()