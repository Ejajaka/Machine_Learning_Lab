import numpy as nump
import pandas as exe

col = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
thy = exe.read_excel("Lab_Session_Data.xlsx", sheet_name='thyroid0387_UCI')
thy = thy.replace('?', nump.nan)
for x in col:
    thy[x] = exe.to_numeric(thy[x], errors='coerce')
    thy[x] = thy[x].fillna(thy[x].median())
thy['sex'] = thy['sex'].fillna(thy['sex'].mode()[0])
print(thy[col + ['sex']].isna().sum())