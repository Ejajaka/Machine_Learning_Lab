import numpy as nump
import pandas as exe

col = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
thy = exe.read_excel("Lab_Session_Data.xlsx", sheet_name='thyroid0387_UCI')
thy = thy.replace('?', nump.nan)
for x in col:
    thy[x] = exe.to_numeric(thy[x], errors='coerce')
    thy[x] = thy[x].fillna(thy[x].median())
print("before:")
print(thy[col].agg(['min', 'max']))
for x in col:
    thy[x] = (thy[x] - thy[x].min()) / (thy[x].max() - thy[x].min())
print("after:")
print(thy[col].agg(['min', 'max']))