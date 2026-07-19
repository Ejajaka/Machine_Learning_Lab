import numpy as nump
import pandas as exe

col = ['on thyroxine', 'query on thyroxine', 'on antithyroid medication', 'sick',
    'pregnant', 'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hyperthyroid',
    'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'T3 measured',
    'TT4 measured', 'T4U measured', 'FTI measured', 'TBG measured']

def jc_smc(thy):
    data = thy[col].replace({'f': 0, 't': 1}).iloc[:2]
    v1 = data.iloc[0].values
    v2 = data.iloc[1].values
    f11 = nump.sum((v1 == 1) & (v2 == 1))
    f00 = nump.sum((v1 == 0) & (v2 == 0))
    f10 = nump.sum((v1 == 1) & (v2 == 0))
    f01 = nump.sum((v1 == 0) & (v2 == 1))
    jc = f11 / (f11 + f10 + f01)
    smc = (f11 + f00) / (f11 + f10 + f01 + f00)
    print("JC:", jc, "SMC:", smc)
    return jc, smc

thy = exe.read_excel("Lab_Session_Data.xlsx", sheet_name='thyroid0387_UCI')
jc, smc = jc_smc(thy)