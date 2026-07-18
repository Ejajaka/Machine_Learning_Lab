import numpy as nump
import pandas as exe

file = "Lab_Session_Data.xlsx"
def load_thyroid(file):
    thy = exe.read_excel(file, sheet_name='thyroid0387_UCI')
    return thy

def attribute_types(thy):
    for x in thy.columns:
        print(x, "-> dtype:", thy[x].dtype, ", unique values:", thy[x].nunique())

def numeric_ranges(thy, numeric_cols):
    for x in numeric_cols:
        col = exe.to_numeric(thy[x], errors='coerce')
        print(x, "-> min:", col.min(), "max:", col.max())

def missing_values(thy):
    missing = thy.replace('?', nump.nan).isna().sum()
    print(missing[missing > 0])
    return missing

def outliers(thy, numeric_cols):
    for x in numeric_cols:
        col = exe.to_numeric(thy[x], errors='coerce').dropna()
        q1, q3 = col.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((col < lo) | (col > hi)).sum()
        print(x, "-> outliers:", n_out)

def mean_variance(thy, numeric_cols):
    for x in numeric_cols:
        col = exe.to_numeric(thy[x], errors='coerce')
        print(x, "-> mean:", col.mean(), "variance:", col.var())

numeric_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
thy = load_thyroid(file)
attribute_types(thy)
numeric_ranges(thy, numeric_cols)
missing_values(thy)
outliers(thy, numeric_cols)
mean_variance(thy, numeric_cols)