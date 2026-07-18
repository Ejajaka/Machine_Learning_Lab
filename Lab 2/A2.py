import numpy as nump
import pandas as exe

def classify(file):
    purchase = exe.read_excel(file, sheet_name='Purchase data').iloc[:, :5]
    purchase["R/P"] = nump.where(purchase["Payment (Rs)"] > 200, "RICH", "POOR")
    return purchase

file = "Lab_Session_Data.xlsx"
purchase = classify(file)
print(purchase)