import numpy as nump
import pandas as exe

def classify(purchase):
    purchase["R/P"] = nump.where(purchase["Payment (Rs)"] > 200, "RICH", "POOR")
    return purchase

purchase = exe.read_excel("Lab_Session_Data.xlsx", sheet_name='Purchase data').iloc[:, :5]
purchase = classify(purchase)
print(purchase)