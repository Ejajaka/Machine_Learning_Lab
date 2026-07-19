import pandas as exe
import numpy as nump

def matrix(data):
    X = data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    Y = data["Payment (Rs)"].values
    return X, Y

def cost(X, Y):
    X_pinv = nump.linalg.pinv(X)
    cost = X_pinv @ Y
    return cost

data = exe.read_excel("Lab_Session_Data.xlsx", sheet_name="Purchase data")
X, Y = matrix(data)
print("matrix:")
print(X)
print("\noutput:")
print(Y)
rank = nump.linalg.matrix_rank(X)
print("rank of matrix:", rank)
cost = cost(X, Y)
print("\ncost of products")
print("candy: ", cost[0])
print("mango: ", cost[1])
print("milk packet: ", cost[2])