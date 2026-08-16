import numpy as np
import pandas as pd
#to measure similarity between target and data points use eucliedian 
def eucl(x1, x2):  
    return np.sqrt(np.sum((x1 - x2) ** 2))
# Indentifying neighbors
def knn(data,pt,k):
    distance=[]
    for i in range(len(data)):
        dist = eucl(pt, data[i])
        distance.append((dist, i))
# Sorting
    distance.sort(key=lambda x: x[0])
    neighbour = [distance[i][1] for i in range(k)]
    return neighbour
# prediction
data=pd.read_csv("simulation_500.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
pt = X[0]
k = 3
neighbour = knn(X, pt, k)
labels = [y[i] for i in neighbour]
prediction = max(set(labels), key=labels.count)
print("Neighbors:", neighbour)
print("Prediction:", prediction)
