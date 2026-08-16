import numpy as np
import pandas as pd
# similarity between two points we use euclidean distance
def eucl(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
# find the k closest rows to our point
def knn(data, pt, k):
    distance = []
    for i in range(len(data)):
        dist = eucl(pt, data[i])
        distance.append((dist, i))
# sort so smallest distance is first
    distance.sort(key=lambda x: x[0])
    neighbour = [distance[i][1] for i in range(k)]
    return neighbour
# weight by distance
def weighted(data, y, pt, k):
    neighbour = knn(data, pt, k)
    votes = {}
    for i in neighbour:
        d = eucl(pt, data[i])
        w = 1 / (d + 0.000000001)     # small distance gives big weight and division by 0 shouldnt be possible so we add 0.0000001
        label = y[i]
        votes[label] = votes.get(label, 0) + w
    prediction = max(votes, key=votes.get)   # winner is heaviest class
    return prediction, neighbour, votes
data = pd.read_csv("simulation_500.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
pt = X[0]
k = 3
pred, nb, votes = weighted(X, y, pt, k)
print("Neighbors:", nb)
print("Distances:", [float(round(eucl(pt, X[i]), 4)) for i in nb])
print("Weighted votes:",{int(c): float(round(w, 4))for c, w in votes.items()})
print("Prediction:", pred)
print("True label:", y[0])