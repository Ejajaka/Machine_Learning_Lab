import pandas as pd
import numpy as np

d = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="marketing_campaign")

def mean(x):
    return np.sum(x, axis=0) / len(x)

def distance(a, b):
    s = 0
    for i in range(len(a)):
        s = s + (a[i] - b[i]) ** 2
    return np.sqrt(s)

cols = ["Income", "Recency", "MntWines", "MntFruits",
        "MntMeatProducts", "MntFishProducts",
        "MntSweetProducts", "MntGoldProds"]
x = d[cols].dropna().values

def kmeans(x, k):
    n = len(x)
    r = np.random.choice(n, k, replace=False)
    centroid = x[r]
    label = np.zeros(n)
    for t in range(100):
        for i in range(n):
            best = 0
            mind = distance(x[i], centroid[0])
            for j in range(1, k):
                d = distance(x[i], centroid[j])
                if d < mind:
                    mind = d
                    best = j
            label[i] = best
        new = centroid.copy()
        for j in range(k):
            p = x[label == j]
            if len(p) > 0:
                new[j] = mean(p)
        if np.array_equal(new, centroid):
            break
        centroid = new
    return centroid, label

centroids, labels = kmeans(x, 3)
print("Cluster Sizes:")
print(np.bincount(labels.astype(int)))
print("\nCentroids:")
print(centroids)