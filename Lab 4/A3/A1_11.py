import pandas as pd
import numpy as np


def mean(x):
    # Generated with Claude
    x = np.array(x)
    return np.sum(x, axis=0) / len(x)


def distance(a, b):
    # Generated with Claude
    a = np.array(a)
    b = np.array(b)
    return np.sqrt(np.sum((a - b) ** 2))


def kmeans(x, k, max_iters=100):
    # Generated with Claude
    x = np.array(x)
    n_samples = x.shape[0]

    # randomly pick k initial centroids from the data points
    rng = np.random.default_rng()
    idx = rng.choice(n_samples, k, replace=False)
    centroids = x[idx]

    for _ in range(max_iters):
        # assign each point to the nearest centroid
        labels = []
        for point in x:
            dists = [distance(point, c) for c in centroids]
            labels.append(np.argmin(dists))
        labels = np.array(labels)

        # recompute centroids as the mean of assigned points
        new_centroids = []
        for i in range(k):
            cluster_points = x[labels == i]
            if len(cluster_points) > 0:
                new_centroids.append(mean(cluster_points))
            else:
                new_centroids.append(centroids[i])  # keep old centroid if cluster is empty
        new_centroids = np.array(new_centroids)

        # stop if centroids didn't move
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels, centroids


if __name__ == "__main__":
    d = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="marketing_campaign")

    cols = ["Income", "Recency", "MntWines", "MntFruits",
            "MntMeatProducts", "MntFishProducts",
            "MntSweetProducts", "MntGoldProds"]
    x = d[cols].dropna().values  # matches A11.py's preprocessing

    # NOTE: kmeans() returns (labels, centroids) in this order - unpack
    # accordingly, unlike A11.py which returns (centroid, label).
    labels, centroids = kmeans(x, 3)
    print("Cluster Sizes:")
    print(np.bincount(labels.astype(int)))
    print("\nCentroids:")
    print(centroids)
