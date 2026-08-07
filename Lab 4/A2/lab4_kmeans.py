"""
Lab 4 - AI-assisted distance/kmeans, pulled from A1_11.py.

mean() here is the axis=0 version (reduces a 2D array of points down to a
single centroid row) as needed by kmeans - distinct from the flat mean()
in lab4_stats.py.
"""
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

    rng = np.random.default_rng()
    idx = rng.choice(n_samples, k, replace=False)
    centroids = x[idx]

    for _ in range(max_iters):
        labels = []
        for point in x:
            dists = [distance(point, c) for c in centroids]
            labels.append(np.argmin(dists))
        labels = np.array(labels)

        new_centroids = []
        for i in range(k):
            cluster_points = x[labels == i]
            if len(cluster_points) > 0:
                new_centroids.append(mean(cluster_points))
            else:
                new_centroids.append(centroids[i])
        new_centroids = np.array(new_centroids)

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels, centroids
