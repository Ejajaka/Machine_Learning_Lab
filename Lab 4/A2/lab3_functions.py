"""
Lab 3 - manually written implementations (no AI assistance), pulled from
A2.py/A3.py (label, onehot), A4-A6.py (minkowski), A7.py (dot, norm),
A8-A10.py (mean, variance, std), A11.py (distance, kmeans).

Consolidated into one module purely so the functions can be imported and
unit tested; the logic inside each function is unchanged from the original
lab scripts.
"""
import numpy as np


def label(s):
    c = sorted(s)
    m = {}
    i = 0
    for x in c:
        m[x] = i
        i += 1
    return s.map(m), m


def onehot(d, col):
    n = d.copy()
    vals = d[col]
    for x in vals:
        l = []
        for v in d[col]:
            if v == x:
                l.append(1)
            else:
                l.append(0)
        n[str(x)] = l
    n = n.drop(columns=[col])
    return n


def minkowski(x1, x2, p):
    distance = np.sum(np.abs(x1 - x2) ** p) ** (1 / p)
    return distance


def dot(a, b):
    s = 0
    for i in range(len(a)):
        s = s + a[i] * b[i]
    return s


def norm(a):
    s = 0
    for i in range(len(a)):
        s = s + a[i] * a[i]
    return np.sqrt(s)


def mean(x):
    return np.sum(x, axis=0) / len(x)


def variance(x):
    m = mean(x)
    return np.sum((x - m) ** 2, axis=0) / len(x)


def std(x):
    return np.sqrt(variance(x))


def distance(a, b):
    s = 0
    for i in range(len(a)):
        s = s + (a[i] - b[i]) ** 2
    return np.sqrt(s)


def kmeans(x, k):
    n = len(x)
    r = np.random.choice(n, k, replace=False)
    centroid = x[r]
    lab = np.zeros(n)
    for t in range(100):
        for i in range(n):
            best = 0
            mind = distance(x[i], centroid[0])
            for j in range(1, k):
                dd = distance(x[i], centroid[j])
                if dd < mind:
                    mind = dd
                    best = j
            lab[i] = best
        new = centroid.copy()
        for j in range(k):
            p = x[lab == j]
            if len(p) > 0:
                new[j] = mean(p)
        if np.array_equal(new, centroid):
            break
        centroid = new
    return centroid, lab
