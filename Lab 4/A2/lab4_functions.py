"""
Lab 4 - AI-assisted implementations, pulled from A1_2.py/A1_3.py (label,
onehot), A1_4-A1_6.py (minkowski), A1_7.py (dot, norm).

Consolidated into one module purely so the functions can be imported and
unit tested; the "Generated with Claude" comments are kept on the function
bodies to preserve the attribution from the original lab scripts.
"""
import numpy as np


def label(s):
    # Generated with Claude
    categories = sorted(s.unique())
    mapping = {category: idx for idx, category in enumerate(categories)}
    return s.map(mapping), mapping


def onehot(d, col):
    # Generated with Claude
    n = d.copy()
    for val in d[col].unique():
        n[str(val)] = (d[col] == val).astype(int)
    return n


def minkowski(x1, x2, p):
    # Generated with Claude
    x1 = np.array(x1)
    x2 = np.array(x2)
    return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)


def dot(a, b):
    # Generated with Claude
    a = np.array(a)
    b = np.array(b)
    return np.sum(a * b)


def norm(a):
    # Generated with Claude
    a = np.array(a)
    return np.sqrt(np.sum(a ** 2))
