"""
Lab 4 - AI-assisted mean/variance/std, pulled from A1_8.py/A1_9.py/A1_10.py.

Note this is a *flat* (no axis) implementation, unlike the kmeans-oriented
mean() in lab4_kmeans.py which reduces along axis=0. Both versions came out
of the AI-assisted rewrite of last week's code depending on what each
script needed, so both are kept and tested separately here rather than
silently merged into one "correct" version.
"""
import numpy as np


def mean(x):
    # Generated with Claude
    x = np.array(x)
    return np.sum(x) / len(x)


def variance(x):
    # Generated with Claude
    x = np.array(x)
    m = mean(x)
    return np.sum((x - m) ** 2) / len(x)


def std(x):
    return np.sqrt(variance(x))
