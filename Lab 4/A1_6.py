import numpy as nump
from scipy.spatial.distance import minkowski as minkowski1

def minkowski(x1, x2, p):
    # Generated with Claude
    x1 = nump.array(x1)
    x2 = nump.array(x2)
    return nump.sum(nump.abs(x1 - x2) ** p) ** (1 / p)

x1 = nump.array([7, 3, 4])
x2 = nump.array([17, 6, 9])
for p in range(1, 11):
    o = minkowski(x1, x2, p)
    l = minkowski1(x1, x2, p)
    print(f"p={p}: own={o}, scipy={l}")