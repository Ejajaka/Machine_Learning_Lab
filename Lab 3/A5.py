import numpy as nump
import matplotlib.pyplot as plt

def minkowski(x1, x2, p):
    distance= nump.sum(nump.abs(x1 - x2) ** p) ** (1 / p)
    return distance

x1 = nump.array([7, 3, 4])
x2 = nump.array([17, 6, 9])
pvalues = list(range(1, 11))
distances = [minkowski(x1, x2, p) for p in pvalues]
for p, dist in zip(pvalues, distances):
    print(f"p={p}: distance={dist}")
plt.plot(pvalues, distances, marker="o")
plt.xlabel("p")
plt.ylabel("Distance")
plt.title("Minkowski distance vs p")
plt.grid(True)
plt.show()