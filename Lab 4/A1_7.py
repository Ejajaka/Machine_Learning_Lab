import numpy as np

def dot(a, b):
    # Generated with Claude
    a = np.array(a)
    b = np.array(b)
    return np.sum(a * b)

def norm(a):
    # Generated with Claude
    a = np.array(a)
    return np.sqrt(np.sum(a ** 2))

x1 = [7, 3, 4]
x2 = [17, 6, 9]
odot = dot(x1, x2)
ndot = np.dot(x1, x2)
onorm1 = norm(x1)
nnorm1 = np.linalg.norm(x1)
onorm2 = norm(x2)
nnorm2 = np.linalg.norm(x2)
print("Dot Product:", odot)
print("Dot Product using np:", ndot)
print("Norm of x1:", onorm1)
print("Norm of x1 using np:", nnorm1)
print("Norm of x2:", onorm2)
print("Norm of x2 using np:", nnorm2)