import numpy as np

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

x1 = [7, 3, 4]
x2 = [17, 6, 9]
o_dot = dot(x1, x2)
n_dot = np.dot(x1, x2)
o_norm1 = norm(x1)
n_norm1 = np.linalg.norm(x1)
o_norm2 = norm(x2)
n_norm2 = np.linalg.norm(x2)
print("Dot Product:", o_dot)
print("Dot Product_np:", n_dot)
print("Norm of x1:", o_norm1)
print("Norm of x1_np:", n_norm1)
print("Norm of x2:", o_norm2)
print("Norm of x2_np:", n_norm2)
