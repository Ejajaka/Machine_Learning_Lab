import numpy as np
# summation function
def summation(ip, w, bias):
    result = bias
    for i in range(len(ip)):
        result = result + ip[i] * w[i]
    return result
# activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# set XOR gate data
x = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
t = np.array([0, 1, 1, 0])
# set learning rate
lr = 0.05
# set initial weights
u1 = 0.01
u2 = 0.02
u3 = 0.03
u4 = 0.04
u5 = 0.05
u6 = -0.01
# set initial bias
b1 = 0.01
b2 = 0.01
b3 = 0.01
# start training
for ep in range(10000):
    err = 0
    for i in range(4):
        a = x[i][0]
        b = x[i][1]
        target = t[i]
        # calculate hidden layer output
        h1 = sigmoid(summation([a, b], [u1, u2], b1))
        h2 = sigmoid(summation([a, b], [u3, u4], b2))
        # calculate output layer
        y = sigmoid(summation([h1, h2], [u5, u6], b3))
        e = target - y
        err = err + e ** 2
        # update weights
        d_o = e * y * (1 - y)
        d_h1 = h1 * (1 - h1) * u5 * d_o
        d_h2 = h2 * (1 - h2) * u6 * d_o
        u5 = u5 + lr * d_o * h1
        u6 = u6 + lr * d_o * h2
        b3 = b3 + lr * d_o
        u1 = u1 + lr * d_h1 * a
        u2 = u2 + lr * d_h1 * b
        b1 = b1 + lr * d_h1
        u3 = u3 + lr * d_h2 * a
        u4 = u4 + lr * d_h2 * b
        b2 = b2 + lr * d_h2
    # check convergence
    if err <= 0.002:
        break
print("Epochs:", ep + 1)
print("Error:", err)
print("\nFinal Weights:")
print("u1 =", u1)
print("u2 =", u2)
print("u3 =", u3)
print("u4 =", u4)
print("u5 =", u5)
print("u6 =", u6)