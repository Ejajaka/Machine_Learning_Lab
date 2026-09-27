import math
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def forward(a, b, u1, u2, v1, v2, w1, w2):
    # calculate hidden layer
    h1 = sigmoid(a*u1 + b*v1)
    h2 = sigmoid(a*u2 + b*v2)
    # calculate output layer
    o = sigmoid(h1*w1 + h2*w2)
    return h1, h2, o
def backprop(a, b, t, h1, h2, o,u1, u2, v1, v2, w1, w2, lr):
    # calculate output error
    eo = t - o
    # calculate output delta
    d_o = eo * o * (1 - o)
    # calculate hidden layer deltas
    d_h1 = d_o * w1 * h1 * (1 - h1)
    d_h2 = d_o * w2 * h2 * (1 - h2)
    # update output weights
    w1 = w1 + lr * d_o * h1
    w2 = w2 + lr * d_o * h2
    # update hidden weights
    u1 = u1 + lr * d_h1 * a
    v1 = v1 + lr * d_h1 * b
    u2 = u2 + lr * d_h2 * a
    v2 = v2 + lr * d_h2 * b
    return u1, u2, v1, v2, w1, w2
def calc_sse(x, t, u1, u2, v1, v2, w1, w2):
    sse = 0
    for i in range(len(x)):
        a, b = x[i]
        h1, h2, o = forward(a, b,u1, u2, v1, v2,w1, w2)
        e = t[i] - o
        sse += e**2
    return sse
def train(x, t, u1, u2, v1, v2, w1, w2, lr):
    errs = []
    for ep in range(1, 1001):
        for i in range(len(x)):
            a, b = x[i]
            # do forward propagation
            h1, h2, o = forward(a, b,u1, u2, v1, v2,w1, w2)
            # do back propagation
            u1, u2, v1, v2, w1, w2 = backprop(a, b, t[i],h1, h2, o,u1, u2, v1, v2,w1, w2,lr)
        # calculate sse after each epoch
        sse = calc_sse(x, t,u1, u2, v1, v2,w1, w2)
        errs.append(sse)
        # check convergence
        if sse <= 0.002:
            break
    return u1, u2, v1, v2, w1, w2, ep, errs
# set AND gate data
x = [[0, 0],[0, 1],[1, 0],[1, 1]]
t = [0, 0, 0, 1]
# set initial weights
u1 = 0.5
u2 = -0.5
v1 = 0.5
v2 = -0.5
w1 = 0.5
w2 = 0.5
# set learning rate
lr = 0.05
# start training
u1, u2, v1, v2, w1, w2, ep, errs = train(x, t,u1, u2, v1, v2,w1, w2,lr)
print("Number of epochs:", ep)
print("\nFinal weights:")
print("u1 =", u1)
print("u2 =", u2)
print("v1 =", v1)
print("v2 =", v2)
print("w1  =", w1)
print("w2  =", w2)
# check predictions
print("\nAND Gate Output:")
for i in range(len(x)):
    a, b = x[i]
    h1, h2, o = forward(a, b,u1, u2, v1, v2,w1, w2)
    y = 1 if o >= 0.5 else 0
    print(a, b,"->",round(o, 4),"->",y)
plt.plot(range(1, ep + 1),errs)
plt.xlabel("Epoch")
plt.ylabel("SSE")
plt.title("Epoch vs SSE - Backpropagation")
plt.grid()
plt.show()