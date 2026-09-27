import math
import matplotlib.pyplot as plt
# activation functions
def step(z):
    if z >= 0:
        return 1
    else:
        return 0
def bipolar(z):
    if z > 0:
        return 1
    elif z < 0:
        return -1
    else:
        return 0
def sigmoid(z):
    return 1 / (1 + math.exp(-z))
def relu(z):
    if z > 0:
        return z
    else:
        return 0
# calculate net input
def summ(p, q, c, u, v):
    return c + u*p + v*q
def calc_err(r, y):
    return r - y
# update weights and bias
def upd(c, u, v, p, q, e, lr):
    c = c + lr*e
    u = u + lr*e*p
    v = v + lr*e*q
    return c, u, v
# calculate sse
def sse_calc(d, r, c, u, v, act):
    sse = 0
    for i in range(len(d)):
        p, q = d[i]
        net = summ(p, q, c, u, v)
        y = act(net)
        e = calc_err(r[i], y)
        sse += e**2
    return sse
# train the perceptron
def train(d, r, act, c, u, v, lr):
    errs = []
    for ep in range(1, 1001):
        for i in range(len(d)):
            p, q = d[i]
            net = summ(p, q, c, u, v)
            y = act(net)
            e = calc_err(r[i], y)
            c, u, v = upd(c, u, v, p, q, e, lr)
        sse = sse_calc(d, r,c, u, v,act)
        errs.append(sse)
        if sse <= 0.002:
            break
    return ep, c, u, v, errs
    
# set xor gate data
d = [[0, 0],[0, 1],[1, 0],[1, 1]]
r = [0, 1, 1, 0]
# set initial weights
c = 10
u = 0.2
v = -0.75
# set learning rate
lr = 0.05
# train with step function
ep_st, c1, u1, v1, err_st = train(d, r, step,c, u, v, lr)
# train with bipolar step
rb = [-1, 1, 1, -1]
ep_bp, c2, u2, v2, err_bp = train(d, rb, bipolar,c, u, v, lr)
# train with sigmoid
ep_sg, c3, u3, v3, err_sg = train(d, r, sigmoid,c, u, v, lr)
# train with relu
ep_rl, c4, u4, v4, err_rl = train(d, r, relu,c, u, v, lr)
print("XOR Gate")
print("\nEpochs:")
print("Step         :", ep_st)
print("Bipolar Step :", ep_bp)
print("Sigmoid      :", ep_sg)
print("ReLU         :", ep_rl)
print("\nFinal SSE:")
print("Step         :", err_st[-1])
print("Bipolar Step :", err_bp[-1])
print("Sigmoid      :", err_sg[-1])
print("ReLU         :", err_rl[-1])
plt.plot(range(1, len(err_st) + 1),err_st,label="Step")
plt.plot(range(1, len(err_bp) + 1),err_bp,label="Bipolar Step")
plt.plot(range(1, len(err_sg) + 1),err_sg,label="Sigmoid")
plt.plot(range(1, len(err_rl) + 1),err_rl,label="ReLU")
plt.xlabel("Epoch")
plt.ylabel("SSE")
plt.title("XOR Gate - Epoch vs SSE")
plt.legend()
plt.grid()
plt.show()