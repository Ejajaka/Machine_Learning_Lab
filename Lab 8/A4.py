import matplotlib.pyplot as plt
# activation function
def step(z):
    if z >= 0:
        return 1
    else:
        return 0
# summation unit
def summ(p, q, c, u, v):
    return c + u*p + v*q
# calculate error
def calc_err(r, y):
    return r - y
# update weights
def upd(c, u, v, p, q, e, lr):
    c = c + lr*e
    u = u + lr*e*p
    v = v + lr*e*q
    return c, u, v
# calculate SSE
def sse_calc(d, r, c, u, v):
    sse = 0
    for i in range(len(d)):
        p, q = d[i]
        net = summ(p, q, c, u, v)
        y = step(net)
        e = calc_err(r[i], y)
        sse += e**2
    return sse
# train the perceptron
def train(d, r, c, u, v, lr):
    for ep in range(1, 1001):
        for i in range(len(d)):
            p, q = d[i]
            net = summ(p, q, c, u, v)
            y = step(net)
            e = calc_err(r[i], y)
            c, u, v = upd(c, u, v, p, q, e, lr)

        sse = sse_calc(d, r, c, u, v)
        if sse <= 0.002:
            break
    return ep

# set AND gate data
d = [[0, 0], [0, 1], [1, 0], [1, 1]]
r = [0, 0, 0, 1]
c = 10
u = 0.2
v = -0.75
# set learning rates
rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
iters = []
# train for each learning rate
for lr in rates:
    # reset weights for each learning rate
    ep = train(d, r, c, u, v, lr)
    iters.append(ep)
print("learning rate  iterations")
for i in range(len(rates)):
    print(rates[i], "          ", iters[i])
plt.plot(rates, iters, marker='o')
plt.xlabel("Learning Rate")
plt.ylabel("Number of Iterations")
plt.title("Learning Rate vs Number of Iterations")
plt.grid()
plt.show()