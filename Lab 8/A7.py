import numpy as np
# activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# normalize the data
def normalize(x):
    mn = np.min(x, axis=0)
    mx = np.max(x, axis=0)
    return (x - mn) / (mx - mn)
# calculate net input
def summ(x, w, b):
    return np.dot(x, w) + b
# calculate error
def calc_err(t, y):
    return t - y
# update weights
def upd(w, b, x, e, y, lr):
    d = e * y * (1 - y)
    w = w + lr * d * x
    b = b + lr * d
    return w, b
# train the perceptron
def train(x, t, w, b, lr, max_ep=10000):
    for ep in range(1, max_ep + 1):
        for i in range(len(x)):
            net = summ(x[i], w, b)
            y = sigmoid(net)
            e = calc_err(t[i], y)
            w, b = upd(w, b, x[i], e, y, lr)
    return w, b, ep
# get perceptron prediction
def predict_perceptron(x, w, b):
    y = sigmoid(summ(x, w, b))
    if y >= 0.5:
        return 1
    else:
        return 0
# calculate pseudo-inverse
def pseudo_inverse(x, t):
    # add bias column
    xb = np.c_[np.ones(len(x)), x]
    # calculate weights
    w = np.linalg.pinv(xb) @ t
    return w
# get pseudo-inverse prediction
def predict_pinv(x, w):
    xb = np.c_[np.ones(len(x)), x]
    y = xb @ w
    result = []
    for val in y:
        if val >= 0.5:
            result.append(1)
        else:
            result.append(0)
    return result
# calculate accuracy
def accuracy(t, y):
    return np.mean(np.array(t) == np.array(y)) * 100
# set customer data
x = np.array([[20, 6, 2, 386],[16, 3, 6, 289],[27, 6, 2, 393],[19, 1, 2, 110],[24, 4, 2, 280],[22, 1, 5, 167],[15, 4, 2, 271],[18, 4, 2, 274],[21, 1, 4, 148],[16, 2, 4, 198]], dtype=float)
# set target values
t = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])
# normalize the inputs
x = normalize(x)
# set perceptron weights and bias
w = np.zeros(4)
b = 0
lr = 10
w, b, ep = train(x, t, w, b, lr)
yp = []
for i in range(len(x)):
    yp.append(predict_perceptron(x[i], w, b))
# calculate pseudo-inverse weights
wp = pseudo_inverse(x, t)
ypi = predict_pinv(x, wp)
print("customer actual perceptron pseudo-inverse")
for i in range(len(x)):
    actual = "high" if t[i] == 1 else "low"
    p = "high" if yp[i] == 1 else "low"
    pi = "high" if ypi[i] == 1 else "low"
    print("c_" + str(i + 1)," ",actual," ",p," ",pi)
acc_p = accuracy(t, yp)
acc_pi = accuracy(t, ypi)
print("\naccuracy:")
print("perceptron     :", acc_p, "%")
print("pseudo-inverse :", acc_pi, "%")