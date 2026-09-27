import pandas as pd
import matplotlib.pyplot as plt
# summation function
def summation(x, w, bias):
    result = bias
    for i in range(len(x)):
        result = result + x[i] * w[i]
    return result
# activation function
def step(x):
    if x >= 0:
        return 1
    else:
        return 0
# load data
data = pd.read_csv("ACTG175.csv")
# set input and target data
X = data[["age", "wtkg"]]
Y = data["treat"]
# set initial weights
w01 = 10
w11 = 0.2
w21 = -0.75
w02 = 10
w12 = 0.2
w22 = -0.75
# set learning rate
alpha = 0.05
errors = []
# start training
for epoch in range(1000):
    for i in range(len(X)):
        x1 = X.iloc[i, 0]
        x2 = X.iloc[i, 1]
        target = Y.iloc[i]
        # set target values for two outputs
        if target == 0:
            t1 = 1
            t2 = 0
        else:
            t1 = 0
            t2 = 1
        # calculate summation for both outputs
        total1 = summation([x1, x2], [w11, w21], w01)
        total2 = summation([x1, x2], [w12, w22], w02)
        # find prediction
        prediction1 = step(total1)
        prediction2 = step(total2)
        # calculate error
        error1 = t1 - prediction1
        error2 = t2 - prediction2
        # update weights
        w01 = w01 + alpha * error1
        w11 = w11 + alpha * error1 * x1
        w21 = w21 + alpha * error1 * x2
        w02 = w02 + alpha * error2
        w12 = w12 + alpha * error2 * x1
        w22 = w22 + alpha * error2 * x2
    sse = 0
# checking performance
    for i in range(len(X)):
        x1 = X.iloc[i, 0]
        x2 = X.iloc[i, 1]
        target = Y.iloc[i]
        # set target values for two outputs
        if target == 0:
            t1 = 1
            t2 = 0
        else:
            t1 = 0
            t2 = 1
        # calculate summation for both outputs
        total1 = summation([x1, x2], [w11, w21], w01)
        total2 = summation([x1, x2], [w12, w22], w02)
        # find prediction
        prediction1 = step(total1)
        prediction2 = step(total2)
        # calculate error
        error1 = t1 - prediction1
        error2 = t2 - prediction2
        sse = sse + error1 ** 2 + error2 ** 2
    errors.append(sse)
    print("Epoch:", epoch + 1, "Error:", sse)
    # check convergence
    if sse <= 0.002:
        break
print("\nFinal Weights:")
print("Output 1:")
print("W01 =", w01)
print("W11 =", w11)
print("W21 =", w21)
print("\nOutput 2:")
print("W02 =", w02)
print("W12 =", w12)
print("W22 =", w22)
print("\nEpochs needed:", epoch + 1)
# plot error
plt.plot(range(1, len(errors) + 1), errors)
plt.xlabel("Epoch")
plt.ylabel("Sum-Square Error")
plt.title("Epoch vs Error")
plt.show()