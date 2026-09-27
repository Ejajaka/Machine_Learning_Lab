import pandas as pd
import matplotlib.pyplot as plt

def summation(x, w, bias):
    result = bias
    for i in range(len(x)):
        result = result + x[i] * w[i]
    return result
def step(x):
    if x >= 0:
        return 1
    else:
        return 0

data = pd.read_csv("ACTG175.csv")
X = data[["age", "wtkg"]]
Y = data["treat"]
w0 = 10
w1 = 0.2
w2 = -0.75
alpha = 0.05
errors = []
#training 
for epoch in range(1000):
    for i in range(len(X)):
        x1 = X.iloc[i, 0]
        x2 = X.iloc[i, 1]
        target = Y.iloc[i]
        total = summation([x1, x2], [w1, w2], w0)
        prediction = step(total)
        error = target - prediction
        w0 = w0 + alpha * error
        w1 = w1 + alpha * error * x1
        w2 = w2 + alpha * error * x2
    sse = 0
    for i in range(len(X)):
        x1 = X.iloc[i, 0]
        x2 = X.iloc[i, 1]
        target = Y.iloc[i]
        total = summation([x1, x2], [w1, w2], w0)
        prediction = step(total)
        error = target - prediction
        sse = sse + error ** 2
    errors.append(sse)
    print("Epoch:", epoch + 1, "Error:", sse)
    if sse <= 0.002:
        break

print("\nfinal weights:")
print("W0 =", w0)
print("W1 =", w1)
print("W2 =", w2)
print("epochs needed:", epoch + 1)
plt.plot(range(1, len(errors) + 1), errors)
plt.xlabel("epoch")
plt.ylabel("sum-square error")
plt.title("epoch vs error")
plt.show()