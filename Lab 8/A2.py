
import pandas as pd
import matplotlib.pyplot as plt

# summation function
def summation(x, w, bias):
    result = bias
    for i in range(len(x)):
        result = result + x[i] * w[i]
    return result

# step activation function
def step(x):
    if x >= 0:
        return 1
    else:
        return 0

# AND gate training data
data = pd.DataFrame({
    "x1": [0, 0, 1, 1],
    "x2": [0, 1, 0, 1],
    "target": [0, 0, 0, 1]
})

X = data[["x1", "x2"]]
Y = data["target"]

# initial weights and learning rate
w0 = 10
w1 = 0.2
w2 = -0.75
alpha = 0.05

errors = []

# training
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

    # calculate SSE after each epoch
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

# final weights
print("\nFinal weights:")
print("W0 =", w0)
print("W1 =", w1)
print("W2 =", w2)
print("Epochs needed:", epoch + 1)

# plot epochs against SSE
plt.plot(range(1, len(errors) + 1), errors, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Sum-Squared Error")
plt.title("Epoch vs Error for AND Gate")
plt.grid(True)
plt.show()
