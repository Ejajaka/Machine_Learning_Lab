import pandas as pd
from sklearn.neural_network import MLPClassifier
# Training MLP
def train(x, t, hidden):
    model = MLPClassifier(hidden_layer_sizes=(hidden,), activation="tanh", solver="lbfgs", max_iter=2000, random_state=1)
    model.fit(x, t)
    return model
# Predict
def predict(model, x):
    return model.predict(x)

data = pd.read_csv("ACTG175.csv")
x = data[["age", "wtkg"]]
t = data["treat"]
# MLP
model = train(x, t, 4)
y = predict(model, x)
print("Actual values:")
print(t.values)
print("\nPredicted values:")
print(y)
accuracy = model.score(x, t)
print("\nAccuracy:", accuracy * 100, "%")