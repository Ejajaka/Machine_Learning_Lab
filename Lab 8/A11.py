import numpy as np
from sklearn.neural_network import MLPClassifier
# Input values
X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
# AND gate output
AND = np.array([0, 0, 0, 1])
# XOR gate output
XOR = np.array([0, 1, 1, 0])
# AND Gate
and_model = MLPClassifier(hidden_layer_sizes=(2,),activation="tanh",solver="lbfgs",max_iter=2000,random_state=1)
and_model.fit(X, AND)
and_prediction = and_model.predict(X)
print("AND Gate")
print("Input Actual Predicted")
for i in range(len(X)):
    print(X[i],"   ", AND[i], "  ", and_prediction[i])
# XOR Gate
xor_model = MLPClassifier(hidden_layer_sizes=(4,),activation="tanh",solver="lbfgs",max_iter=2000,random_state=1)
xor_model.fit(X, XOR)
xor_prediction = xor_model.predict(X)
print("\nXOR Gate")
print("Input Actual Predicted")
for i in range(len(X)):
    print(X[i], "   ", XOR[i], "  ", xor_prediction[i])