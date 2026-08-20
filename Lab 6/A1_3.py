import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv("simulation_500.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
# spliting the dataset into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)
print("Total samples :", len(X))
print("Training samples :", len(X_train))
print("Testing samples :", len(X_test))