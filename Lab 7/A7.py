import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay

data = pd.read_csv("ACTG175.csv")
TARGET = "treat"
PAIR = ["cd40", "age"]
X = data[PAIR]
y = data[TARGET]
clf = DecisionTreeClassifier(random_state=1)
clf.fit(X, y)
plt.figure(figsize=(8, 6))
DecisionBoundaryDisplay.from_estimator(clf,X,response_method="predict",alpha=0.5)
plt.scatter(X["cd40"],X["age"],c=y,edgecolor="black",s=30)
plt.xlabel("CD40")
plt.ylabel("Age")
plt.title("Decision Boundary using CD40 and Age")
plt.show()