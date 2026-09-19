import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("ACTG175.csv")
features = ["age", "wtkg", "hemo", "homo", "drugs", "karnof",
            "oprior", "z30", "zprior", "preanti", "race", "gender",
            "str2", "strat", "symptom", "cd40", "cd80"]
X = data[features]
y = data["treat"]
clf = DecisionTreeClassifier(random_state=1)
param_grid = {"criterion": ["gini", "entropy"],"max_depth": [3, 5, 7, 10, None],"min_samples_split": [2, 5, 10],"min_samples_leaf": [1, 2, 4]}
grid = GridSearchCV(clf,param_grid,cv=5,scoring="accuracy") # checks all parameters and tell compares and finds best accuracy
grid.fit(X, y)
print("Best Hyperparameters:")
print(grid.best_params_)
print("Best Accuracy:", grid.best_score_)