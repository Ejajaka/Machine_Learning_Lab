import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics
from six import StringIO
import pydotplus

data = pd.read_csv("ACTG175.csv")
features = ["age", "wtkg", "hemo", "homo", "drugs", "karnof",
            "oprior", "z30", "zprior", "preanti", "race", "gender",
            "str2", "strat", "symptom", "cd40", "cd80"]
X = data[features]
y = data["treat"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
dot_data = StringIO()
export_graphviz(clf,out_file=dot_data,filled=True,rounded=True,special_characters=True,feature_names=features,class_names=["0", "1"])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("ACTG175_tree.png")
print("Decision tree saved as ACTG175_tree.png")