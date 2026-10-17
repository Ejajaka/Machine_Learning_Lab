import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

def load(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna(subset=["arms"])
    features = data.drop(columns=["arms", "Unnamed: 0"], errors="ignore")
    target = data["arms"]
    return features, target

def split(features, target):
    return train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

#function to return the base classifiers as a list of (name, estimator).
def base_models():
    return [("logistic_regression",make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))),("decision_tree",make_pipeline(SimpleImputer(strategy="median"), DecisionTreeClassifier(random_state=42))),("random_forest",make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=100, random_state=42))),("svm",make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVC(random_state=42))),]

#function to return the final estimators.
def meta_models():
    return {"Logistic Regression": make_pipeline(SimpleImputer(strategy="median"), LogisticRegression(max_iter=1000)),"Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),"Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),}

#function to train stacking classifier 
def evaluate_stacking(base_models, meta_models, X_train, X_test, y_train, y_test):
    results = []
    for meta_name, meta_model in meta_models.items():
        stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
        stacking_model.fit(X_train, y_train)
        y_pred = stacking_model.predict(X_test)
        results.append({"Meta Model": meta_name,"Accuracy": accuracy_score(y_test, y_pred),"F1 Score": f1_score(y_test, y_pred, average="weighted"),})
    return pd.DataFrame(results)

features, target = load("ACTG175.csv")
X_train, X_test, y_train, y_test = split(features, target)
results_table = evaluate_stacking(base_models(), meta_models(), X_train, X_test, y_train, y_test)
print("Stacking Classifier Results")
print(results_table.round(4).to_string(index=False))