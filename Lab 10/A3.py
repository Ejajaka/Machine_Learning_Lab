import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

def load(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna(subset=["arms"])
    features = data.drop(columns=["arms", "Unnamed: 0"], errors="ignore")
    features = features.select_dtypes(include="number")
    target = data["arms"]
    return features, target

def split(features, target):
    return train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

# function to handle missing values and standardise features
def scale_features(X_train, X_test):
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# function to find the number of PCA components needed to retain 99% variance
def num_components(X_train_scaled, variance_to_keep=0.99):
    pca_full = PCA().fit(X_train_scaled)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    return int(np.argmax(cumulative_variance >= variance_to_keep) + 1)

# function to apply PCA to training and testing data
def apply_pca(X_train_scaled, X_test_scaled, num_components):
    pca = PCA(n_components=num_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    return X_train_pca, X_test_pca

# function to return the classifiers to compare
def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(),
    }

# function to train and compare each model using original features and PCA features
def compare_models(models, feature_sets, y_train, y_test):
    results = []
    for model_name, model in models.items():
        for feature_label, (X_train, X_test) in feature_sets.items():
            from sklearn.base import clone
            model_copy = clone(model)
            model_copy.fit(X_train, y_train)
            y_pred = model_copy.predict(X_test)
            results.append({
                "Model": model_name,
                "Features": feature_label,
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            })
    return pd.DataFrame(results)

features, target = load("ACTG175.csv")
X_train, X_test, y_train, y_test = split(features, target)
X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
num_components = num_components(X_train_scaled, variance_to_keep=0.99)
X_train_pca, X_test_pca = apply_pca(X_train_scaled, X_test_scaled, num_components)
print("Original features:", X_train.shape[1])
print("Components retained for 99% variance:", num_components)
feature_sets = {
    "Original": (X_train_scaled, X_test_scaled),
    "PCA": (X_train_pca, X_test_pca),
}
results_table = compare_models(get_models(), feature_sets, y_train, y_test)
print("\nModel Comparison Results")
print(results_table.round(4).to_string(index=False))