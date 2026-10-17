import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load(file_path):
    data = pd.read_csv(file_path)
    return data.drop(columns=["arms", "Unnamed: 0"], errors="ignore").select_dtypes(include="number")

# function to compute the feature-to-feature correlation matrix
def compute_correlation(features):
    return features.corr()

# function to display the correlation matrix as an annotated heatmap
def plot(correlation_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

# function to find feature pairs whose absolute correlation exceeds the threshold
def high_correlations(correlation_matrix, threshold=0.7):
    pairs = []
    columns = correlation_matrix.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):  # upper triangle only, avoids duplicates
            value = correlation_matrix.iloc[i, j]
            if pd.notna(value) and abs(value) > threshold:
                pairs.append((columns[i], columns[j], round(value, 2)))
    return pairs

features = load("ACTG175.csv")
correlation_matrix = compute_correlation(features)
print("Correlation matrix")
print(correlation_matrix.round(2))
print("\nHighly correlated pairs (|r| > 0.7):")
for feature_1, feature_2, value in high_correlations(correlation_matrix):
    print(feature_1, "-", feature_2, value)
plot(correlation_matrix)