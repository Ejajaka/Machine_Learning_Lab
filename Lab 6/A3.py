"""
A3. Performance comparison of three k-NN implementations:
    1. Own code        -> the student-written version (as originally submitted,
                           nested-loop distances + manual tuple sort + majority vote)
    2. Scikit-learn     -> sklearn.neighbors.KNeighborsClassifier (inbuilt)
    3. GenAI code       -> Claude-generated version (vectorized NumPy distances,
                           np.argsort neighbor selection, dict-based vote count)

Metrics: Accuracy, Precision, Recall, F-score (weighted average, sklearn.metrics)
Timing:  fit+predict wall-clock time, averaged over 10 runs (10 different
         train/test splits), reported in milliseconds.

Data: project dataset "simulation_500.csv" (last column = label).
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

RANDOM_SEEDS = list(range(10))   # 10 runs
K = 3
DATA_PATH = "simulation_500.csv"


# ---------------------------------------------------------------------------
# 1. OWN CODE  -- student-written version (as originally submitted)
# ---------------------------------------------------------------------------
def own_eucl(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def own_predict(X_train, y_train, X_test, k):
    y_pred = []
    for pt in X_test:
        distance = []
        for i in range(len(X_train)):
            dist = own_eucl(pt, X_train[i])
            distance.append((dist, i))
        distance.sort(key=lambda x: x[0])
        neighbour = [distance[i][1] for i in range(k)]
        labels = [y_train[i] for i in neighbour]
        prediction = max(set(labels), key=labels.count)
        y_pred.append(prediction)
    return y_pred


# ---------------------------------------------------------------------------
# 2. SCIKIT-LEARN CODE -- inbuilt KNeighborsClassifier
# ---------------------------------------------------------------------------
def sklearn_predict(X_train, y_train, X_test, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model.predict(X_test)


# ---------------------------------------------------------------------------
# 3. GENAI CODE -- Claude-generated, vectorized version
# ---------------------------------------------------------------------------
def genai_eucl(x1, x2):
    diff = x1 - x2
    squared = diff ** 2
    total = squared.sum()
    return total ** 0.5


def genai_knn(data, pt, k):
    distances = np.sqrt(np.sum((data - pt) ** 2, axis=1))
    nearest_indices = np.argsort(distances)[:k]
    return nearest_indices.tolist()


def genai_predict(X_train, y_train, X_test, k):
    y_pred = []
    for pt in X_test:
        neighbour = genai_knn(X_train, pt, k)
        labels = [y_train[i] for i in neighbour]
        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        prediction = max(counts, key=counts.get)
        y_pred.append(prediction)
    return y_pred


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F-score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def run_comparison():
    data = pd.read_csv(DATA_PATH)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    methods = {
        "Own Code": own_predict,
        "Scikit-learn": sklearn_predict,
        "GenAI Code": genai_predict,
    }

    results = {name: {"Accuracy": [], "Precision": [], "Recall": [], "F-score": [], "Time (ms)": []}
               for name in methods}

    for seed in RANDOM_SEEDS:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed
        )

        for name, func in methods.items():
            start = time.perf_counter()
            y_pred = func(X_train, y_train, X_test, K)
            elapsed_ms = (time.perf_counter() - start) * 1000

            metrics = evaluate(y_test, y_pred)
            for m_name, m_val in metrics.items():
                results[name][m_name].append(m_val)
            results[name]["Time (ms)"].append(elapsed_ms)

    # Average over the 10 runs
    summary = []
    for name in methods:
        row = {"Method": name}
        for metric in ["Accuracy", "Precision", "Recall", "F-score", "Time (ms)"]:
            row[metric] = np.mean(results[name][metric])
        summary.append(row)

    summary_df = pd.DataFrame(summary).set_index("Method")
    return summary_df, results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(summary_df, filename="knn_comparison_plots.png"):
    methods = summary_df.index.tolist()
    accuracy_metrics = ["Accuracy", "Precision", "Recall", "F-score"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left panel: grouped bar chart of accuracy metrics ---
    x = np.arange(len(accuracy_metrics))
    width = 0.25
    for i, method in enumerate(methods):
        values = summary_df.loc[method, accuracy_metrics].values
        axes[0].bar(x + (i - 1) * width, values, width, label=method, color=colors[i])

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(accuracy_metrics)
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Accuracy Metrics by Method")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # --- Right panel: computational time (log scale, since sklearn/GenAI << own code) ---
    times = summary_df["Time (ms)"].values
    bars = axes[1].bar(methods, times, color=colors)
    axes[1].set_ylabel("Average Time (ms, log scale)")
    axes[1].set_yscale("log")
    axes[1].set_title(f"Computational Time (avg of {len(RANDOM_SEEDS)} runs)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    for bar, t in zip(bars, times):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                      f"{t:.2f} ms", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"\nSaved comparison plots to {filename}")


if __name__ == "__main__":
    summary_df, raw_results = run_comparison()

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(f"\nk-NN Performance Comparison (k={K}, averaged over {len(RANDOM_SEEDS)} runs, "
          f"70/30 train-test split)\n")
    print(summary_df.to_string())

    summary_df.to_csv("knn_comparison_results.csv")
    print("\nSaved detailed results to knn_comparison_results.csv")

    plot_results(summary_df)