"""
A3 - Performance comparison of two kmeans implementations:
    - A11.py    -> your own (manual) kmeans
    - A1_11.py  -> AI-assisted kmeans

Run this file from inside the same folder as A11.py and A1_11.py.

NOTE: if A11.py / A1_11.py have top-level code (reading a CSV, plotting,
printing results) sitting outside a `if __name__ == "__main__":` guard,
that code will execute the moment they're imported below. If you get
unwanted output/plots/errors when running this script, wrap that
top-level code in your two files with:

    if __name__ == "__main__":
        ...

so only the function definitions load on import.
"""
# Generated with Claude
import time

import numpy as np
import matplotlib.pyplot as plt

from A11 import kmeans as kmeans_own
from A1_11 import kmeans as kmeans_ai


def make_dataset(n_points, n_clusters=4, seed=0):
    """Synthetic dataset: n_clusters gaussian blobs, n_points total."""
    rng = np.random.default_rng(seed)
    per_cluster = n_points // n_clusters
    centers = rng.uniform(0, 100, size=(n_clusters, 2))
    blobs = [c + rng.normal(0, 3, size=(per_cluster, 2)) for c in centers]
    return np.vstack(blobs)


def time_kmeans(func, x, k, repeats=5):
    """Run func(x, k) `repeats` times, return (mean_seconds, std_seconds)."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(x, k)
        end = time.perf_counter()
        times.append(end - start)
    return float(np.mean(times)), float(np.std(times))


def run_comparison(sizes, k=4, repeats=5):
    results = []
    for n in sizes:
        x = make_dataset(n, n_clusters=k)

        mean_own, std_own = time_kmeans(kmeans_own, x, k, repeats)
        mean_ai, std_ai = time_kmeans(kmeans_ai, x, k, repeats)

        results.append({
            "n": n,
            "own_mean": mean_own, "own_std": std_own,
            "ai_mean": mean_ai, "ai_std": std_ai,
        })

        print(
            f"n={n:6d} | own: {mean_own*1000:8.2f} ms (+/-{std_own*1000:5.2f}) "
            f"| ai: {mean_ai*1000:8.2f} ms (+/-{std_ai*1000:5.2f}) "
            f"| speedup (own/ai): {mean_own/mean_ai:5.2f}x"
        )
    return results


def plot_results(results):
    n_vals = [r["n"] for r in results]
    own_vals = [r["own_mean"] * 1000 for r in results]
    ai_vals = [r["ai_mean"] * 1000 for r in results]
    own_std = [r["own_std"] * 1000 for r in results]
    ai_std = [r["ai_std"] * 1000 for r in results]

    plt.figure(figsize=(8, 5))
    plt.errorbar(n_vals, own_vals, yerr=own_std, marker="o", label="Own (A11)")
    plt.errorbar(n_vals, ai_vals, yerr=ai_std, marker="s", label="AI-assisted (A1_11)")
    plt.xlabel("Number of data points (n)")
    plt.ylabel("Runtime (ms)")
    plt.title("K-Means Runtime: Own vs AI-Assisted Implementation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("kmeans_runtime_comparison.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    SIZES = [100, 500, 1000, 2000, 5000]
    K = 4
    REPEATS = 5

    results = run_comparison(SIZES, k=K, repeats=REPEATS)
    plot_results(results)