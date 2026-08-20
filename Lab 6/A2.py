"""
Unit tests for the modular KNN lab functions (this week's and last week's exercises).

Covers:
  - eucl(x1, x2)                         -> Euclidean distance
  - knn(data, pt, k)                     -> indices of k nearest neighbors
  - fit(X_train, y_train)                -> pass-through fit
  - predict(X_train, y_train, X_test, k) -> majority-vote prediction
  - weighted(data, y, pt, k) / distance-weighted predict -> weighted-vote prediction
  - score(y_test, y_pred)                -> accuracy

Run with:
    pytest test_knn_labs.py -v
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Reference implementations under test
# (mirrors the functions built across the lab exercises)
# ---------------------------------------------------------------------------

def eucl(x1, x2):
    diff = np.asarray(x1) - np.asarray(x2)
    squared = diff ** 2
    total = squared.sum()
    return total ** 0.5


def knn(data, pt, k):
    data = np.asarray(data)
    distances = np.sqrt(np.sum((data - pt) ** 2, axis=1))
    nearest_indices = np.argsort(distances)[:k]
    return nearest_indices.tolist()


def fit(X_train, y_train):
    return X_train, y_train


def predict_majority(X_train, y_train, X_test, k):
    y_pred = []
    for pt in X_test:
        neighbour = knn(X_train, pt, k)
        labels = [y_train[i] for i in neighbour]
        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        prediction = max(counts, key=counts.get)
        y_pred.append(prediction)
    return y_pred


def predict_weighted(X_train, y_train, X_test, k):
    y_pred = []
    for pt in X_test:
        distances = np.sqrt(np.sum((np.asarray(X_train) - pt) ** 2, axis=1))
        nearest_indices = np.argsort(distances)[:k]
        votes = {}
        for i in nearest_indices:
            dist = distances[i]
            weight = 1 / (dist + 1e-9)
            label = y_train[i]
            votes[label] = votes.get(label, 0) + weight
        prediction = max(votes, key=votes.get)
        y_pred.append(prediction)
    return y_pred


def score(y_test, y_pred):
    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)
    correct = np.sum(y_test_arr == y_pred_arr)
    return correct / len(y_test)


# ---------------------------------------------------------------------------
# Fixtures: small, hand-crafted datasets so expected results are known exactly
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_dataset():
    """
    5 points on a line, two classes.
    Index:  0    1    2    3    4
    X:     [0], [1], [2], [10],[11]
    y:      0    0    0    1    1
    """
    X = np.array([[0], [1], [2], [10], [11]], dtype=float)
    y = np.array([0, 0, 0, 1, 1])
    return X, y


@pytest.fixture
def tie_dataset():
    """
    Dataset designed so a majority vote among k=4 neighbors is tied 2-2,
    but distance weighting breaks the tie because one class's points
    are much closer.
    """
    X = np.array([[0], [1], [5], [6]], dtype=float)
    y = np.array([0, 1, 0, 1])
    return X, y


# ---------------------------------------------------------------------------
# eucl()
# ---------------------------------------------------------------------------

class TestEucl:
    def test_known_distance_1d(self):
        assert eucl(np.array([0]), np.array([3])) == pytest.approx(3.0)

    def test_known_distance_2d_3_4_5_triangle(self):
        assert eucl(np.array([0, 0]), np.array([3, 4])) == pytest.approx(5.0)

    def test_zero_distance_identical_points(self):
        p = np.array([2.5, -1.0, 7])
        assert eucl(p, p) == pytest.approx(0.0)

    def test_symmetry(self):
        a = np.array([1, 2, 3])
        b = np.array([4, -1, 0])
        assert eucl(a, b) == pytest.approx(eucl(b, a))

    def test_non_negative(self):
        a = np.array([-5, 10])
        b = np.array([3, -8])
        assert eucl(a, b) >= 0


# ---------------------------------------------------------------------------
# knn()
# ---------------------------------------------------------------------------

class TestKnn:
    def test_returns_k_indices(self, simple_dataset):
        X, _ = simple_dataset
        result = knn(X, np.array([0]), k=3)
        assert len(result) == 3

    def test_nearest_neighbors_correct(self, simple_dataset):
        X, _ = simple_dataset
        # Query point [0] -> nearest should be indices 0, 1, 2 (values 0,1,2)
        result = knn(X, np.array([0]), k=3)
        assert set(result) == {0, 1, 2}

    def test_neighbors_ordered_by_distance(self, simple_dataset):
        X, _ = simple_dataset
        result = knn(X, np.array([0]), k=3)
        assert result == [0, 1, 2]  # exact order: closest first

    def test_k_equals_dataset_size(self, simple_dataset):
        X, _ = simple_dataset
        result = knn(X, np.array([0]), k=len(X))
        assert len(result) == len(X)
        assert set(result) == set(range(len(X)))

    def test_query_point_in_dataset_returns_itself_first(self, simple_dataset):
        X, _ = simple_dataset
        result = knn(X, X[3], k=1)
        assert result == [3]


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

class TestFit:
    def test_returns_same_data_unchanged(self, simple_dataset):
        X, y = simple_dataset
        X_out, y_out = fit(X, y)
        assert np.array_equal(X_out, X)
        assert np.array_equal(y_out, y)


# ---------------------------------------------------------------------------
# predict() -- majority vote
# ---------------------------------------------------------------------------

class TestPredictMajority:
    def test_predicts_expected_class(self, simple_dataset):
        X, y = simple_dataset
        X_test = np.array([[1]])  # near class-0 cluster
        pred = predict_majority(X, y, X_test, k=3)
        assert pred == [0]

    def test_predicts_other_class(self, simple_dataset):
        X, y = simple_dataset
        X_test = np.array([[10.5]])  # near class-1 cluster
        pred = predict_majority(X, y, X_test, k=2)
        assert pred == [1]

    def test_multiple_test_points(self, simple_dataset):
        X, y = simple_dataset
        X_test = np.array([[0], [11]])
        pred = predict_majority(X, y, X_test, k=1)
        assert pred == [0, 1]

    def test_k1_matches_single_nearest_label(self, simple_dataset):
        X, y = simple_dataset
        pt = np.array([2])
        pred = predict_majority(X, y, np.array([pt]), k=1)
        nearest_idx = knn(X, pt, k=1)[0]
        assert pred == [y[nearest_idx]]


# ---------------------------------------------------------------------------
# predict() -- distance-weighted vote
# ---------------------------------------------------------------------------

class TestPredictWeighted:
    def test_no_division_by_zero_on_exact_match(self, simple_dataset):
        # test point identical to a training point -> distance 0 for that
        # neighbor; must not raise ZeroDivisionError / produce inf/nan
        X, y = simple_dataset
        X_test = np.array([X[0]])
        pred = predict_weighted(X, y, X_test, k=3)
        assert pred == [y[0]]
        assert np.isfinite(pred[0]) or isinstance(pred[0], (int, np.integer))

    def test_closer_class_wins_tie_break(self, tie_dataset):
        # majority (k=4) is a 2-2 tie, but class 0 points (indices 0,2)
        # are much closer to pt=0 than class 1 points -> weighted picks 0
        X, y = tie_dataset
        pt = np.array([0])
        pred = predict_weighted(X, y, np.array([pt]), k=4)
        assert pred == [0]

    def test_matches_majority_when_unambiguous(self, simple_dataset):
        X, y = simple_dataset
        X_test = np.array([[1]])
        pred_w = predict_weighted(X, y, X_test, k=3)
        pred_m = predict_majority(X, y, X_test, k=3)
        assert pred_w == pred_m


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------

class TestScore:
    def test_perfect_score(self):
        y_test = [0, 1, 1, 0]
        y_pred = [0, 1, 1, 0]
        assert score(y_test, y_pred) == pytest.approx(1.0)

    def test_zero_score(self):
        y_test = [0, 1, 0, 1]
        y_pred = [1, 0, 1, 0]
        assert score(y_test, y_pred) == pytest.approx(0.0)

    def test_partial_score(self):
        y_test = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 1]  # 3 out of 4 correct
        assert score(y_test, y_pred) == pytest.approx(0.75)

    def test_single_sample(self):
        assert score([1], [1]) == pytest.approx(1.0)
        assert score([1], [0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Integration: fit -> predict -> score pipeline
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_end_to_end_majority_pipeline(self, simple_dataset):
        X, y = simple_dataset
        X_train, y_train = fit(X, y)
        X_test = np.array([[0.5], [10.5]])
        y_true = [0, 1]
        y_pred = predict_majority(X_train, y_train, X_test, k=2)
        acc = score(y_true, y_pred)
        assert acc == pytest.approx(1.0)

    def test_end_to_end_weighted_pipeline(self, simple_dataset):
        X, y = simple_dataset
        X_train, y_train = fit(X, y)
        X_test = np.array([[0.5], [10.5]])
        y_true = [0, 1]
        y_pred = predict_weighted(X_train, y_train, X_test, k=3)
        acc = score(y_true, y_pred)
        assert acc == pytest.approx(1.0)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
