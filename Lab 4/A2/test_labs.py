"""
Unit tests for the modular functions written across Lab 3 (manual
implementation) and Lab 4 (AI-assisted re-implementation of the same
functions).

Covers:
    Lab 3 -> lab3_functions.py   : label, onehot, minkowski, dot, norm,
                                    mean, variance, std, distance, kmeans
    Lab 4 -> lab4_functions.py   : label, onehot, minkowski, dot, norm
             lab4_stats.py       : mean, variance, std (flat/1D version)
             lab4_kmeans.py      : mean, distance, kmeans (axis=0 version)

Test cases were generated with Claude and reviewed/selected by the author
(Sanjeev) to match what each function is supposed to compute. Where
possible, results are checked against trusted reference implementations
(numpy / scipy) rather than hand-computed numbers, since that mirrors how
the labs themselves validate the "own" implementation (see A6.py, A7.py,
A9.py which compare against scipy/numpy directly).
"""
# Generated with Claude
import unittest

import numpy as np
import pandas as pd
from scipy.spatial.distance import minkowski as scipy_minkowski

import lab3_functions as l3
import lab4_functions as l4
import lab4_stats as l4s
import lab4_kmeans as l4k


class TestLabelEncoding(unittest.TestCase):
    # Generated with Claude

    def setUp(self):
        self.s = pd.Series(["b", "a", "c", "a"])

    def test_lab4_label_uses_unique_sorted_categories(self):
        # lab4's label() de-duplicates before assigning codes, so repeated
        # values always get the same code and codes run 0..n_unique-1.
        e, m = l4.label(self.s)
        self.assertEqual(m, {"a": 0, "b": 1, "c": 2})
        self.assertEqual(e.tolist(), [1, 0, 2, 0])

    def test_lab3_label_known_duplicate_behavior(self):
        # lab3's label() sorts the raw (non-deduplicated) values and
        # increments the counter on every occurrence, so a value's code
        # depends on how many times it appears before the last occurrence
        # is processed. This test documents that actual behavior rather
        # than assuming it matches lab4's corrected version.
        e, m = l3.label(self.s)
        self.assertEqual(m, {"a": 1, "b": 2, "c": 3})
        self.assertEqual(e.tolist(), [2, 1, 3, 1])

    def test_lab3_label_no_duplicates_matches_lab4(self):
        # With no repeated values the duplicate-counting bug can't fire,
        # so both implementations should agree.
        s_unique = pd.Series(["c", "a", "b"])
        e3, m3 = l3.label(s_unique)
        e4, m4 = l4.label(s_unique)
        self.assertEqual(m3, m4)
        self.assertEqual(e3.tolist(), e4.tolist())

    def test_single_category(self):
        s_one = pd.Series(["only"])
        e4, m4 = l4.label(s_one)
        self.assertEqual(m4, {"only": 0})
        self.assertEqual(e4.tolist(), [0])


class TestOneHotEncoding(unittest.TestCase):
    # Generated with Claude

    def setUp(self):
        self.d = pd.DataFrame({"cat": ["x", "y", "x", "z"], "val": [1, 2, 3, 4]})

    def test_lab3_onehot_drops_original_column(self):
        o = l3.onehot(self.d, "cat")
        self.assertNotIn("cat", o.columns)
        self.assertIn("x", o.columns)
        self.assertIn("y", o.columns)
        self.assertIn("z", o.columns)
        self.assertEqual(o["x"].tolist(), [1, 0, 1, 0])
        self.assertEqual(o["y"].tolist(), [0, 1, 0, 0])
        self.assertEqual(o["z"].tolist(), [0, 0, 0, 1])
        # original column untouched
        self.assertEqual(o["val"].tolist(), [1, 2, 3, 4])

    def test_lab4_onehot_keeps_original_column(self):
        o = l4.onehot(self.d, "cat")
        self.assertIn("cat", o.columns)
        self.assertEqual(o["x"].tolist(), [1, 0, 1, 0])
        self.assertEqual(o["y"].tolist(), [0, 1, 0, 0])
        self.assertEqual(o["z"].tolist(), [0, 0, 0, 1])

    def test_onehot_does_not_mutate_input(self):
        original_cols = list(self.d.columns)
        l3.onehot(self.d, "cat")
        l4.onehot(self.d, "cat")
        self.assertEqual(list(self.d.columns), original_cols)

    def test_onehot_rows_sum_to_one_per_category(self):
        o = l4.onehot(self.d, "cat")
        row_sums = o[["x", "y", "z"]].sum(axis=1)
        self.assertTrue((row_sums == 1).all())


class TestMinkowski(unittest.TestCase):
    # Generated with Claude

    def setUp(self):
        self.x1 = np.array([7, 3, 4])
        self.x2 = np.array([17, 6, 9])

    def test_lab3_manhattan_matches_scipy(self):
        own = l3.minkowski(self.x1, self.x2, 1)
        ref = scipy_minkowski(self.x1, self.x2, 1)
        self.assertAlmostEqual(own, ref, places=8)

    def test_lab3_euclidean_matches_scipy(self):
        own = l3.minkowski(self.x1, self.x2, 2)
        ref = scipy_minkowski(self.x1, self.x2, 2)
        self.assertAlmostEqual(own, ref, places=8)

    def test_lab4_matches_scipy_for_range_of_p(self):
        for p in range(1, 11):
            own = l4.minkowski(self.x1, self.x2, p)
            ref = scipy_minkowski(self.x1, self.x2, p)
            self.assertAlmostEqual(own, ref, places=6)

    def test_lab3_and_lab4_agree(self):
        for p in range(1, 6):
            self.assertAlmostEqual(
                l3.minkowski(self.x1, self.x2, p),
                l4.minkowski(self.x1, self.x2, p),
                places=8,
            )

    def test_zero_distance_for_identical_vectors(self):
        same = np.array([2, 2, 2])
        self.assertAlmostEqual(l4.minkowski(same, same, 2), 0.0, places=8)

    def test_lab4_accepts_plain_python_lists(self):
        # lab4's version explicitly converts inputs with np.array(), so it
        # should also work on plain lists (unlike lab3's, which relies on
        # numpy array subtraction already being available).
        own = l4.minkowski([7, 3, 4], [17, 6, 9], 2)
        ref = scipy_minkowski([7, 3, 4], [17, 6, 9], 2)
        self.assertAlmostEqual(own, ref, places=8)


class TestDotAndNorm(unittest.TestCase):
    # Generated with Claude

    def setUp(self):
        self.x1 = [7, 3, 4]
        self.x2 = [17, 6, 9]

    def test_lab3_dot_matches_numpy(self):
        self.assertEqual(l3.dot(self.x1, self.x2), np.dot(self.x1, self.x2))

    def test_lab4_dot_matches_numpy(self):
        self.assertEqual(l4.dot(self.x1, self.x2), np.dot(self.x1, self.x2))

    def test_lab3_norm_matches_numpy(self):
        self.assertAlmostEqual(l3.norm(self.x1), np.linalg.norm(self.x1), places=8)

    def test_lab4_norm_matches_numpy(self):
        self.assertAlmostEqual(l4.norm(self.x1), np.linalg.norm(self.x1), places=8)

    def test_dot_of_orthogonal_vectors_is_zero(self):
        a, b = [1, 0], [0, 1]
        self.assertEqual(l4.dot(a, b), 0)

    def test_norm_of_zero_vector_is_zero(self):
        self.assertEqual(l4.norm([0, 0, 0]), 0.0)

    def test_norm_equals_sqrt_of_self_dot(self):
        # norm(a) should equal sqrt(dot(a, a)) for both versions
        self.assertAlmostEqual(l4.norm(self.x1), np.sqrt(l4.dot(self.x1, self.x1)), places=8)
        self.assertAlmostEqual(l3.norm(self.x1), np.sqrt(l3.dot(self.x1, self.x1)), places=8)


class TestMeanVarianceStd(unittest.TestCase):
    # Generated with Claude

    def setUp(self):
        self.x_1d = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])

    def test_lab3_mean_matches_numpy_1d(self):
        self.assertAlmostEqual(l3.mean(self.x_1d), np.mean(self.x_1d), places=8)

    def test_lab3_variance_matches_numpy_population_variance(self):
        # lab3's variance divides by n (population variance, ddof=0),
        # matching numpy's default.
        self.assertAlmostEqual(l3.variance(self.x_1d), np.var(self.x_1d), places=8)

    def test_lab3_std_matches_numpy(self):
        self.assertAlmostEqual(l3.std(self.x_1d), np.std(self.x_1d), places=8)

    def test_lab4_stats_mean_matches_numpy_1d(self):
        self.assertAlmostEqual(l4s.mean(self.x_1d), np.mean(self.x_1d), places=8)

    def test_lab4_stats_variance_and_std_match_numpy(self):
        self.assertAlmostEqual(l4s.variance(self.x_1d), np.var(self.x_1d), places=8)
        self.assertAlmostEqual(l4s.std(self.x_1d), np.std(self.x_1d), places=8)

    def test_lab4_kmeans_mean_matches_numpy_axis0_on_2d_input(self):
        x_2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        own = l4k.mean(x_2d)
        ref = np.mean(x_2d, axis=0)
        np.testing.assert_allclose(own, ref)

    def test_lab3_mean_matches_numpy_axis0_on_2d_input(self):
        x_2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        own = l3.mean(x_2d)
        ref = np.mean(x_2d, axis=0)
        np.testing.assert_allclose(own, ref)

    def test_variance_of_constant_array_is_zero(self):
        constant = np.array([5.0, 5.0, 5.0, 5.0])
        self.assertAlmostEqual(l4s.variance(constant), 0.0, places=8)
        self.assertAlmostEqual(l3.variance(constant), 0.0, places=8)

    def test_std_is_sqrt_of_variance(self):
        self.assertAlmostEqual(l3.std(self.x_1d), np.sqrt(l3.variance(self.x_1d)), places=8)
        self.assertAlmostEqual(l4s.std(self.x_1d), np.sqrt(l4s.variance(self.x_1d)), places=8)


class TestDistance(unittest.TestCase):
    # Generated with Claude

    def setUp(self):
        self.a = np.array([1.0, 2.0, 3.0])
        self.b = np.array([4.0, 6.0, 3.0])

    def test_lab3_distance_matches_numpy_norm_of_difference(self):
        own = l3.distance(self.a, self.b)
        ref = np.linalg.norm(self.a - self.b)
        self.assertAlmostEqual(own, ref, places=8)

    def test_lab4_distance_matches_numpy_norm_of_difference(self):
        own = l4k.distance(self.a, self.b)
        ref = np.linalg.norm(self.a - self.b)
        self.assertAlmostEqual(own, ref, places=8)

    def test_distance_to_self_is_zero(self):
        self.assertAlmostEqual(l4k.distance(self.a, self.a), 0.0, places=8)

    def test_distance_equals_minkowski_p2(self):
        own = l4k.distance(self.a, self.b)
        ref = l4.minkowski(self.a, self.b, 2)
        self.assertAlmostEqual(own, ref, places=8)


class TestKMeans(unittest.TestCase):
    # Generated with Claude

    def setUp(self):
        np.random.seed(0)
        # Two well-separated clusters so the clustering outcome is
        # deterministic-in-substance even though initial centroids are
        # picked randomly.
        self.x = np.array(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0],
             [10.0, 10.0], [10.0, 11.0], [11.0, 10.0]]
        )

    def test_lab3_kmeans_finds_two_clusters_of_three(self):
        centroids, labels = l3.kmeans(self.x, 2)
        counts = sorted(np.bincount(labels.astype(int)).tolist())
        self.assertEqual(counts, [3, 3])

    def test_lab3_kmeans_groups_nearby_points_together(self):
        _, labels = l3.kmeans(self.x, 2)
        # first three points (near origin) must share a label
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[1], labels[2])
        # last three points (near (10,10)) must share a label
        self.assertEqual(labels[3], labels[4])
        self.assertEqual(labels[4], labels[5])
        # the two groups must be different clusters
        self.assertNotEqual(labels[0], labels[3])

    def test_lab4_kmeans_groups_nearby_points_together(self):
        labels, centroids = l4k.kmeans(self.x, 2)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[1], labels[2])
        self.assertEqual(labels[3], labels[4])
        self.assertEqual(labels[4], labels[5])
        self.assertNotEqual(labels[0], labels[3])

    def test_lab4_kmeans_centroids_are_near_cluster_means(self):
        labels, centroids = l4k.kmeans(self.x, 2)
        cluster_of_point0 = labels[0]
        expected_centroid = self.x[:3].mean(axis=0)
        np.testing.assert_allclose(
            centroids[cluster_of_point0], expected_centroid, atol=1e-6
        )

    def test_kmeans_k_equals_n_gives_each_point_its_own_cluster(self):
        x_small = np.array([[0.0, 0.0], [5.0, 5.0], [9.0, 1.0]])
        _, labels = l3.kmeans(x_small, 3)
        # every point should end up in its own singleton cluster
        self.assertEqual(len(set(labels.tolist())), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
