from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pyballmapper.ballmapper import (
    BallMapper,
    _euclid_distance,
    _find_landmarks,
    _find_landmarks_adaptive,
    _find_landmarks_deterministic_nearest_uncovered,
    _find_landmarks_greedy,
)


class TestEuclidDistance:
    def test_simple_distance(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert _euclid_distance(a, b) == 5.0

    def test_zero_distance(self):
        a = np.array([1.5, 2.5])
        assert _euclid_distance(a, a) == 0.0


class TestFindLandmarksGreedy:
    def test_basic_coverage(self, simple_2d):
        order = range(len(simple_2d))
        landmarks, coverage, _ = _find_landmarks_greedy(
            simple_2d, eps=0.5, metric="euclidean", order=order
        )
        covered = set()
        for pts in coverage.values():
            covered.update(pts)
        assert len(covered) == len(simple_2d)

    def test_precomputed_metric(self, distance_matrix):
        order = range(distance_matrix.shape[0])
        landmarks, coverage, _ = _find_landmarks_greedy(
            distance_matrix, eps=0.5, metric="precomputed", order=order
        )
        covered = set()
        for pts in coverage.values():
            covered.update(pts)
        assert len(covered) == distance_matrix.shape[0]

    def test_custom_order(self, simple_2d):
        order = list(range(len(simple_2d) - 1, -1, -1))
        landmarks, coverage, _ = _find_landmarks_greedy(
            simple_2d, eps=0.3, metric="euclidean", order=order
        )
        covered = set()
        for pts in coverage.values():
            covered.update(pts)
        assert len(covered) == len(simple_2d)

    def test_custom_metric(self, simple_2d):
        def manhattan(x, y):
            return float(np.sum(np.abs(x - y)))

        order = range(len(simple_2d))
        landmarks, coverage, _ = _find_landmarks_greedy(
            simple_2d, eps=1.0, metric=manhattan, order=order
        )
        covered = set()
        for pts in coverage.values():
            covered.update(pts)
        assert len(covered) == len(simple_2d)

    def test_with_orbits(self, simple_2d):
        n = len(simple_2d)
        orbits = np.array([[(i + 1) % n] for i in range(n)], dtype=object)
        order = range(n)
        landmarks, coverage, _ = _find_landmarks_greedy(
            simple_2d, eps=0.5, metric="euclidean", orbits=orbits, order=order
        )
        covered = set()
        for pts in coverage.values():
            covered.update(pts)
        assert len(covered) == len(simple_2d)

    def test_single_point(self, single_point):
        landmarks, coverage, _ = _find_landmarks_greedy(
            single_point, eps=0.5, metric="euclidean", order=range(len(single_point))
        )
        assert len(landmarks) == 1


class TestFindLandmarksDeterministic:
    def test_basic_coverage(self, simple_2d):
        landmarks, coverage, _ = _find_landmarks_deterministic_nearest_uncovered(
            simple_2d, eps=0.5
        )
        covered = set()
        for pts in coverage.values():
            covered.update(pts)
        assert len(covered) == len(simple_2d)

    def test_first_landmark_is_medoid(self, simple_2d):
        from scipy.spatial.distance import cdist

        landmarks, coverage, _ = _find_landmarks_deterministic_nearest_uncovered(
            simple_2d, eps=0.5
        )
        expected_medoid = int(np.argmin(cdist(simple_2d, simple_2d).sum(axis=1)))
        assert landmarks[0] == expected_medoid

    def test_deterministic_output(self, simple_2d):
        l1, _, _ = _find_landmarks_deterministic_nearest_uncovered(simple_2d, eps=0.5)
        l2, _, _ = _find_landmarks_deterministic_nearest_uncovered(simple_2d, eps=0.5)
        assert l1 == l2

    def test_single_point(self, single_point):
        landmarks, coverage, _ = _find_landmarks_deterministic_nearest_uncovered(
            single_point, eps=0.5
        )
        assert len(landmarks) == 1


class TestFindLandmarksAdaptive:
    def test_basic_coverage_with_max_size(self, simple_2d):
        order = range(len(simple_2d))
        landmarks, coverage, eps_dict = _find_landmarks_adaptive(
            simple_2d, eps=1.0, max_size=5, metric="euclidean", order=order
        )
        covered = set()
        for pts in coverage.values():
            covered.update(pts)
        assert len(covered) == len(simple_2d)

    def test_ball_sizes_respected(self, simple_2d):
        max_size = 3
        order = range(len(simple_2d))
        _, coverage, _ = _find_landmarks_adaptive(
            simple_2d, eps=2.0, max_size=max_size, metric="euclidean", order=order
        )
        for pts in coverage.values():
            assert len(pts) <= max_size

    def test_eps_dict_returned(self, simple_2d):
        order = range(len(simple_2d))
        _, _, eps_dict = _find_landmarks_adaptive(
            simple_2d, eps=1.0, max_size=5, metric="euclidean", order=order
        )
        assert len(eps_dict) > 0
        for eps_val in eps_dict.values():
            assert eps_val > 0

    def test_single_point(self, single_point):
        landmarks, coverage, _ = _find_landmarks_adaptive(
            single_point,
            eps=0.5,
            max_size=5,
            metric="euclidean",
            order=range(len(single_point)),
        )
        assert len(landmarks) == 1


class TestFindLandmarks:
    def test_greedy_method(self, simple_2d):
        order = range(len(simple_2d))
        l1, c1, _ = _find_landmarks(
            simple_2d, eps=0.5, method="greedy", metric="euclidean", order=order
        )
        l2, c2, _ = _find_landmarks_greedy(
            simple_2d, eps=0.5, metric="euclidean", order=order
        )
        assert len(l1) == len(l2)

    def test_nearest_method(self, simple_2d):
        l1, c1, _ = _find_landmarks(simple_2d, eps=0.5, method="nearest")
        l2, c2, _ = _find_landmarks_deterministic_nearest_uncovered(simple_2d, eps=0.5)
        assert l1 == l2

    def test_adaptive_method(self, simple_2d):
        order = range(len(simple_2d))
        l1, c1, _ = _find_landmarks(
            simple_2d,
            eps=1.0,
            method="adaptive",
            max_size=5,
            eta=0.7,
            metric="euclidean",
            order=order,
        )
        l2, c2, _ = _find_landmarks_adaptive(
            simple_2d, eps=1.0, max_size=5, metric="euclidean", order=order
        )
        assert len(l1) == len(l2)

    def test_default_is_greedy(self, simple_2d):
        order = range(len(simple_2d))
        l1, c1, _ = _find_landmarks(
            simple_2d, eps=0.5, method=None, metric="euclidean", order=order
        )
        l2, c2, _ = _find_landmarks(
            simple_2d, eps=0.5, method="greedy", metric="euclidean", order=order
        )
        assert len(l1) == len(l2)


class TestBallMapperConstruction:
    def test_basic_construction(self, simple_2d):
        bm = BallMapper(simple_2d, eps=0.5, verbose=False)
        assert hasattr(bm, "Graph")
        assert hasattr(bm, "points_covered_by_landmarks")
        assert hasattr(bm, "landmarks_data")
        assert bm.eps == 0.5
        assert len(bm.Graph.nodes) > 0

    def test_precomputed_metric(self, distance_matrix):
        bm = BallMapper(distance_matrix, eps=0.5, metric="precomputed", verbose=False)
        assert len(bm.Graph.nodes) > 0

    def test_custom_column_names(self, simple_2d):
        bm = BallMapper(simple_2d, eps=0.5, column_names=["a", "b"], verbose=False)
        assert list(bm.landmarks_data.columns) == ["a", "b"]

    def test_default_column_names(self, simple_2d):
        bm = BallMapper(simple_2d, eps=0.5, verbose=False)
        expected = ["x{}".format(i) for i in range(simple_2d.shape[1])]
        assert list(bm.landmarks_data.columns) == expected

    def test_custom_order(self, simple_2d):
        order = list(range(len(simple_2d) - 1, -1, -1))
        bm = BallMapper(simple_2d, eps=0.5, order=order, verbose=False)
        assert len(bm.Graph.nodes) > 0

    def test_invalid_order_warning(self, simple_2d):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            BallMapper(simple_2d, eps=0.5, order=[0, 1], verbose=False)
            assert any("order is not compatible" in str(msg.message) for msg in w)

    def test_with_coloring_df(self, simple_2d, coloring_df):
        bm = BallMapper(simple_2d, eps=0.5, coloring_df=coloring_df, verbose=False)
        for node in bm.Graph.nodes:
            assert "color_a" in bm.Graph.nodes[node]
            assert "color_b" in bm.Graph.nodes[node]

    def test_node_attributes(self, simple_2d):
        bm = BallMapper(simple_2d, eps=0.5, verbose=False)
        for node in bm.Graph.nodes:
            assert "landmark" in bm.Graph.nodes[node]
            assert "points covered" in bm.Graph.nodes[node]
            assert "size" in bm.Graph.nodes[node]

    def test_single_point_dataset(self, single_point):
        bm = BallMapper(single_point, eps=0.5, verbose=False)
        assert len(bm.Graph.nodes) == 1
        assert bm.Graph.nodes[0]["size"] == 1


class TestBallMapperAddColoring:
    def test_add_coloring_default(self, bm_instance, coloring_df):
        bm_instance.add_coloring(coloring_df)
        for node in bm_instance.Graph.nodes:
            assert "color_a" in bm_instance.Graph.nodes[node]
            assert "color_b" in bm_instance.Graph.nodes[node]

    def test_add_coloring_with_std(self, bm_instance, coloring_df):
        bm_instance.add_coloring(coloring_df, add_std=True)
        for node in bm_instance.Graph.nodes:
            assert "color_a_std" in bm_instance.Graph.nodes[node]
            assert "color_b_std" in bm_instance.Graph.nodes[node]

    def test_add_coloring_custom_function(self, bm_instance, coloring_df):
        bm_instance.add_coloring(coloring_df, custom_function=np.sum, custom_name="sum")
        for node in bm_instance.Graph.nodes:
            assert "color_a_sum" in bm_instance.Graph.nodes[node]
            assert "color_b_sum" in bm_instance.Graph.nodes[node]


class TestBallMapperColorByVariable:
    def test_color_by_variable(self, bm_with_coloring):
        from matplotlib import colormaps as cm

        my_palette = cm.get_cmap("Reds")
        min_val, max_val = bm_with_coloring.color_by_variable("color_a", my_palette)
        for node in bm_with_coloring.Graph.nodes:
            assert "color" in bm_with_coloring.Graph.nodes[node]

    def test_color_by_none(self, bm_instance):
        min_val, max_val = bm_instance.color_by_variable(None, None)
        for node in bm_instance.Graph.nodes:
            assert bm_instance.Graph.nodes[node]["color"] is not None

    def test_missing_variable_warning(self, bm_instance):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bm_instance.color_by_variable("nonexistent", None)
            assert any("is not a valid coloring" in str(msg.message) for msg in w)


class TestBallMapperFilterBy:
    def test_filter_keeps_subset(self, bm_instance):
        all_points = set()
        for pts in bm_instance.points_covered_by_landmarks.values():
            all_points.update(pts)
        subset = list(all_points)[: max(len(all_points) // 2, 1)]
        filtered = bm_instance.filter_by(subset)
        for node in filtered.Graph.nodes:
            for p in filtered.points_covered_by_landmarks[node]:
                assert p in subset

    def test_filter_empty_removes_nodes(self, bm_instance):
        filtered = bm_instance.filter_by([-1])
        assert len(filtered.Graph.nodes) == 0

    def test_filter_returns_copy(self, bm_instance):
        filtered = bm_instance.filter_by([0])
        assert filtered is not bm_instance


class TestBallMapperPointsAndBalls:
    def test_dataframe_shape(self, bm_instance):
        pab = bm_instance.points_and_balls()
        assert list(pab.columns) == ["point", "ball"]

    def test_all_points_represented(self, bm_instance):
        pab = bm_instance.points_and_balls()
        all_points = set()
        for pts in bm_instance.points_covered_by_landmarks.values():
            all_points.update(pts)
        assert set(pab["point"]) == all_points


class TestBallMapperBallData:
    def test_ball_data_single(self, bm_instance, simple_2d):
        ball_data = bm_instance.ball_data(simple_2d, 0)
        assert 0 in ball_data
        assert isinstance(ball_data[0], pd.DataFrame)

    def test_ball_data_list(self, bm_instance, simple_2d):
        ball_data = bm_instance.ball_data(simple_2d, [0, 1])
        assert 0 in ball_data
        assert 1 in ball_data

    def test_ball_data_invalid_raises(self, bm_instance, simple_2d):
        n = len(bm_instance.Graph.nodes)
        with pytest.raises(Exception, match="Incorrect ball number"):
            bm_instance.ball_data(simple_2d, n + 1)


class TestBallMapperBallDataIndex:
    def test_ball_data_index_single(self, bm_instance):
        idx = bm_instance.ball_data_index(0)
        assert 0 in idx
        assert isinstance(idx[0], list)

    def test_ball_data_index_list(self, bm_instance):
        idx = bm_instance.ball_data_index([0, 1])
        assert 0 in idx
        assert 1 in idx

    def test_ball_data_index_invalid_raises(self, bm_instance):
        n = len(bm_instance.Graph.nodes)
        with pytest.raises(Exception, match="Incorrect ball number"):
            bm_instance.ball_data_index(n + 1)
