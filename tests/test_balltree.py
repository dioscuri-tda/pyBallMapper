"""Tests for the fast BallTree-based landmark method (``method="balltree"``).

The headline guarantee is that the BallTree method produces a graph that is
*identical* to the reference greedy method: the same ordered landmark point-ids
and the same edge set. These tests verify that on small synthetic datasets, so
they run in well under a second and need no bundled data file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyballmapper import BallMapper
from pyballmapper.ballmapper import (
    _find_edges_from_coverage,
    _find_landmarks_balltree,
    _find_landmarks_greedy,
)


def _landmark_sequence(bm: BallMapper) -> list[int]:
    return [int(bm.Graph.nodes[n]["landmark"]) for n in bm.Graph.nodes]


def _edges_by_landmark(bm: BallMapper) -> set[frozenset[int]]:
    pid = {n: int(bm.Graph.nodes[n]["landmark"]) for n in bm.Graph.nodes}
    return {frozenset((pid[u], pid[v])) for u, v in bm.Graph.edges}


def _blobs(n: int = 400, d: int = 5, seed: int = 0) -> np.ndarray:
    """Clustered gaussian blobs, guaranteeing several balls and edges."""
    rng = np.random.default_rng(seed)
    centers = rng.random((6, d))
    assign = rng.integers(0, 6, size=n)
    return np.ascontiguousarray(
        centers[assign] + rng.normal(0, 0.05, size=(n, d)), dtype=np.float64
    )


class TestFindLandmarksBallTree:
    @pytest.mark.parametrize("eps", [0.3, 0.5, 0.8])
    def test_identical_landmarks_to_greedy(self, eps: float):
        X = _blobs()
        order = range(len(X))
        lg, cg, _ = _find_landmarks_greedy(X, eps=eps, metric="euclidean", order=order)
        lb, cb, _ = _find_landmarks_balltree(
            X, eps=eps, metric="euclidean", order=order
        )
        assert lb == lg, "landmark point-ids differ from the greedy method"
        assert {k: set(v) for k, v in cb.items()} == {
            k: set(v) for k, v in cg.items()
        }, "coverage sets differ from the greedy method"

    def test_full_coverage(self):
        X = _blobs()
        _, coverage, _ = _find_landmarks_balltree(X, eps=0.5, metric="euclidean")
        covered: set[int] = set()
        for pts in coverage.values():
            covered.update(pts)
        assert covered == set(range(len(X)))

    def test_coverage_within_eps(self):
        X = _blobs()
        eps = 0.5
        landmarks, coverage, _ = _find_landmarks_balltree(
            X, eps=eps, metric="euclidean"
        )
        for v, pts in coverage.items():
            center = X[landmarks[v]]
            dist = np.linalg.norm(X[pts] - center, axis=1)
            assert np.all(dist <= eps + 1e-9)

    def test_custom_order_matches_greedy(self):
        X = _blobs(n=200, seed=3)
        order = list(range(len(X) - 1, -1, -1))
        lg, _, _ = _find_landmarks_greedy(X, eps=0.5, metric="euclidean", order=order)
        lb, _, _ = _find_landmarks_balltree(X, eps=0.5, metric="euclidean", order=order)
        assert lb == lg

    def test_single_point(self, single_point: np.ndarray):
        landmarks, coverage, _ = _find_landmarks_balltree(single_point, eps=0.5)
        assert len(landmarks) == 1
        assert coverage[0] == [0]

    def test_fallback_on_non_euclidean(self, simple_2d: np.ndarray):
        def manhattan(x: np.ndarray, y: np.ndarray) -> float:
            return float(np.sum(np.abs(x - y)))

        with pytest.warns(UserWarning, match="balltree"):
            _, coverage, _ = _find_landmarks_balltree(
                simple_2d, eps=1.0, metric=manhattan, order=range(len(simple_2d))
            )
        covered: set[int] = set()
        for pts in coverage.values():
            covered.update(pts)
        assert covered == set(range(len(simple_2d)))


class TestBallTreeQueryCount:
    """One radius query per landmark, not two.

    The query that marks a new ball's points as covered already returns the
    whole ball, so re-querying every landmark afterwards to build the coverage
    is pure repetition -- and the repeated pass is the same size as the first,
    so it doubles the tree work.
    """

    @staticmethod
    def _count_queries(X: np.ndarray, eps: float) -> tuple[int, int]:
        """Returns (radius queries issued, landmarks found)."""
        import sklearn.neighbors

        calls = 0
        original = sklearn.neighbors.BallTree.query_radius

        def counting(self, target, r, **kwargs):
            nonlocal calls
            calls += len(np.atleast_2d(target))
            return original(self, target, r, **kwargs)

        sklearn.neighbors.BallTree.query_radius = counting
        try:
            landmarks, _, _ = _find_landmarks_balltree(X, eps=eps, metric="euclidean")
        finally:
            sklearn.neighbors.BallTree.query_radius = original
        return calls, len(landmarks)

    @pytest.mark.parametrize("eps", [0.3, 0.5, 0.8])
    def test_one_query_per_landmark(self, eps: float):
        queries, n_landmarks = self._count_queries(_blobs(), eps)
        assert queries == n_landmarks

    def test_still_matches_greedy_exactly(self):
        """The property the halving must not cost: the same graph as greedy."""
        X = _blobs()
        order = range(len(X))
        lg, cg, _ = _find_landmarks_greedy(X, eps=0.5, metric="euclidean", order=order)
        lb, cb, _ = _find_landmarks_balltree(
            X, eps=0.5, metric="euclidean", order=order
        )
        assert lb == lg
        assert cb == {k: sorted(v) for k, v in cg.items()}


class TestFindEdgesFromCoverage:
    def test_matches_reference_loop(self):
        X = _blobs()
        _, coverage, _ = _find_landmarks_balltree(X, eps=0.5, metric="euclidean")
        fast = {frozenset(e) for e in _find_edges_from_coverage(coverage, len(X))}
        # reference O(n_landmarks**2) set-intersection loop
        keys = list(coverage.keys())
        ref: set[frozenset[int]] = set()
        for i, v in enumerate(keys[:-1]):
            for u in keys[i + 1 :]:
                if set(coverage[v]) & set(coverage[u]):
                    ref.add(frozenset((v, u)))
        assert fast == ref

    def test_empty(self):
        assert _find_edges_from_coverage({}, 10) == []


class TestBallMapperBallTreeMethod:
    @pytest.mark.parametrize("eps", [0.3, 0.5, 0.8])
    def test_graph_identical_to_default(self, eps: float):
        X = _blobs()
        ref = BallMapper(X, eps=eps, verbose=False)  # default greedy
        fast = BallMapper(X, eps=eps, method="balltree", verbose=False)
        assert _landmark_sequence(fast) == _landmark_sequence(ref)
        assert _edges_by_landmark(fast) == _edges_by_landmark(ref)

    def test_node_attributes(self):
        X = _blobs()
        bm = BallMapper(X, eps=0.5, method="balltree", verbose=False)
        for node in bm.Graph.nodes:
            attrs = bm.Graph.nodes[node]
            assert "landmark" in attrs
            assert "points covered" in attrs
            assert attrs["size"] == len(attrs["points covered"])

    def test_with_coloring(self):
        X = _blobs(n=100)
        cdf = pd.DataFrame({"c": np.arange(len(X), dtype=float)})
        bm = BallMapper(X, eps=0.5, method="balltree", coloring_df=cdf, verbose=False)
        for node in bm.Graph.nodes:
            assert "c" in bm.Graph.nodes[node]

    def test_single_point_dataset(self, single_point: np.ndarray):
        bm = BallMapper(single_point, eps=0.5, method="balltree", verbose=False)
        assert len(bm.Graph.nodes) == 1
        assert bm.Graph.nodes[0]["size"] == 1
