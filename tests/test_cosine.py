"""Tests for ``metric="cosine"``.

The greedy method computes the cosine dissimilarity ``1 - cos(x, y)`` by its
definition and is the reference. The balltree method reduces it to the
Euclidean case -- on unit vectors ``1 - cos(x, y) <= eps`` holds exactly when
``||x - y|| <= sqrt(2 * eps)`` -- so it must reproduce the greedy graph, with
coverage differing at most for points within float64 rounding of ``eps``. The
GPU counterparts are in ``test_gpu.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.spatial.distance import cosine as scipy_cosine

from pyballmapper import BallMapper
from pyballmapper.ballmapper import (
    _cosine_distance,
    _find_landmarks_balltree,
    _find_landmarks_gpu,
    _find_landmarks_greedy,
    _gpu_unit_rows_float32,
)


def _directions(n: int = 400, d: int = 5, seed: int = 0) -> np.ndarray:
    """Clusters of directions at random lengths, so that only angles matter."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(6, d))
    assign = rng.integers(0, 6, size=n)
    X = centers[assign] + rng.normal(0, 0.15, size=(n, d))
    return np.ascontiguousarray(X * rng.uniform(0.1, 10.0, size=(n, 1)))


def _balltree_quietly(X: np.ndarray, eps: float, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _find_landmarks_balltree(X, eps, metric="cosine", **kwargs)


def _assert_coverage_differs_only_at_the_boundary(X, eps, landmarks, cov, ref_cov):
    for node, landmark in landmarks.items():
        for p in set(cov[node]) ^ set(ref_cov[node]):
            gap = scipy_cosine(X[p], X[landmark]) - eps
            assert abs(gap) < 1e-12, f"point {p} is {gap:.3g} from the boundary"


class TestCosineDistance:
    def test_matches_scipy(self):
        X = _directions(n=50)
        for i in range(0, 50, 7):
            for j in range(50):
                assert _cosine_distance(X[i], X[j]) == pytest.approx(
                    scipy_cosine(X[i], X[j]), abs=1e-15
                )

    def test_integer_input(self):
        x = np.array([1, 0, 1])
        y = np.array([1, 1, 0])
        assert _cosine_distance(x, y) == pytest.approx(0.5)

    def test_clipped_to_the_valid_range(self):
        x = np.array([0.1, 0.2, 0.3])
        assert _cosine_distance(x, x) >= 0.0
        assert _cosine_distance(x, -x) == pytest.approx(2.0)


class TestGreedyCosine:
    @pytest.mark.parametrize("eps", [0.02, 0.05, 0.1])
    def test_string_matches_scipy_callable(self, eps: float):
        """The string used to raise TypeError; it now means the same thing."""
        X = _directions()
        string = _find_landmarks_greedy(X, eps, metric="cosine", order=range(len(X)))
        callable_ = _find_landmarks_greedy(
            X, eps, metric=scipy_cosine, order=range(len(X))
        )
        assert string[0] == callable_[0]
        _assert_coverage_differs_only_at_the_boundary(
            X, eps, string[0], string[1], callable_[1]
        )

    def test_ballmapper_accepts_the_string(self):
        X = _directions(n=200)
        bm = BallMapper(X, eps=0.05, metric="cosine", verbose=False)
        covered = set()
        for node in bm.Graph.nodes:
            covered.update(bm.Graph.nodes[node]["points covered"].tolist())
        assert covered == set(range(len(X)))

    def test_adaptive_accepts_the_string(self):
        X = _directions(n=200)
        bm = BallMapper(
            X, eps=0.1, metric="cosine", method="adaptive", max_size=30, verbose=False
        )
        assert bm.Graph.number_of_nodes() > 0


class TestBalltreeCosine:
    @pytest.mark.parametrize("n,d", [(400, 5), (3000, 20)])
    @pytest.mark.parametrize("eps", [0.02, 0.05, 0.1])
    def test_matches_greedy(self, eps: float, n: int, d: int):
        X = _directions(n=n, d=d)
        order = range(len(X))
        ref, ref_cov, _ = _find_landmarks_greedy(X, eps, metric="cosine", order=order)
        got, got_cov, _ = _balltree_quietly(X, eps, order=order)
        assert got == ref
        _assert_coverage_differs_only_at_the_boundary(X, eps, ref, got_cov, ref_cov)

    def test_ballmapper_graph_matches_greedy(self):
        X = _directions()
        greedy = BallMapper(X, eps=0.05, metric="cosine", verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fast = BallMapper(
                X, eps=0.05, metric="cosine", method="balltree", verbose=False
            )
        assert dict(greedy.Graph.nodes(data="landmark")) == dict(
            fast.Graph.nodes(data="landmark")
        )
        assert set(greedy.Graph.edges) == set(fast.Graph.edges)

    def test_custom_order_is_honoured(self):
        X = _directions()
        order = np.random.default_rng(1).permutation(len(X)).tolist()
        ref, _, _ = _find_landmarks_greedy(X, 0.05, metric="cosine", order=order)
        got, _, _ = _balltree_quietly(X, 0.05, order=order)
        assert got == ref

    def test_invariant_to_row_scaling(self):
        X = _directions()
        scaled = X * np.random.default_rng(2).uniform(1e-3, 1e3, size=(len(X), 1))
        assert _balltree_quietly(X, 0.05)[0] == _balltree_quietly(scaled, 0.05)[0]

    def test_warns_and_says_what_it_did(self):
        X = _directions(n=50)
        with pytest.warns(UserWarning, match=r"sqrt\(2\*eps\) = 0\.316228"):
            _find_landmarks_balltree(X, 0.05, metric="cosine")

    def test_cosine_callable_still_falls_back_to_greedy(self):
        """A callable is never given to the BallTree, whose pruning needs a metric."""
        X = _directions(n=100)
        with pytest.warns(UserWarning, match="falling back to the greedy method"):
            got, _, _ = _find_landmarks_balltree(
                X, 0.05, metric=scipy_cosine, order=range(len(X))
            )
        ref, _, _ = _find_landmarks_greedy(
            X, 0.05, metric=scipy_cosine, order=range(len(X))
        )
        assert got == ref

    def test_negative_eps_is_rejected(self):
        with pytest.raises(ValueError, match="eps"):
            _find_landmarks_balltree(_directions(n=20), -0.1, metric="cosine")


class TestZeroVectors:
    """Undefined for the cosine, so every method refuses them by name."""

    @staticmethod
    def _with_zero_rows() -> np.ndarray:
        X = _directions(n=50)
        X[[3, 7]] = 0.0
        return X

    @pytest.mark.parametrize("method", [None, "balltree", "adaptive", "gpu"])
    def test_rejected_with_the_row_ids(self, method: str | None):
        """Checked before torch is imported, so "gpu" fails the same everywhere."""
        with pytest.raises(ValueError, match=r"rows \[3, 7\] have zero norm"):
            BallMapper(
                self._with_zero_rows(),
                eps=0.1,
                metric="cosine",
                method=method,
                max_size=30,
                verbose=False,
            )

    def test_gpu_function_directly(self):
        with pytest.raises(ValueError, match="zero norm"):
            _find_landmarks_gpu(self._with_zero_rows(), 0.1, metric="cosine")

    def test_long_list_is_truncated(self):
        X = _directions(n=50)
        X[:12] = 0.0
        with pytest.raises(ValueError, match=r"9\] and 2 more"):
            _find_landmarks_balltree(X, 0.1, metric="cosine")


class TestGpuUnitRows:
    """Pure host code, so this runs everywhere -- CI included."""

    def test_rows_are_normalised_in_order(self):
        X = _directions(n=333, d=7) + 50.0
        positions = np.random.default_rng(0).permutation(len(X)).astype(np.int64)
        expected = X[positions] / np.linalg.norm(X[positions], axis=1, keepdims=True)
        out = _gpu_unit_rows_float32(X, positions)
        assert out.dtype == np.float32
        assert np.array_equal(out, expected.astype(np.float32))

    def test_blocking_is_transparent(self, monkeypatch: pytest.MonkeyPatch):
        import pyballmapper.ballmapper as module

        X = _directions(n=333, d=7)
        positions = np.arange(len(X), dtype=np.int64)
        expected = _gpu_unit_rows_float32(X, positions)
        for budget in (7 * 8, 50 * 7 * 8):
            monkeypatch.setattr(module, "GPU_CENTRING_BLOCK_BYTES", budget)
            assert np.array_equal(_gpu_unit_rows_float32(X, positions), expected)

    def test_empty_input(self):
        out = _gpu_unit_rows_float32(np.empty((0, 4)), np.empty(0, dtype=np.int64))
        assert out.shape == (0, 4) and out.dtype == np.float32
