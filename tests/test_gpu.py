"""Tests for the CUDA landmark method (``method="gpu"``).

The GPU method does not reproduce the greedy landmark sequence, so -- unlike
:mod:`tests.test_balltree` -- these tests cannot assert graph equality against a
reference. What they assert instead is the property that actually defines a
BallMapper cover, and which the greedy sequence is only one way of reaching: the
landmarks are a *maximal eps-net*, meaning pairwise farther than ``eps`` apart
(independence) and jointly covering every data point (maximality).

Three tiers of tests live here:

* the fallback behaviour, which is exercised everywhere -- including on CI,
  where neither ``torch`` nor CUDA is present -- by faking the import;
* :func:`~pyballmapper.ballmapper._gpu_wavefront_mis`, which is pure tensor
  algebra and runs on CPU tensors when ``torch`` happens to be installed;
* the real device path, skipped unless a CUDA GPU is actually available.
"""

from __future__ import annotations

import builtins
import sys
import types
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pyballmapper import BallMapper
from pyballmapper.ballmapper import (
    _find_edges_from_coverage,
    _find_landmarks_balltree,
    _find_landmarks_gpu,
    _gpu_centred_float32,
    _gpu_exact_float32,
    _gpu_group_coverage,
    _gpu_has_repeat_within_ball,
)


def _blobs(n: int = 400, d: int = 5, seed: int = 0) -> np.ndarray:
    """Clustered gaussian blobs, guaranteeing several balls and edges."""
    rng = np.random.default_rng(seed)
    centers = rng.random((6, d))
    assign = rng.integers(0, 6, size=n)
    return np.ascontiguousarray(
        centers[assign] + rng.normal(0, 0.05, size=(n, d)), dtype=np.float64
    )


def _jittered_lattice(sites: int = 8, replicas: int = 20, seed: int = 0) -> np.ndarray:
    """An integer lattice, barely jittered, repeated.

    Distances here pile up on exact integers, so an ``eps`` sitting on one of
    them puts a large fraction of all pairs within a few ulps of the boundary --
    which is the regime where two float32 evaluations of the same distance can
    disagree. Continuous gaussian data essentially never produces such a tie.
    """
    axis = np.arange(sites, dtype=np.float64)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    points = np.repeat(grid.reshape(-1, 3), replicas, axis=0)
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(points + rng.normal(0.0, 1e-7, size=points.shape))


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _has_cuda() -> bool:
    if not _has_torch():
        return False
    import torch

    return bool(torch.cuda.is_available())


needs_torch = pytest.mark.skipif(not _has_torch(), reason="pytorch is not installed")
needs_cuda = pytest.mark.skipif(not _has_cuda(), reason="no CUDA device available")


def _assert_is_maximal_eps_net(
    X: np.ndarray, eps: float, landmarks: dict[int, int], coverage: dict[int, list[int]]
) -> None:
    """Asserts the two properties that make a landmark set a valid cover."""
    ids = np.asarray([landmarks[v] for v in sorted(landmarks)], dtype=np.int64)

    # independence: no two landmarks lie within eps of each other
    if len(ids) > 1:
        centers = X[ids]
        gaps = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        np.fill_diagonal(gaps, np.inf)
        assert gaps.min() > eps, "two landmarks are within eps of each other"

    # maximality: the balls jointly cover every point
    covered: set[int] = set()
    for points in coverage.values():
        covered.update(int(p) for p in points)
    assert covered == set(range(len(X))), "some points are covered by no ball"

    # the coverage lists are exactly the points inside each ball, and sorted.
    # The device decides in float32 against a float64 reference here, so points
    # sitting within rounding of the boundary are excused on both sides.
    scale = float(np.abs(X - X.mean(0)).max()) or 1.0
    slack = max(1e-9, 8.0 * scale * np.finfo(np.float32).eps)
    for idx_v, points in coverage.items():
        gaps = np.linalg.norm(X - X[landmarks[idx_v]], axis=1)
        assert list(points) == sorted(points), "coverage list is not sorted"
        listed = set(int(p) for p in points)
        assert listed >= set(np.flatnonzero(gaps <= eps - slack).tolist())
        assert listed <= set(np.flatnonzero(gaps <= eps + slack).tolist())


class TestFallbacks:
    """These run on every machine, CUDA or not -- CI included."""

    def test_fallback_on_non_euclidean(self, simple_2d: np.ndarray):
        def manhattan(x: np.ndarray, y: np.ndarray) -> float:
            return float(np.sum(np.abs(x - y)))

        with pytest.warns(UserWarning, match="gpu"):
            _, coverage, _ = _find_landmarks_gpu(
                simple_2d, eps=1.0, metric=manhattan, order=range(len(simple_2d))
            )
        covered: set[int] = set()
        for points in coverage.values():
            covered.update(points)
        assert covered == set(range(len(simple_2d)))

    def test_fallback_with_orbits(self, simple_2d: np.ndarray):
        orbits = [[i] for i in range(len(simple_2d))]
        with pytest.warns(UserWarning, match="gpu"):
            _, coverage, _ = _find_landmarks_gpu(simple_2d, eps=0.5, orbits=orbits)
        assert coverage

    def test_fallback_without_pytorch(self, monkeypatch: pytest.MonkeyPatch):
        """Without ``torch`` the method must degrade to ``balltree``, not raise."""
        real_import = builtins.__import__

        def no_torch(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "torch":
                raise ImportError("pytorch is not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "torch", raising=False)
        monkeypatch.setattr(builtins, "__import__", no_torch)

        X = _blobs(n=200)
        with pytest.warns(UserWarning, match="pytorch"):
            landmarks, coverage, _ = _find_landmarks_gpu(X, eps=0.5, metric="euclidean")
        expected_landmarks, expected_coverage, _ = _find_landmarks_balltree(
            X, eps=0.5, metric="euclidean"
        )
        assert landmarks == expected_landmarks
        assert coverage == expected_coverage

    def test_fallback_without_cuda(self, monkeypatch: pytest.MonkeyPatch):
        """A CPU-only pytorch build must degrade to ``balltree``, not raise."""
        stub = types.ModuleType("torch")
        stub.cuda = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", stub)

        X = _blobs(n=200)
        with pytest.warns(UserWarning, match="CUDA"):
            landmarks, coverage, _ = _find_landmarks_gpu(X, eps=0.5, metric="euclidean")
        expected_landmarks, expected_coverage, _ = _find_landmarks_balltree(
            X, eps=0.5, metric="euclidean"
        )
        assert landmarks == expected_landmarks
        assert coverage == expected_coverage

    def test_ballmapper_falls_back_end_to_end(self, monkeypatch: pytest.MonkeyPatch):
        stub = types.ModuleType("torch")
        stub.cuda = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", stub)

        X = _blobs(n=200)
        with pytest.warns(UserWarning, match="CUDA"):
            bm = BallMapper(X, eps=0.5, method="gpu", verbose=False)
        reference = BallMapper(X, eps=0.5, method="balltree", verbose=False)
        assert bm.Graph.number_of_nodes() == reference.Graph.number_of_nodes()
        assert set(bm.Graph.edges) == set(reference.Graph.edges)

    def test_unknown_method_message_lists_gpu(self, simple_2d: np.ndarray):
        with pytest.raises(ValueError, match="gpu"):
            BallMapper(simple_2d, eps=0.5, method="not-a-method", verbose=False)


@needs_torch
class TestWavefrontMis:
    """The independent-set core is plain tensor algebra; CPU tensors suffice."""

    @staticmethod
    def _lower_adjacency(X: np.ndarray, eps: float) -> Any:
        import torch

        gaps = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
        return torch.tril(
            torch.as_tensor((gaps <= eps).astype(np.float32)), diagonal=-1
        )

    @pytest.mark.parametrize("eps", [0.1, 0.3, 0.6])
    def test_selection_is_a_maximal_independent_set(self, eps: float):
        from pyballmapper.ballmapper import _gpu_wavefront_mis

        rng = np.random.default_rng(7)
        X = rng.random((60, 2))
        keep = _gpu_wavefront_mis(self._lower_adjacency(X, eps)).numpy()
        chosen = np.flatnonzero(keep)
        gaps = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)

        # independent
        sub = gaps[np.ix_(chosen, chosen)]
        np.fill_diagonal(sub, np.inf)
        assert sub.min() > eps
        # maximal: nothing outside the set could be added
        for i in range(len(X)):
            if not keep[i]:
                assert (gaps[i, chosen] <= eps).any()

    def test_empty_input(self):
        import torch

        from pyballmapper.ballmapper import _gpu_wavefront_mis

        keep = _gpu_wavefront_mis(torch.zeros((0, 0), dtype=torch.float32))
        assert keep.numel() == 0

    def test_isolated_vertices_all_selected(self):
        from pyballmapper.ballmapper import _gpu_wavefront_mis

        X = np.arange(10, dtype=np.float64).reshape(-1, 1) * 10.0
        keep = _gpu_wavefront_mis(self._lower_adjacency(X, 1.0)).numpy()
        assert keep.all()


@needs_cuda
class TestFindLandmarksGpu:
    """The real device path."""

    @pytest.mark.parametrize("eps", [0.3, 0.5, 0.8])
    def test_returns_a_maximal_eps_net(self, eps: float):
        X = _blobs()
        landmarks, coverage, _ = _find_landmarks_gpu(X, eps=eps, metric="euclidean")
        _assert_is_maximal_eps_net(X, eps, landmarks, coverage)

    def test_chunked_path_matches_single_chunk(self):
        """A tiny chunk forces the streaming path; it must stay a valid net."""
        X = _blobs(n=300)
        landmarks, coverage, _ = _find_landmarks_gpu(
            X, eps=0.5, metric="euclidean", chunk=37
        )
        _assert_is_maximal_eps_net(X, 0.5, landmarks, coverage)

    def test_landmark_block_tiling_is_transparent(self):
        X = _blobs(n=300)
        wide, wide_coverage, _ = _find_landmarks_gpu(
            X, eps=0.5, metric="euclidean", chunk=64
        )
        tiled, tiled_coverage, _ = _find_landmarks_gpu(
            X, eps=0.5, metric="euclidean", chunk=64, lblk=3
        )
        assert wide == tiled
        # the landmark tile offset exists only in the coverage pass, so the
        # landmark comparison above cannot see it
        _assert_is_maximal_eps_net(X, 0.5, tiled, tiled_coverage)
        assert wide_coverage == tiled_coverage

    def test_custom_order_is_honoured(self):
        X = _blobs(n=200, seed=3)
        order = list(range(len(X) - 1, -1, -1))
        landmarks, coverage, _ = _find_landmarks_gpu(
            X, eps=0.5, metric="euclidean", order=order, chunk=1_000_000
        )
        _assert_is_maximal_eps_net(X, 0.5, landmarks, coverage)
        # with one chunk holding everything, the first landmark is the first
        # point of `order`, exactly as the greedy rule would have it
        assert landmarks[0] == order[0]

    def test_single_point(self, single_point: np.ndarray):
        landmarks, coverage, _ = _find_landmarks_gpu(single_point, eps=0.5)
        assert landmarks == {0: 0}
        assert coverage[0] == [0]

    def test_widely_separated_points_are_all_landmarks(self):
        X = (np.arange(50, dtype=np.float64) * 10.0).reshape(-1, 1)
        landmarks, coverage, _ = _find_landmarks_gpu(X, eps=1.0, metric="euclidean")
        assert len(landmarks) == len(X)
        _assert_is_maximal_eps_net(X, 1.0, landmarks, coverage)

    def test_large_constant_offset_is_removed_exactly(self):
        """Centring happens in float64 before the cast, so a shift is free.

        Cast first and a shift of this size quantises the coordinates onto a
        float32 grid far coarser than ``eps``, which no later centring can
        undo -- the net would come out invalid rather than merely different.
        """
        X = _blobs(n=200) + 5.0e4
        landmarks, coverage, _ = _find_landmarks_gpu(X, eps=0.5, metric="euclidean")
        reference, _, _ = _find_landmarks_balltree(X, eps=0.5, metric="euclidean")
        assert landmarks == reference
        _assert_is_maximal_eps_net(X, 0.5, landmarks, coverage)

    @pytest.mark.parametrize("seed", range(6))
    def test_boundary_ties_leave_no_point_uncovered(self, seed: int):
        """Every point lands in some ball even when the passes disagree.

        Pass 1 decides "already covered" and pass 2 decides "in this ball" with
        matrix products of different shapes, so for a distance within a few ulps
        of ``eps`` the two can come out on opposite sides of the boundary. A
        point dropped by the first and rejected by the second would belong to no
        ball at all -- it would simply vanish from the cover, silently. With
        ``eps`` sitting exactly on the lattice spacing, a large fraction of the
        pairs here are such ties.
        """
        X = _jittered_lattice(seed=seed)
        landmarks, coverage, _ = _find_landmarks_gpu(X, eps=1.0, metric="euclidean")

        covered: set[int] = set()
        for points in coverage.values():
            covered.update(int(p) for p in points)
        missing = len(X) - len(covered)
        assert covered == set(range(len(X))), f"{missing} points lie in no ball"

        # and nothing was attached to a ball it does not belong in
        for idx_v, points in coverage.items():
            gaps = np.linalg.norm(X[list(points)] - X[landmarks[idx_v]], axis=1)
            assert gaps.max() <= 1.0 + 1e-5

    def test_falls_back_when_float32_cannot_resolve_eps(self):
        """A spread this large makes the Gram identity noise at this eps."""
        X = _blobs(n=200) * 1.0e4
        with pytest.warns(UserWarning, match="float32"):
            landmarks, coverage, _ = _find_landmarks_gpu(X, eps=0.5, metric="euclidean")
        reference, _, _ = _find_landmarks_balltree(X, eps=0.5, metric="euclidean")
        assert landmarks == reference

    @pytest.mark.parametrize("bad", [np.nan, np.inf, 1.0e39])
    def test_non_finite_input_is_rejected(self, bad: float):
        """1e39 is a finite float64 that becomes inf under the float32 cast."""
        X = _blobs(n=100)
        X[7, 2] = bad
        with pytest.raises(ValueError, match="finite"):
            _find_landmarks_gpu(X, eps=0.5, metric="euclidean")

    @pytest.mark.parametrize("bad_lblk", [0, -1])
    def test_invalid_lblk_is_rejected(self, bad_lblk: int):
        with pytest.raises(ValueError, match="lblk"):
            _find_landmarks_gpu(
                _blobs(n=50), eps=0.5, metric="euclidean", lblk=bad_lblk
            )

    def test_negative_eps_is_rejected(self):
        with pytest.raises(ValueError, match="eps"):
            _find_landmarks_gpu(_blobs(n=50), eps=-0.5, metric="euclidean")


@needs_cuda
class TestBallMapperGpuMethod:
    def test_graph_is_a_valid_cover(self):
        X = _blobs()
        bm = BallMapper(X, eps=0.5, method="gpu", verbose=False)
        landmarks = {n: int(bm.Graph.nodes[n]["landmark"]) for n in bm.Graph.nodes}
        _assert_is_maximal_eps_net(X, 0.5, landmarks, bm.points_covered_by_landmarks)

    def test_edges_match_the_coverage(self):
        X = _blobs()
        bm = BallMapper(X, eps=0.5, method="gpu", verbose=False)
        expected = {
            frozenset(e)
            for e in _find_edges_from_coverage(bm.points_covered_by_landmarks, len(X))
        }
        assert {frozenset(e) for e in bm.Graph.edges} == expected

    def test_node_attributes(self):
        X = _blobs()
        bm = BallMapper(X, eps=0.5, method="gpu", verbose=False)
        for node in bm.Graph.nodes:
            attrs = bm.Graph.nodes[node]
            assert "landmark" in attrs
            assert attrs["size"] == len(attrs["points covered"])

    def test_with_coloring(self):
        X = _blobs(n=100)
        cdf = pd.DataFrame({"c": np.arange(len(X), dtype=float)})
        bm = BallMapper(X, eps=0.5, method="gpu", coloring_df=cdf, verbose=False)
        for node in bm.Graph.nodes:
            assert "c" in bm.Graph.nodes[node]

    def test_single_point_dataset(self, single_point: np.ndarray):
        bm = BallMapper(single_point, eps=0.5, method="gpu", verbose=False)
        assert len(bm.Graph.nodes) == 1
        assert bm.Graph.nodes[0]["size"] == 1

    def test_device_keyword_is_forwarded(self):
        X = _blobs(n=100)
        bm = BallMapper(X, eps=0.5, method="gpu", device=0, verbose=False)
        assert bm.Graph.number_of_nodes() > 0

    def test_non_cuda_device_is_rejected(self):
        X = _blobs(n=50)
        with pytest.raises(ValueError, match="CUDA device"):
            BallMapper(X, eps=0.5, method="gpu", device="cpu", verbose=False)


@needs_torch
class TestGpuGroupCoverage:
    """The grouping runs on the device, so it needs torch -- CPU tensors do."""

    @staticmethod
    def _pairs(landmarks: list[int], points: list[int]) -> tuple[list[Any], list[Any]]:
        import torch

        return (
            [torch.as_tensor(landmarks, dtype=torch.int64)],
            [torch.as_tensor(points, dtype=torch.int64)],
        )

    def test_groups_and_maps_back_to_point_ids(self):
        # streaming position p holds data point positions[p]
        positions = np.array([30, 10, 20, 0], dtype=np.int64)
        # landmark 0 sits at streaming position 1, landmark 1 at position 3
        landmark_positions = [1, 3]
        incidence_landmarks, incidence_points = self._pairs([0, 0, 1], [1, 2, 3])

        coverage = _gpu_group_coverage(
            incidence_landmarks, incidence_points, positions, landmark_positions, 4
        )
        # streaming positions {1, 2} -> point ids {10, 20}
        assert coverage[0] == [10, 20]
        # position 3 -> point id 0, ascending within the ball
        assert coverage[1] == [0]

    def test_de_duplicates_if_a_pair_ever_arrives_twice(self):
        """Cannot happen by construction; the sweep is there in case it does."""
        positions = np.arange(4, dtype=np.int64)
        incidence_landmarks, incidence_points = self._pairs([0, 0, 0], [1, 2, 2])

        coverage = _gpu_group_coverage(
            incidence_landmarks, incidence_points, positions, [0], 4
        )
        assert coverage[0] == [0, 1, 2]

    def test_injects_self_coverage_the_device_may_have_missed(self):
        positions = np.arange(4, dtype=np.int64)
        landmark_positions = [0, 2]
        # the device reported nothing at all for landmark 1, not even its centre
        incidence_landmarks, incidence_points = self._pairs([0], [0])

        coverage = _gpu_group_coverage(
            incidence_landmarks, incidence_points, positions, landmark_positions, 4
        )
        assert coverage[0] == [0]
        assert coverage[1] == [2]

    def test_no_incidence_at_all(self):
        positions = np.arange(5, dtype=np.int64)
        coverage = _gpu_group_coverage([], [], positions, [4, 1], 5)
        assert coverage == {0: [4], 1: [1]}

    def test_no_landmarks(self):
        assert _gpu_group_coverage([], [], np.arange(3, dtype=np.int64), [], 3) == {}


class TestGpuRepeatSweep:
    """Pure numpy, so this runs everywhere -- CI included."""

    def test_clean_balls(self):
        grouped = np.array([1, 5, 9, 2, 7], dtype=np.int64)
        bounds = np.array([0, 3, 5], dtype=np.int64)
        assert not _gpu_has_repeat_within_ball(grouped, bounds)

    def test_repeat_inside_a_ball(self):
        grouped = np.array([1, 5, 5, 2, 7], dtype=np.int64)
        bounds = np.array([0, 3, 5], dtype=np.int64)
        assert _gpu_has_repeat_within_ball(grouped, bounds)

    def test_equal_across_a_boundary_is_not_a_repeat(self):
        """Two balls may perfectly well share a point at the seam."""
        grouped = np.array([1, 5, 5, 9], dtype=np.int64)
        bounds = np.array([0, 2, 4], dtype=np.int64)
        assert not _gpu_has_repeat_within_ball(grouped, bounds)

    @pytest.mark.parametrize("size", [0, 1])
    def test_degenerate_sizes(self, size: int):
        grouped = np.arange(size, dtype=np.int64)
        bounds = np.array([0, size], dtype=np.int64)
        assert not _gpu_has_repeat_within_ball(grouped, bounds)


class _FakeMatmul:
    def __init__(self, modern: bool) -> None:
        if modern:
            self.fp32_precision = "tf32"
        else:
            self.allow_tf32 = True


class _FakeTorch:
    """A torch stand-in exposing only what :func:`_gpu_exact_float32` touches."""

    def __init__(self, modern: bool) -> None:
        self.backends = types.SimpleNamespace(
            cuda=types.SimpleNamespace(matmul=_FakeMatmul(modern))
        )
        self.entered: list[Any] = []
        self.exited = 0
        outer = self

        class _Device:
            def __init__(self, device: Any) -> None:
                self.device = device

            def __enter__(self) -> None:
                outer.entered.append(self.device)

            def __exit__(self, *exc: Any) -> bool:
                outer.exited += 1
                return False

        self.cuda = types.SimpleNamespace(device=_Device)


class TestGpuExactFloat32:
    """Also pure host code -- the device is stubbed, so this runs in CI."""

    @pytest.mark.parametrize("modern", [True, False])
    def test_restores_the_matmul_mode(self, modern: bool):
        torch = _FakeTorch(modern)
        matmul = torch.backends.cuda.matmul
        attr = "fp32_precision" if modern else "allow_tf32"
        before = getattr(matmul, attr)

        with _gpu_exact_float32(torch, "cuda:0"):
            assert getattr(matmul, attr) == ("ieee" if modern else False)
        assert getattr(matmul, attr) == before
        assert torch.entered == ["cuda:0"] and torch.exited == 1

    @pytest.mark.parametrize("modern", [True, False])
    def test_restores_the_matmul_mode_on_failure(self, modern: bool):
        torch = _FakeTorch(modern)
        matmul = torch.backends.cuda.matmul
        attr = "fp32_precision" if modern else "allow_tf32"
        before = getattr(matmul, attr)

        with pytest.raises(RuntimeError, match="boom"):
            with _gpu_exact_float32(torch, "cuda:0"):
                raise RuntimeError("boom")
        assert getattr(matmul, attr) == before
        assert torch.exited == 1

    def test_touches_only_one_of_the_two_apis(self):
        """Mixing fp32_precision with allow_tf32 makes torch raise on any read."""
        torch = _FakeTorch(modern=True)
        with _gpu_exact_float32(torch, "cuda:0"):
            pass
        assert not hasattr(torch.backends.cuda.matmul, "allow_tf32")


def _centred_naive(X: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """The obvious version: one full float64 copy, centre, cast."""
    centred = np.asarray(X[positions], dtype=np.float64)
    centred -= centred.mean(0)
    return np.ascontiguousarray(centred, dtype=np.float32)


class TestGpuCentredFloat32:
    """Pure host code, so this runs everywhere -- CI included."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("offset", [0.0, 5.0e4, -1.0e6])
    def test_matches_the_naive_version_bit_for_bit(self, dtype, offset: float):
        X = (_blobs(n=500, d=7) + offset).astype(dtype)
        positions = np.arange(len(X), dtype=np.int64)
        assert np.array_equal(
            _gpu_centred_float32(X, positions), _centred_naive(X, positions)
        )

    def test_permuted_order_matches(self):
        X = _blobs(n=500, d=7)
        positions = np.random.default_rng(0).permutation(len(X)).astype(np.int64)
        assert np.array_equal(
            _gpu_centred_float32(X, positions), _centred_naive(X, positions)
        )

    def test_blocking_is_transparent(self, monkeypatch: pytest.MonkeyPatch):
        """Force several blocks, including a short final one."""
        import pyballmapper.ballmapper as module

        X = _blobs(n=333, d=7)
        positions = np.arange(len(X), dtype=np.int64)
        expected = _centred_naive(X, positions)
        for budget in (7 * 8, 50 * 7 * 8, 10**9):
            monkeypatch.setattr(module, "GPU_CENTRING_BLOCK_BYTES", budget)
            assert np.array_equal(_gpu_centred_float32(X, positions), expected)

    def test_centring_beats_casting_first_on_offset_data(self):
        """The reason this runs on the host in float64 at all."""
        X = _blobs(n=200, d=3) + 5.0e6
        positions = np.arange(len(X), dtype=np.int64)

        good = _gpu_centred_float32(X, positions)
        # what casting first would have produced
        cast_first = np.asarray(X[positions], dtype=np.float32)
        bad = cast_first - cast_first.mean(0)

        reference = np.asarray(X, dtype=np.float64)
        reference = reference - reference.mean(0)
        assert np.abs(good - reference).max() < np.abs(bad - reference).max()

    def test_empty_input(self):
        X = np.empty((0, 4), dtype=np.float64)
        out = _gpu_centred_float32(X, np.empty(0, dtype=np.int64))
        assert out.shape == (0, 4) and out.dtype == np.float32
