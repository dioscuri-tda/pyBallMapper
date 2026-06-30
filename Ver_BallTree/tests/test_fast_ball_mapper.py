"""Correctness tests for ``fast_ball_mapper.FastBallMapper``.

The headline guarantee is that FastBallMapper produces a graph that is
*identical* to the reference ``pyballmapper.BallMapper`` (same ordered landmark
point-ids and the same edge set). These tests check that on small synthetic
datasets so they run in well under a second and need no bundled data file.

Run with:  pytest -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_ball_mapper import FastBallMapper  # noqa: E402

pyballmapper = pytest.importorskip("pyballmapper")
RefBallMapper = pyballmapper.BallMapper


def _landmarks(bm):
    return [int(bm.Graph.nodes[n]["landmark"]) for n in bm.Graph.nodes]


def _edges_by_pid(bm):
    pid = {n: int(bm.Graph.nodes[n]["landmark"]) for n in bm.Graph.nodes}
    return set(frozenset((pid[u], pid[v])) for u, v in bm.Graph.edges)


def _blobs(n=400, d=5, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.random((6, d))
    assign = rng.integers(0, 6, size=n)
    X = centers[assign] + rng.normal(0, 0.05, size=(n, d))
    return np.ascontiguousarray(X, dtype=np.float64)


@pytest.mark.parametrize("eps", [0.3, 0.5, 0.8])
def test_identical_to_reference(eps):
    X = _blobs()
    ref = RefBallMapper(X=X, eps=eps)
    fast = FastBallMapper(X=X, eps=eps, n_jobs=1)
    assert _landmarks(fast) == _landmarks(ref), "landmark point-ids differ"
    assert _edges_by_pid(fast) == _edges_by_pid(ref), "edge sets differ"


def test_parallel_matches_serial():
    X = _blobs(n=600, d=8, seed=3)
    serial = FastBallMapper(X=X, eps=0.5, n_jobs=1)
    parallel = FastBallMapper(X=X, eps=0.5, n_jobs=4)
    assert _landmarks(serial) == _landmarks(parallel)
    assert _edges_by_pid(serial) == _edges_by_pid(parallel)


def test_api_attributes():
    X = _blobs()
    bm = FastBallMapper(X=X, eps=0.5)
    assert bm.eps == 0.5
    assert bm.eps_dict is None
    assert isinstance(bm.points_covered_by_landmarks, dict)
    for node in bm.Graph.nodes:
        attrs = bm.Graph.nodes[node]
        assert "landmark" in attrs and "points covered" in attrs
        assert attrs["size"] == len(attrs["points covered"])


def test_coverage_is_correct():
    """Every reported covered point really lies within eps of its landmark."""
    X = _blobs()
    eps = 0.5
    bm = FastBallMapper(X=X, eps=eps, n_jobs=1)
    for node in bm.Graph.nodes:
        c = X[bm.Graph.nodes[node]["landmark"]]
        covered = bm.points_covered_by_landmarks[node]
        d = np.linalg.norm(X[covered] - c, axis=1)
        assert np.all(d <= eps + 1e-9)
