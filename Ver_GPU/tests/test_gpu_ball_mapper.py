"""Tests for the GPU-waveMIS Ball Mapper folder.

The GPU build tests require a CUDA device + torch and are skipped automatically
when none is available. The renderer and validator tests run on any machine
(CPU only) so CI still exercises the report pipeline and the validity contract.

Run with:  pytest -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _has_cuda():
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


HAS_CUDA = _has_cuda()
gpu_only = pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device / torch")


def _blobs(n=1500, d=10, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.random((8, d))
    assign = rng.integers(0, 8, size=n)
    X = centers[assign] + rng.normal(0, 0.05, size=(n, d))
    return np.ascontiguousarray(X, dtype=np.float32)


# ── CPU-side tests (always run) ──────────────────────────────────────────────
def test_report_renders_from_saved_results(tmp_path):
    """The bundled results/ render into a self-contained report.html."""
    import benchmark_gpu as b

    out = b.render(os.path.join(ROOT, "results"), str(tmp_path / "report.html"))
    assert os.path.exists(out)
    html = open(out, encoding="utf-8").read()
    assert "GPU-waveMIS" in html and "eps-sweep" in html.lower()
    # the embedded plots make this comfortably large
    assert len(html) > 100_000


def test_validator_accepts_balltree_net():
    """bm_validate must accept the canonical BallTree maximal eps-net (V1-V4).
    This pins the validity contract the GPU result is checked against."""
    import bm_validate as bv
    from fast_ball_mapper import FastBallMapper

    X = _blobs(n=1200, d=8).astype(np.float64)
    eps = 0.5
    bm = FastBallMapper(X=X, eps=eps, n_jobs=1)
    lm = [bm.Graph.nodes[v]["landmark"] for v in bm.Graph.nodes]
    cov = {v: bm.points_covered_by_landmarks[v] for v in bm.Graph.nodes}
    res = bv.validate(X, eps, lm, cov, list(bm.Graph.edges))
    assert res["valid"], res


# ── GPU tests (skipped without CUDA) ─────────────────────────────────────────
@gpu_only
def test_gpu_result_is_valid():
    import bm_validate as bv
    from gpu_ball_mapper import GpuBallMapper

    X = _blobs(n=2000, d=10)
    eps = 0.5
    bm = GpuBallMapper(X=X, eps=eps, device=0)
    lm = [bm.Graph.nodes[v]["landmark"] for v in bm.Graph.nodes]
    cov = {v: bm.points_covered_by_landmarks[v] for v in bm.Graph.nodes}
    res = bv.validate(
        X.astype(np.float64), eps, lm, cov, list(bm.Graph.edges), atol=1e-4
    )
    assert res["valid"], res


@gpu_only
def test_gpu_matches_balltree_landmark_count():
    from fast_ball_mapper import FastBallMapper
    from gpu_ball_mapper import GpuBallMapper

    X = _blobs(n=2000, d=10)
    eps = 0.5
    cpu = FastBallMapper(X=X, eps=eps, n_jobs=1)
    gpu = GpuBallMapper(X=X, eps=eps, device=0)
    # both are maximal eps-nets; counts should be very close (identical on the
    # bundled regime, within a few on fp32 boundary effects).
    assert abs(cpu.n_landmarks - gpu.n_landmarks) <= max(2, cpu.n_landmarks // 50)
