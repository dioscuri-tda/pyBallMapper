"""gen_synthetic.py — deterministic synthetic high-dimensional data for the
BallMapper scaling benchmark.

Data is generated *in process* with a fixed seed so that we never have to
ship large .npy files.  The goal is data that (a) is clustered like real
feature matrices, (b) is min-max normalised to [0, 1] per feature, and
(c) lets us pick an eps that yields a comparable landmark fraction across
very different N.

Public API
----------
make_highd(n, d=100, n_clusters=None, spread=0.06, seed=0)
    -> float32 ndarray (n, d), each column min-max normalised to [0, 1].

make_bootstrap(ref_X, n, jitter=0.004, seed=0)
    -> float32 ndarray (n, d), resampled from *ref_X* with small jitter.

calibrate_eps(X, quantiles=(0.02, 0.05, 0.12), sample=4000, seed=0)
    -> list[float] eps values at the given pairwise-distance quantiles.
"""

from __future__ import annotations

import numpy as np


def make_highd(
    n: int,
    d: int = 100,
    n_clusters: int | None = None,
    spread: float = 0.06,
    seed: int = 0,
) -> np.ndarray:
    """Generate *n* points in *d* dims as a Gaussian mixture, normalised to
    [0, 1] per feature.

    Parameters
    ----------
    n : int
        Number of points.
    d : int
        Number of features (default 100).
    n_clusters : int, optional
        Number of blobs; default ``max(8, n // 2000)``.
    spread : float
        Per-cluster std in the raw (pre-normalisation) space.
    seed : int
        RNG seed (deterministic).
    """
    rng = np.random.default_rng(seed)
    if n_clusters is None:
        n_clusters = max(8, n // 2000)

    centers = rng.random((n_clusters, d), dtype=np.float64)
    assign = rng.integers(0, n_clusters, size=n)
    X = centers[assign] + rng.normal(0.0, spread, size=(n, d))

    col_min = X.min(axis=0, keepdims=True)
    col_max = X.max(axis=0, keepdims=True)
    span = np.where(col_max - col_min > 1e-12, col_max - col_min, 1.0)
    X = (X - col_min) / span
    return np.ascontiguousarray(X, dtype=np.float32)


def make_bootstrap(
    ref_X: np.ndarray,
    n: int,
    jitter: float = 0.004,
    seed: int = 0,
) -> np.ndarray:
    """Resample *n* rows (with replacement) from *ref_X* and add small
    Gaussian jitter, then clip to [0, 1].

    This preserves the reference data's distance structure at arbitrary N.
    """
    rng = np.random.default_rng(seed)
    ref = np.ascontiguousarray(ref_X, dtype=np.float32)
    idx = rng.integers(0, ref.shape[0], size=n)
    X = ref[idx].astype(np.float32, copy=True)
    if jitter > 0.0:
        X += rng.normal(0.0, jitter, size=X.shape).astype(np.float32)
        np.clip(X, 0.0, 1.0, out=X)
    return np.ascontiguousarray(X, dtype=np.float32)


def default_reference(n: int = 20000, d: int = 100, seed: int = 0) -> np.ndarray:
    """Return a deterministic synthetic reference point cloud."""
    return make_highd(n, d=d, seed=seed)


def calibrate_eps(
    X: np.ndarray,
    quantiles: tuple[float, ...] = (0.02, 0.05, 0.12),
    sample: int = 4000,
    seed: int = 0,
) -> list[float]:
    """Return eps values at the given quantiles of the pairwise-distance
    distribution, estimated from a random sample of rows.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    m = min(sample, n)
    idx = rng.choice(n, size=m, replace=False)
    S = np.asarray(X[idx], dtype=np.float64)
    sq = (S * S).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (S @ S.T)
    iu = np.triu_indices(m, k=1)
    dvals = np.sqrt(np.maximum(d2[iu], 0.0))
    return [float(np.quantile(dvals, q)) for q in quantiles]
