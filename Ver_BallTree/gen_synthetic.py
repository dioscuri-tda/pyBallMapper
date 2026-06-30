"""gen_synthetic.py — deterministic synthetic high-dimensional data for the
GPU Ball Mapper scaling benchmark.

Data is generated *in process* on the GPU host (with a fixed seed) so that we
never have to transfer multi-hundred-MB .npy files over the network.  The goal
is data that (a) is clustered like real GC x GC-HRMS feature matrices (chemical
families form blobs), (b) is min-max normalised to [0, 1] per feature exactly
like the bundled ``wine_massspec_sample.npy`` pipeline, and (c) lets us pick an
eps that yields a *comparable* landmark fraction across very different N, so the
N-scaling comparison against BallTree is honest.

Public API
----------
make_highd(n, d=100, n_clusters=None, spread=0.06, seed=0, sparse_frac=0.0)
    -> float32 ndarray (n, d), each column min-max normalised to [0, 1].

calibrate_eps(X, quantiles=(0.02, 0.05, 0.12), sample=4000, seed=0)
    -> list[float] eps values taken from quantiles of a sampled pairwise
       distance distribution, so "tightness" is comparable across datasets.
"""

from __future__ import annotations

import numpy as np


def make_highd(n, d=100, n_clusters=None, spread=0.06, seed=0, sparse_frac=0.0):
    """Generate ``n`` points in ``d`` dims as a Gaussian mixture, normalised to
    [0, 1] per feature.

    Parameters
    ----------
    n : int            number of points
    d : int            number of features (default 100, like the wine sample)
    n_clusters : int   number of blobs; default max(8, n // 2000)
    spread : float     per-cluster std in the raw (pre-normalisation) space
    seed : int         RNG seed (deterministic)
    sparse_frac : float
        Fraction of features to zero out per point *before* normalisation, to
        mimic the sparse/peaky nature of mass-spec features.  0 disables it.
    """
    rng = np.random.default_rng(seed)
    if n_clusters is None:
        n_clusters = max(8, n // 2000)

    # Cluster centres spread over the unit cube.
    centers = rng.random((n_clusters, d), dtype=np.float64)

    # Assign points to clusters (roughly balanced) and jitter around centres.
    assign = rng.integers(0, n_clusters, size=n)
    X = centers[assign] + rng.normal(0.0, spread, size=(n, d))

    if sparse_frac > 0.0:
        mask = rng.random((n, d)) < sparse_frac
        X[mask] = 0.0

    # Min-max normalise each column to [0, 1] (matches _prepare_data.py).
    col_min = X.min(axis=0, keepdims=True)
    col_max = X.max(axis=0, keepdims=True)
    span = np.where(col_max - col_min > 1e-12, col_max - col_min, 1.0)
    X = (X - col_min) / span
    return np.ascontiguousarray(X, dtype=np.float32)


def make_bootstrap(ref_X, n, jitter=0.004, seed=0):
    """Resample ``n`` rows (with replacement) from a reference matrix and add a
    small per-feature Gaussian jitter, then re-clip to [0, 1].

    This is the PRIMARY scaling generator: it preserves the real wine sample's
    distance structure (clustering, sparsity, per-feature scale) at arbitrary
    N, so the meaningful eps grid (~[0.04, 0.24]) carries over directly.  The
    jitter makes resampled duplicates near-but-distinct (dense-sampling regime),
    which is exactly how more samples of a fixed chemical space behave.

    Parameters
    ----------
    ref_X : (M, D) array     reference data (e.g. the bundled wine sample)
    n : int                  number of points to produce
    jitter : float           std of additive Gaussian noise (in normalised units)
    seed : int               RNG seed
    """
    rng = np.random.default_rng(seed)
    ref = np.ascontiguousarray(ref_X, dtype=np.float32)
    idx = rng.integers(0, ref.shape[0], size=n)
    X = ref[idx].astype(np.float32, copy=True)
    if jitter > 0.0:
        X += rng.normal(0.0, jitter, size=X.shape).astype(np.float32)
        np.clip(X, 0.0, 1.0, out=X)
    return np.ascontiguousarray(X, dtype=np.float32)


def default_reference(here=None, n=20000, d=100, seed=0):
    """Return a reference point cloud for the benchmarks.

    Uses the bundled ``wine_massspec_sample.npy`` when it is present next to
    this module; otherwise falls back to deterministic synthetic high-D data
    (``make_highd``), so the benchmarks run with no binary data file at all
    (the repo ``.gitignore`` excludes ``*.npy``).  Returns a float32 array.
    """
    import os

    if here is None:
        here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "wine_massspec_sample.npy")
    if os.path.exists(path):
        return np.load(path).astype(np.float32)
    return make_highd(n, d=d, seed=seed)


def calibrate_eps(X, quantiles=(0.02, 0.05, 0.12), sample=4000, seed=0):
    """Return eps values at the given quantiles of the pairwise-distance
    distribution, estimated from a random sample of rows.

    Using distance quantiles keeps the *relative tightness* (hence the rough
    landmark fraction) comparable across datasets of very different N.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    m = min(sample, n)
    idx = rng.choice(n, size=m, replace=False)
    S = np.asarray(X[idx], dtype=np.float64)
    # Pairwise squared distances within the sample via the Gram trick.
    sq = (S * S).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (S @ S.T)
    iu = np.triu_indices(m, k=1)
    dvals = np.sqrt(np.maximum(d2[iu], 0.0))
    return [float(np.quantile(dvals, q)) for q in quantiles]


if __name__ == "__main__":
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    ref = np.load(os.path.join(here, "wine_massspec_sample.npy")).astype(np.float32)
    print("=== bootstrap-from-wine (PRIMARY: preserves real eps regime) ===")
    for n in (20000, 100000, 500000):
        X = make_bootstrap(ref, n, seed=1)
        eps_list = calibrate_eps(X)
        print(
            f"n={n:>7d} shape={X.shape} "
            f"range=[{X.min():.3f},{X.max():.3f}] "
            f"eps@quantiles={['%.3f' % e for e in eps_list]}"
        )
    print("=== independent blobs (secondary) ===")
    for n in (5000, 20000):
        X = make_highd(n, d=100, seed=1)
        eps_list = calibrate_eps(X)
        print(
            f"n={n:>7d} shape={X.shape} "
            f"eps@quantiles={['%.3f' % e for e in eps_list]}"
        )
