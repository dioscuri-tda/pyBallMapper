"""fast_ball_mapper.py — Standalone, drop-in replacement for
``pyballmapper.BallMapper`` using sklearn ``BallTree`` + joblib parallel
processing.

Self-contained: depends only on numpy / pandas / networkx / scikit-learn /
joblib.  No internal project imports — safe to drop into any environment.

Public API mirrors ``pyballmapper.BallMapper``:
    bm.eps
    bm.eps_dict                       (always None; API compat)
    bm.points_covered_by_landmarks    (dict: node_id -> list[int])
    bm.Graph                          (networkx.Graph; nodes have
                                       'landmark', 'points covered', 'size')

Algorithmic improvements over the reference greedy O(N² · D) implementation:

    Phase 1  Build BallTree                O(N log N)
    Phase 2  Greedy landmark selection     O(N + L · (log N + k))   [serial]
    Phase 3  Coverage computation          O(L · (log N + k))       [parallel]
    Phase 4  Edge finding (sparse SpGEMM)  O(L · k̄²)                [C++, no GIL]

where N = samples, D = features, L = landmarks, k = avg ball size, E = edges.

Mathematical equivalence to the reference implementation:

  - Phase 2 uses a boolean *covered* mask: a point is skipped iff some
    already-selected landmark covers it.  This matches the greedy criterion
    of the reference algorithm exactly (same landmarks, same order).
  - Phase 3 queries *all* points within ``eps`` of every landmark (not just
    the still-uncovered ones), so coverage lists are complete.
  - Phase 4 builds a sparse incidence matrix M ∈ {0,1}^(L × N) where
    M[v, p] = 1 iff point p lies in landmark v's ball.  The sparse product
    S = M @ M.T is an L × L matrix whose (v, u) entry equals |cov(v) ∩ cov(u)|.
    Nonzero off-diagonal entries of S are exactly the BallMapper graph
    edges (two balls share at least one data point).  The triangle
    inequality `dist(c1, c2) > 2·eps ⇒ cov(c1) ∩ cov(c2) = ∅` is automatic:
    disjoint coverage produces zero in the product, so no candidate
    prefilter is needed.  This replaces the per-pair Python
    ``set.isdisjoint`` loop with a single C++-level SpGEMM
    (``scipy._sparsetools``) that releases the GIL during computation.
"""

from __future__ import annotations

import copy
import os
import time

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from sklearn.neighbors import BallTree

# ── CPU resource detection ────────────────────────────────────────────────────
try:
    import psutil

    N_PHYSICAL_CORES = psutil.cpu_count(logical=False) or 1
except ImportError:
    N_PHYSICAL_CORES = max(1, (os.cpu_count() or 2) // 2)

N_LOGICAL_THREADS = os.cpu_count() or 1


def _resolve_n_jobs(n_jobs):
    """Return actual worker count.  ``None`` or ``-1`` -> all physical cores."""
    if n_jobs is None or n_jobs == -1:
        return N_PHYSICAL_CORES
    return max(1, int(n_jobs))


class FastBallMapper:
    """Fast BallMapper using ``BallTree`` + joblib parallel processing.

    Euclidean metric only.  Output is mathematically identical to
    ``pyballmapper.BallMapper`` for the same ``X``, ``eps`` and natural
    landmark order ``0, 1, ..., N-1``.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    eps : float
        Ball radius.
    coloring_df : pandas.DataFrame, optional
    order : array-like of int, optional
        Order in which points are considered for landmark selection.
        Default: ``np.arange(N)``.
    verbose : bool, default False
    n_jobs : int or None, default None
        Number of parallel workers for Phases 3 and 4.
        ``None`` / ``-1`` -> all physical cores.

    Attributes
    ----------
    eps, eps_dict, points_covered_by_landmarks, Graph
        Same as ``pyballmapper.BallMapper``.
    n_landmarks : int
    build_time_s : dict
        Per-phase wall-clock timing in seconds.
    """

    def __init__(
        self, X, eps, coloring_df=None, order=None, verbose=False, n_jobs=None
    ):
        t_total = time.time()
        timing = {}

        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=float)

        self.eps = eps
        self.eps_dict = None  # pyballmapper API compatibility

        n_points, _n_dims = X.shape
        n_workers = _resolve_n_jobs(n_jobs)

        if order is None:
            order = np.arange(n_points)
        else:
            order = np.asarray(order)

        # ── Phase 1: Build BallTree ──────────────────────────────────────────
        t1 = time.time()
        tree = BallTree(X, metric="euclidean")
        timing["build_tree"] = time.time() - t1
        if verbose:
            print(f"  [Phase 1] BallTree built  {timing['build_tree']:.3f}s")

        # ── Phase 2: Greedy landmark selection ───────────────────────────────
        # Reference algorithm:  for each candidate point, iterate over all
        # already-selected landmarks and compute distance — O(N · L · D).
        # Here:  mark all points inside the new ball "covered" in one batch
        # query, and skip any candidate that is already covered — O(1) check.
        t2 = time.time()
        covered = np.zeros(n_points, dtype=bool)
        landmark_point_ids: list[int] = []

        for idx_p in order:
            if covered[idx_p]:
                continue
            landmark_point_ids.append(int(idx_p))
            in_ball = tree.query_radius(X[idx_p : idx_p + 1], r=eps)[0]
            covered[in_ball] = True

        n_landmarks = len(landmark_point_ids)
        timing["landmark_selection"] = time.time() - t2
        if verbose:
            print(
                f"  [Phase 2] {n_landmarks} landmarks selected  "
                f"{timing['landmark_selection']:.3f}s"
            )

        # ── Phase 3: Coverage computation (batched, parallel) ────────────────
        # BallTree.query_radius accepts a batch of query points and releases
        # the GIL — threading backend gives true parallelism here.
        t3 = time.time()
        landmark_pts = X[landmark_point_ids]  # (L, D)

        if n_workers <= 1 or n_landmarks < n_workers:
            coverage_arrays = tree.query_radius(landmark_pts, r=eps)
        else:
            chunk_indices = np.array_split(np.arange(n_landmarks), n_workers)
            chunk_results = Parallel(n_jobs=n_workers, backend="threading")(
                delayed(tree.query_radius)(landmark_pts[idx], r=eps)
                for idx in chunk_indices
                if len(idx) > 0
            )
            coverage_arrays = np.concatenate(chunk_results)

        self.points_covered_by_landmarks = {
            v: coverage_arrays[v].tolist() for v in range(n_landmarks)
        }
        timing["coverage"] = time.time() - t3
        if verbose:
            print(f"  [Phase 3] Coverage computed  {timing['coverage']:.3f}s")

        # ── Phase 4: Edge finding via sparse incidence-matrix product ────────
        # Build M ∈ {0,1}^(L × N) where M[v, p] = 1 iff point p is in
        # landmark v's ball.  S = M @ M.T is then a sparse L × L matrix whose
        # (v, u) entry equals |cov(v) ∩ cov(u)|; nonzero off-diagonal entries
        # are exactly the BallMapper edges.  Disjoint coverage gives zero in
        # the product, so the triangle-inequality prefilter is implicit.
        # This is a single C++-level SpGEMM (scipy._sparsetools), which
        # releases the GIL and replaces the per-pair Python set.isdisjoint
        # loop entirely.  Result is bit-for-bit identical (no false positives
        # / negatives).
        t4 = time.time()

        sizes = np.fromiter(
            (len(arr) for arr in coverage_arrays), dtype=np.int64, count=n_landmarks
        )
        indptr = np.empty(n_landmarks + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(sizes, out=indptr[1:])
        if n_landmarks > 0:
            indices = np.concatenate(coverage_arrays).astype(np.int64, copy=False)
        else:
            indices = np.empty(0, dtype=np.int64)
        data = np.ones(int(indptr[-1]), dtype=np.int32)

        M = csr_matrix((data, indices, indptr), shape=(n_landmarks, n_points))
        S = (M @ M.T).tocoo()
        mask = S.row < S.col
        all_edges = list(zip(S.row[mask].tolist(), S.col[mask].tolist()))

        timing["edge_finding"] = time.time() - t4
        if verbose:
            print(
                f"  [Phase 4] {len(all_edges)} edges found  "
                f"{timing['edge_finding']:.3f}s"
            )

        # ── Phase 5: Build NetworkX graph ────────────────────────────────────
        t5 = time.time()
        self.Graph = nx.Graph()
        self.Graph.add_nodes_from(range(n_landmarks))
        self.Graph.add_edges_from(all_edges)

        for v in range(n_landmarks):
            pts = np.array(self.points_covered_by_landmarks[v])
            self.Graph.nodes[v]["landmark"] = landmark_point_ids[v]
            self.Graph.nodes[v]["points covered"] = pts
            self.Graph.nodes[v]["size"] = len(pts)

        timing["build_graph"] = time.time() - t5

        self.n_landmarks = n_landmarks
        self.landmark_point_ids = landmark_point_ids  # convenience
        self.build_time_s = timing

        if isinstance(coloring_df, pd.DataFrame):
            self.add_coloring(coloring_df)

        if verbose:
            total = time.time() - t_total
            print(
                f"  [Total] FastBallMapper done  {total:.3f}s  "
                f"| tree={timing['build_tree']:.3f}  "
                f"landmark={timing['landmark_selection']:.3f}  "
                f"coverage={timing['coverage']:.3f}  "
                f"edges={timing['edge_finding']:.3f}"
            )

    # ── pyballmapper-compatible methods (subset) ─────────────────────────────

    def add_coloring(
        self, coloring_df, custom_function=np.mean, custom_name=None, add_std=False
    ):
        """Aggregate ``coloring_df`` values per node (matches BallMapper)."""
        for node in self.Graph.nodes:
            for col_name, avg in (
                coloring_df.loc[self.Graph.nodes[node]["points covered"]]
                .apply(custom_function, axis=0)
                .items()
            ):
                name = f"{col_name}_{custom_name}" if custom_name else col_name
                self.Graph.nodes[node][name] = avg
            if add_std:
                for col_name, std in (
                    coloring_df.loc[self.Graph.nodes[node]["points covered"]]
                    .std()
                    .items()
                ):
                    self.Graph.nodes[node][f"{col_name}_std"] = std

    def filter_by(self, list_of_points):
        """Return a deep copy keeping only nodes whose coverage intersects
        ``list_of_points`` (matches BallMapper)."""
        filtered_bm = copy.deepcopy(self)
        pt_set = set(list_of_points)
        for node in filtered_bm.Graph.nodes:
            kept = list(set(filtered_bm.points_covered_by_landmarks[node]) & pt_set)
            filtered_bm.points_covered_by_landmarks[node] = kept
            filtered_bm.Graph.nodes[node]["points covered"] = np.array(kept)
            filtered_bm.Graph.nodes[node]["size"] = len(kept)
        filtered_bm.Graph.remove_nodes_from(
            [n for n in filtered_bm.Graph if filtered_bm.Graph.nodes[n]["size"] == 0]
        )
        return filtered_bm

    def points_and_balls(self):
        """Long-form DataFrame: every (point, ball) membership pair."""
        rows = []
        for ball, points in self.points_covered_by_landmarks.items():
            for p in points:
                rows.append([p, ball])
        return pd.DataFrame(rows, columns=["point", "ball"])


if __name__ == "__main__":
    # Smoke test on synthetic data when invoked directly.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 5)).astype(np.float64)
    bm = FastBallMapper(X=X, eps=0.8, verbose=True)
    print(f"nodes={bm.n_landmarks}  edges={len(bm.Graph.edges)}")
