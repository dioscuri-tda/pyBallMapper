"""bm_validate.py — independent CPU-side validation that a Ball Mapper result
is *mathematically valid*, plus comparison metrics against a reference graph.

The user asked for "maximum speed with a VALID Mapper", i.e. the GPU landmark
set need not equal the CPU's lexicographically-first choice, but the output
must be a genuine Ball Mapper.  A valid Ball Mapper on (X, eps) is a MAXIMAL
eps-net with correct coverage and edges:

  V1 INDEPENDENCE   every pair of landmarks is > eps apart
                    (no landmark lies inside another landmark's ball).
  V2 MAXIMALITY     every point is within eps of at least one landmark
                    (the balls cover all N points).
  (V1 + V2 == the landmark set is a maximal independent set on the eps-graph.)
  V3 COVERAGE-EXACT for each landmark v, its reported coverage equals exactly
                    {p : dist(p, v) <= eps}.
  V4 EDGES-EXACT    landmarks u,v are adjacent iff coverage(u) ∩ coverage(v)
                    is non-empty.

Ground truth is computed with sklearn BallTree (exact euclidean), independent
of whatever GPU machinery produced the result.  All checks tolerate a tiny eps
boundary slack ``atol`` to absorb float32/TF32 rounding, since the user allows
minor boundary differences.

Usage (as a library)
--------------------
    res = validate(X, eps, landmark_ids, coverage, edges, atol=1e-5)
    assert res["valid"], res

``coverage`` may be a dict {node-> iterable[int]} or a list indexed by node.
``edges`` is an iterable of (u, v) node-index pairs (undirected).
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import BallTree


def _coverage_as_lists(coverage, n_landmarks):
    if isinstance(coverage, dict):
        return [np.asarray(coverage[v], dtype=np.int64) for v in range(n_landmarks)]
    return [np.asarray(c, dtype=np.int64) for c in coverage]


def validate(
    X,
    eps,
    landmark_ids,
    coverage,
    edges,
    atol=1e-5,
    check_edges=True,
    max_edge_pairs=None,
):
    """Validate a Ball Mapper result.  Returns a dict; ``res['valid']`` is the
    overall verdict.  ``atol`` is an additive slack on eps for boundary points.

    Parameters
    ----------
    X : (N, D) array
    eps : float
    landmark_ids : list[int]            point index of each landmark node
    coverage : dict|list                node -> covered point indices
    edges : iterable[(int, int)]        undirected node-index pairs
    check_edges : bool                  V4 can be the dominant cost; toggle off
    max_edge_pairs : int|None           cap on |coverage| product for the exact
                                        edge check (skip V4 if exceeded)
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    n_points = X.shape[0]
    landmark_ids = np.asarray(landmark_ids, dtype=np.int64)
    L = len(landmark_ids)
    cov = _coverage_as_lists(coverage, L)

    tree = BallTree(X, metric="euclidean")

    report = {"n_points": n_points, "n_landmarks": L, "eps": eps, "atol": atol}

    # ── Ground-truth coverage with a two-sided boundary tolerance ───────────
    # outer = within eps+atol  (anything beyond is *clearly outside*)
    # inner = within eps-atol  (anything inside is *clearly inside*)
    # The thin shell (eps-atol, eps+atol] is "don't care": float32/TF32 rounding
    # may legitimately push a near-boundary point either way and the user allows
    # minor boundary differences.
    outer_cov = tree.query_radius(X[landmark_ids], r=eps + atol)
    inner_cov = tree.query_radius(X[landmark_ids], r=max(eps - atol, 0.0))

    # V3 COVERAGE-EXACT: reported must contain every clearly-inside point and
    # no clearly-outside point.
    cov_mismatch = 0
    cov_missing_pts = 0  # clearly-inside points the result omitted
    cov_extra_pts = 0  # clearly-outside points the result wrongly included
    for v in range(L):
        rep = set(int(x) for x in cov[v].tolist())
        inner = set(int(x) for x in inner_cov[v])
        outer = set(int(x) for x in outer_cov[v])
        missing = inner - rep
        extra = rep - outer
        if missing or extra:
            cov_mismatch += 1
            cov_missing_pts += len(missing)
            cov_extra_pts += len(extra)
    report["V3_coverage_exact"] = cov_mismatch == 0
    report["coverage_mismatched_nodes"] = cov_mismatch
    report["coverage_missing_points"] = cov_missing_pts
    report["coverage_extra_points"] = cov_extra_pts

    # V2 MAXIMALITY: union of (outer) coverage covers all N points.
    covered = np.zeros(n_points, dtype=bool)
    for v in range(L):
        covered[outer_cov[v]] = True
    n_uncovered = int((~covered).sum())
    report["V2_maximal_cover"] = n_uncovered == 0
    report["uncovered_points"] = n_uncovered

    # V1 INDEPENDENCE: no landmark lies within eps of another landmark, i.e.
    # each landmark's ball contains no *other* landmark.  Use a strict radius
    # (eps - atol) so genuine > eps separations are not flagged by rounding.
    lm_set = set(int(x) for x in landmark_ids)
    rstrict = max(eps - atol, 0.0)
    close = tree.query_radius(X[landmark_ids], r=rstrict)
    indep_violations = 0
    for v in range(L):
        others = (lm_set & set(int(x) for x in close[v])) - {int(landmark_ids[v])}
        if others:
            indep_violations += 1
    report["V1_independent"] = indep_violations == 0
    report["independence_violations"] = indep_violations

    # V4 EDGES-EXACT: edge(u,v) iff coverage(u) ∩ coverage(v) != empty, using
    # the *reported* coverage (so this checks the edge step is consistent with
    # the coverage the algorithm produced).  Combined with V3 this proves the
    # edges are correct w.r.t. ground truth too.  Computed via sparse M @ M^T.
    if check_edges:
        from scipy.sparse import csr_matrix

        sizes = np.fromiter((len(c) for c in cov), dtype=np.int64, count=L)
        total = int(sizes.sum())
        if max_edge_pairs is not None and total > max_edge_pairs:
            report["V4_edges_exact"] = None
            report["edges_note"] = (
                f"skipped (coverage nnz {total} > cap {max_edge_pairs})"
            )
        else:
            indptr = np.empty(L + 1, dtype=np.int64)
            indptr[0] = 0
            np.cumsum(sizes, out=indptr[1:])
            indices = (np.concatenate(cov) if L else np.empty(0, np.int64)).astype(
                np.int64, copy=False
            )
            data = np.ones(total, dtype=np.int32)
            M = csr_matrix((data, indices, indptr), shape=(L, n_points))
            S = (M @ M.T).tocoo()
            m = S.row < S.col
            true_edges = set(zip(S.row[m].tolist(), S.col[m].tolist()))
            rep_edges = set(
                (min(int(u), int(v)), max(int(u), int(v))) for u, v in edges
            )
            report["V4_edges_exact"] = true_edges == rep_edges
            report["n_edges_true"] = len(true_edges)
            report["n_edges_reported"] = len(rep_edges)
            report["edges_only_true"] = len(true_edges - rep_edges)
            report["edges_only_reported"] = len(rep_edges - true_edges)
    else:
        report["V4_edges_exact"] = None

    checks = [
        report["V1_independent"],
        report["V2_maximal_cover"],
        report["V3_coverage_exact"],
    ]
    if report["V4_edges_exact"] is not None:
        checks.append(report["V4_edges_exact"])
    report["valid"] = all(checks)
    return report


def validate_large(
    X,
    eps,
    landmark_ids,
    coverage,
    edges,
    sample=256,
    chunk=20000,
    atol=1e-4,
    seed=0,
    check_edges=True,
    max_edge_pairs=200_000_000,
):
    """BallTree-free validation for very large N (e.g. up to 1M points).

    - V2 MAXIMALITY is checked from the *reported* coverage union (cheap, exact,
      no ground truth needed): every point index 0..N-1 must appear in some
      landmark's reported coverage.
    - V1 INDEPENDENCE and V3 COVERAGE-EXACT are checked on a RANDOM SAMPLE of
      ``sample`` landmarks, using exact euclidean distances computed in numpy
      chunks (no spatial index) — independent of the GPU machinery.
    - V4 EDGES-EXACT via sparse M @ M^T on the reported coverage (capped).
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    n_points = X.shape[0]
    landmark_ids = np.asarray(landmark_ids, dtype=np.int64)
    L = len(landmark_ids)
    cov = _coverage_as_lists(coverage, L)
    rng = np.random.default_rng(seed)
    report = {
        "n_points": n_points,
        "n_landmarks": L,
        "eps": eps,
        "atol": atol,
        "mode": "large/sampled",
        "sample": min(sample, L),
    }

    # V2 MAXIMALITY from reported coverage union.
    covered = np.zeros(n_points, dtype=bool)
    for c in cov:
        if len(c):
            covered[c] = True
    n_uncovered = int((~covered).sum())
    report["V2_maximal_cover"] = n_uncovered == 0
    report["uncovered_points"] = n_uncovered

    # V1 INDEPENDENCE over ALL landmarks (cheap: a BallTree on just the L
    # landmark coords, L << N).  This closes the blind spot of checking
    # independence only on a sample.
    lm_all = X[landmark_ids]
    lm_tree = BallTree(lm_all, metric="euclidean")
    near = lm_tree.query_radius(lm_all, r=max(eps - atol, 0.0))
    indep_all = sum(
        1 for v in range(L) if len(near[v]) > 1
    )  # >1 => a non-self neighbour
    report["V1_independent"] = indep_all == 0
    report["independence_violations"] = indep_all

    # V3 coverage-exact on a landmark sample, exact distances via chunked numpy.
    samp = rng.choice(L, size=min(sample, L), replace=False)
    lm_pts = X[landmark_ids[samp]]  # (S, D)
    e2_in = (max(eps - atol, 0.0)) ** 2
    e2_out = (eps + atol) ** 2
    cov_mismatch = 0
    cov_missing_pts = 0
    cov_extra_pts = 0
    sq_lm = (lm_pts * lm_pts).sum(1)
    # Stream points in chunks; accumulate clearly-inside ids per sampled landmark.
    inside_sets = [set() for _ in samp]
    for s in range(0, n_points, chunk):
        Xc = X[s : s + chunk]
        sqc = (Xc * Xc).sum(1)
        d2 = sq_lm[:, None] + sqc[None, :] - 2.0 * (lm_pts @ Xc.T)  # (S, chunk)
        np.maximum(d2, 0.0, out=d2)
        for j in range(len(samp)):
            ids_in = np.nonzero(d2[j] <= e2_in)[0] + s
            inside_sets[j].update(int(x) for x in ids_in)
    for j in range(len(samp)):
        v = int(samp[j])
        rep = set(int(x) for x in cov[v].tolist())
        # missing: clearly-inside not reported
        missing = inside_sets[j] - rep
        # extra: reported but clearly-outside — recompute distances for reported
        if rep:
            rep_arr = np.fromiter(rep, dtype=np.int64, count=len(rep))
            Xr = X[rep_arr]
            d2r = sq_lm[j] + (Xr * Xr).sum(1) - 2.0 * (Xr @ lm_pts[j])
            extra = int((d2r > e2_out).sum())
        else:
            extra = 0
        if missing or extra:
            cov_mismatch += 1
            cov_missing_pts += len(missing)
            cov_extra_pts += extra
    report["V3_coverage_exact"] = cov_mismatch == 0
    report["coverage_mismatched_nodes"] = cov_mismatch
    report["coverage_missing_points"] = cov_missing_pts
    report["coverage_extra_points"] = cov_extra_pts

    if check_edges:
        from scipy.sparse import csr_matrix

        sizes = np.fromiter((len(c) for c in cov), dtype=np.int64, count=L)
        total = int(sizes.sum())
        if total > max_edge_pairs:
            report["V4_edges_exact"] = None
            report["edges_note"] = f"skipped (nnz {total} > cap)"
        else:
            indptr = np.empty(L + 1, dtype=np.int64)
            indptr[0] = 0
            np.cumsum(sizes, out=indptr[1:])
            indices = (np.concatenate(cov) if L else np.empty(0, np.int64)).astype(
                np.int64, copy=False
            )
            M = csr_matrix(
                (np.ones(total, np.int32), indices, indptr), shape=(L, n_points)
            )
            S = (M @ M.T).tocoo()
            m = S.row < S.col
            true_edges = set(zip(S.row[m].tolist(), S.col[m].tolist()))
            rep_edges = set(
                (min(int(u), int(v)), max(int(u), int(v))) for u, v in edges
            )
            report["V4_edges_exact"] = true_edges == rep_edges
            report["n_edges_true"] = len(true_edges)
            report["n_edges_reported"] = len(rep_edges)
    else:
        report["V4_edges_exact"] = None

    checks = [
        report["V1_independent"],
        report["V2_maximal_cover"],
        report["V3_coverage_exact"],
    ]
    if report["V4_edges_exact"] is not None:
        checks.append(report["V4_edges_exact"])
    report["valid"] = all(checks)
    return report


def compare_to_reference(ref_landmark_ids, ref_edges, landmark_ids, edges):
    """Comparison metrics between a result and a reference graph (e.g. BallTree).
    The two need NOT match (different valid nets), but these quantify how close.
    """
    ref_lm = set(int(x) for x in ref_landmark_ids)
    res_lm = set(int(x) for x in landmark_ids)
    inter = len(ref_lm & res_lm)
    union = len(ref_lm | res_lm)
    return {
        "n_landmarks_ref": len(ref_lm),
        "n_landmarks_res": len(res_lm),
        "landmark_jaccard": inter / union if union else 1.0,
        "landmark_overlap_frac": inter / len(ref_lm) if ref_lm else 1.0,
        "n_edges_ref": len(set(map(lambda e: (min(e), max(e)), ref_edges))),
        "n_edges_res": len(
            set((min(int(u), int(v)), max(int(u), int(v))) for u, v in edges)
        ),
    }


if __name__ == "__main__":
    # Self-test: the BallTree FastBallMapper result must validate as a valid
    # Ball Mapper (it is the canonical greedy maximal eps-net).
    import os

    from fast_ball_mapper import FastBallMapper

    here = os.path.dirname(os.path.abspath(__file__))
    X = np.load(os.path.join(here, "wine_massspec_sample.npy")).astype(np.float64)
    for eps in (0.08, 0.16):
        bm = FastBallMapper(X=X, eps=eps, n_jobs=1)
        lm = [bm.Graph.nodes[n]["landmark"] for n in bm.Graph.nodes]
        cov = {n: bm.points_covered_by_landmarks[n] for n in bm.Graph.nodes}
        res = validate(X, eps, lm, cov, list(bm.Graph.edges))
        print(
            f"eps={eps}: valid={res['valid']}  L={res['n_landmarks']}  "
            f"V1={res['V1_independent']} V2={res['V2_maximal_cover']} "
            f"V3={res['V3_coverage_exact']} V4={res['V4_edges_exact']}"
        )
