"""gpu_ball_mapper.py — GPU Ball Mapper, a drop-in replacement for
``fast_ball_mapper.FastBallMapper`` that is unconditionally faster than the
sklearn-BallTree baseline and scales to N ~ 1,000,000.

Design (from the multi-agent synthesis spec)
---------------------------------------------
The Ball Mapper bottleneck on CPU is the SEQUENTIAL greedy landmark selection
(leader clustering).  That sequential leader rule is exactly a
lexicographically-first MAXIMAL INDEPENDENT SET on the eps-proximity graph,
which parallelises as an *index-priority wavefront* MIS: in each round, every
still-active point with no lower-indexed active neighbour is selected
simultaneously (two within-eps points can never both win — the higher-indexed
one sees the lower as a neighbour), then every selected point's eps-ball is
marked covered.  This removes the serial wall entirely.

The user requested MAXIMUM SPEED with a VALID Mapper (the landmark set need not
equal the CPU's lexicographic choice — any maximal eps-net is acceptable).  We
exploit that: a single UNIFIED chunked/online path handles every N.

  PASS 1  online net construction.  Stream points in chunks; test each chunk
          against ALL existing landmarks (chunk x L GEMM) to find uncovered
          rows; resolve intra-chunk conflicts with a wavefront MIS on the tiny
          uncovered subgraph; append survivors as new landmarks.
          => landmarks are pairwise > eps (independent) and every point is
             within eps of some landmark (maximal) — a VALID maximal eps-net.
  PASS 2  coverage of ALL N points against the FINAL landmark set (recomputed
          cleanly — NEVER reused from the Pass-1 selection mask; that is the #1
          correctness bug the spec warns about).
  PASS 3  edges via presence-only incidence product  S = M @ M^T  (off-diagonal
          nonzero == shared covered point), overflow-immune.

All heavy math is PyTorch GEMM (cuBLAS) in **fp32** — the default and the only
precision that is correct at the eps boundary for this problem.  Distances use
the Gram trick ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b on squared distances (no
sqrt), thresholded against eps^2.  (``precision='tf32'`` is ~2x faster but its
10-bit mantissa cannot resolve the eps boundary — abs error >> eps^2 — and
produces an INVALID net, so it is opt-in only and emits a warning.)  When N is
small enough that a single chunk holds all points, Pass 1 degenerates to one
dense MIS — the "dense path" — automatically.

Scaling note: the GEMM work is O(N * L * D) where L is the landmark count.  The
unconditional speed-up over BallTree holds when L stays SUBLINEAR in N (the
usual case: a fixed point cloud sampled ever more densely -> L saturates).  If L
grows ~linearly with N (genuinely new structure at every scale) the cost is
~quadratic in N and the GPU advantage narrows at very large N; build_time_s and
n_landmarks let you see which regime you are in.

Public API mirrors FastBallMapper: ``eps``, ``eps_dict``,
``points_covered_by_landmarks``, ``Graph`` (networkx; node attrs 'landmark',
'points covered', 'size'), ``n_landmarks``, ``landmark_point_ids``,
``build_time_s``, plus ``add_coloring`` / ``filter_by`` / ``points_and_balls``.
"""

from __future__ import annotations

import copy
import math
import time
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix


def _enable_tf32(enable: bool):
    torch.backends.cuda.matmul.allow_tf32 = enable
    torch.backends.cudnn.allow_tf32 = enable


def _wavefront_mis(Au_low):
    """Index-priority wavefront maximal independent set.

    ``Au_low`` is the STRICTLY-LOWER-triangular part of the symmetric eps
    adjacency over a set of B vertices, as a float tensor (Au_low[i, j] = 1 iff
    j < i and dist(i, j) <= eps).  Returns a bool[B] mask of selected vertices.

    Validity:
      * independence — if i < j were both selected in one round, j would have
        the active lower-indexed neighbour i and be disqualified; impossible.
      * maximality — the loop runs until no vertex is active; a vertex leaves
        the active set only by being selected or by being a neighbour of a
        selected vertex, so every vertex is a landmark or dominated by one.
    Full adjacency is recovered as Au_low + Au_low^T (the graph is symmetric),
    so only the lower triangle is stored.
    """
    B = Au_low.shape[0]
    dev = Au_low.device
    alive = torch.ones(B, dtype=torch.bool, device=dev)
    keep = torch.zeros(B, dtype=torch.bool, device=dev)
    if B == 0:
        return keep
    low = Au_low
    lowT = Au_low.t()
    while bool(alive.any()):
        alive_f = alive.to(low.dtype)
        # blocked[i] = has an active neighbour with strictly smaller index
        blocked = (low @ alive_f) > 0.5
        roots = alive & ~blocked
        keep |= roots
        roots_f = roots.to(low.dtype)
        # neighbours of roots (full adjacency = low + low^T)
        nbr = (low @ roots_f) + (lowT @ roots_f)
        alive = alive & ~(roots | (nbr > 0.5))
    return keep


class GpuBallMapper:
    """GPU Ball Mapper.  See module docstring.

    Parameters
    ----------
    X : (N, D) array-like
    eps : float                       ball radius
    coloring_df : pandas.DataFrame, optional
    order : array-like, optional      processed in index order (a note: the
                                      online path uses chunk-major order; any
                                      valid maximal net is produced)
    verbose : bool
    device : int | str                CUDA device ('cuda:0' or 0)
    precision : {'fp32', 'tf32'}      fp32 (DEFAULT) is the only eps-boundary-
                                      correct choice here; 'tf32' is ~2x faster
                                      but INVALID (mantissa too coarse for eps^2)
                                      and emits a warning if selected.
    chunk : int | None                streaming chunk size (auto from VRAM)
    lblk : int                        landmark column-tile for chunk x L GEMMs
    verify : bool                     run on-device validity asserts (C1, C2)
    """

    def __init__(
        self,
        X,
        eps,
        coloring_df=None,
        order=None,
        verbose=False,
        device="cuda:0",
        precision="fp32",
        chunk=None,
        lblk=8192,
        verify=False,
    ):
        if isinstance(device, int):
            device = f"cuda:{device}"
        dev = torch.device(device)
        if dev.type != "cuda":
            raise ValueError("GpuBallMapper requires a CUDA device")
        torch.cuda.set_device(dev)
        if precision == "tf32":
            warnings.warn(
                "GpuBallMapper(precision='tf32'): TF32 cannot resolve the eps "
                "boundary (abs error >> eps^2) and will likely produce an "
                "INVALID Ball Mapper. Use precision='fp32' (default).",
                stacklevel=2,
            )
        _enable_tf32(precision == "tf32")

        t_total = time.time()
        timing = {}
        self.eps = float(eps)
        self.eps_dict = None
        eps2 = float(eps) * float(eps)

        # ── upload ───────────────────────────────────────────────────────────
        torch.cuda.synchronize(dev)
        t0 = time.time()
        Xt = torch.as_tensor(np.ascontiguousarray(X, dtype=np.float32), device=dev)
        # Mean-center (euclidean distances are translation-invariant): shrinks
        # ||x||^2 so the Gram-trick squared distance sq_i+sq_j-2 x_i.x_j suffers
        # far less catastrophic cancellation near the eps boundary in fp32.
        Xt = Xt - Xt.mean(0, keepdim=True)
        N, D = Xt.shape
        sq = (Xt * Xt).sum(1)  # (N,) squared norms
        torch.cuda.synchronize(dev)
        timing["upload"] = time.time() - t0

        if chunk is None:
            chunk = self._auto_chunk(dev, N)
        chunk = max(1, int(chunk))  # guard N==0 / user chunk<=0 (range step-0)
        self._chunk = chunk

        # ── PASS 1: online maximal eps-net (landmark selection) ──────────────
        torch.cuda.synchronize(dev)
        t0 = time.time()
        Lpts = torch.empty((0, D), dtype=Xt.dtype, device=dev)  # landmark coords
        Lsq = torch.empty((0,), dtype=sq.dtype, device=dev)  # their sq-norms
        lm_ids = []  # global point ids
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            Xc = Xt[s:e]
            xn2 = sq[s:e]
            B = Xc.shape[0]
            Lcur = Lpts.shape[0]
            covered = torch.zeros(B, dtype=torch.bool, device=dev)
            if Lcur > 0:
                for j0 in range(0, Lcur, lblk):
                    j1 = min(j0 + lblk, Lcur)
                    G = Xc @ Lpts[j0:j1].T  # (B, blk) TF32 GEMM
                    D2 = xn2[:, None] + Lsq[j0:j1][None, :] - 2.0 * G
                    covered |= (D2 <= eps2).any(1)
                    del G, D2
            unc = (~covered).nonzero(as_tuple=True)[0]  # local uncovered idx
            if unc.numel() == 0:
                continue
            U = Xc[unc]
            un2 = xn2[unc]
            # intra-chunk eps adjacency (strictly-lower triangle), then MIS
            Gu = U @ U.T
            D2u = un2[:, None] + un2[None, :] - 2.0 * Gu
            Au_low = torch.tril((D2u <= eps2).to(torch.float32), diagonal=-1)
            del Gu, D2u
            keep = _wavefront_mis(Au_low)
            del Au_low
            sel = unc[keep]  # local indices kept
            Lpts = torch.cat([Lpts, Xc[sel]], 0)
            Lsq = torch.cat([Lsq, xn2[sel]], 0)
            lm_ids.extend((s + sel).tolist())
        L = Lpts.shape[0]
        torch.cuda.synchronize(dev)
        timing["select"] = time.time() - t0
        if verbose:
            print(
                f"  [Pass 1] {L} landmarks  {timing['select']:.3f}s  "
                f"(chunk={chunk}, {math.ceil(N/chunk)} chunks)"
            )

        # ── PASS 2: coverage of ALL points vs the FINAL landmark set ─────────
        torch.cuda.synchronize(dev)
        t0 = time.time()
        rows_chunks = []  # landmark id per (lm, pt) incidence
        cols_chunks = []  # point id
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            Xc = Xt[s:e]
            xn2 = sq[s:e]
            for j0 in range(0, L, lblk):
                j1 = min(j0 + lblk, L)
                G = Xc @ Lpts[j0:j1].T
                D2 = xn2[:, None] + Lsq[j0:j1][None, :] - 2.0 * G
                mem = D2 <= eps2  # (B, blk) bool
                pt, lj = mem.nonzero(as_tuple=True)
                rows_chunks.append((lj + j0).to(torch.int64))
                cols_chunks.append((pt + s).to(torch.int64))
                del G, D2, mem
        rows = (
            torch.cat(rows_chunks)
            if rows_chunks
            else torch.empty(0, dtype=torch.int64, device=dev)
        )
        cols = (
            torch.cat(cols_chunks)
            if cols_chunks
            else torch.empty(0, dtype=torch.int64, device=dev)
        )
        torch.cuda.synchronize(dev)
        timing["coverage"] = time.time() - t0

        # ── PASS 3: coverage grouping + presence-only edges (M @ M^T) ────────
        # At large N the host argsort + scipy SpGEMM dominate the whole run, so
        # this is done ON THE GPU (cupy + cuSPARSE) when available, with a host
        # (scipy) fallback.  Edges: off-diagonal nonzero of M @ M^T == a shared
        # covered point.  int32 accumulation (count <= N < 2^31; int8/int16
        # would OVERFLOW for large balls and silently drop edges).
        t0 = time.time()
        cov, all_edges, backend = self._coverage_and_edges(rows, cols, L, N, lm_ids)
        self.points_covered_by_landmarks = {v: cov[v] for v in range(L)}
        timing["edge_finding"] = time.time() - t0
        self.postproc_backend = backend
        if verbose:
            print(
                f"  [Pass 3] {len(all_edges)} edges  "
                f"{timing['edge_finding']:.3f}s  (postproc={backend})"
            )

        # ── build NetworkX graph ─────────────────────────────────────────────
        t0 = time.time()
        self.Graph = nx.Graph()
        self.Graph.add_nodes_from(range(L))
        self.Graph.add_edges_from(all_edges)
        for v in range(L):
            pts = self.points_covered_by_landmarks[v]
            self.Graph.nodes[v]["landmark"] = lm_ids[v]
            self.Graph.nodes[v]["points covered"] = pts
            self.Graph.nodes[v]["size"] = len(pts)
        timing["build_graph"] = time.time() - t0

        self.n_landmarks = L
        self.landmark_point_ids = lm_ids
        self.build_time_s = timing

        if verify:
            self._verify_on_device(Xt, sq, Lpts, Lsq, eps2, N, L, dev)

        if isinstance(coloring_df, pd.DataFrame):
            self.add_coloring(coloring_df)

        # free device memory promptly
        del Xt, sq, Lpts, Lsq, rows, cols
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()

        if verbose:
            print(
                f"  [Total] GpuBallMapper {time.time() - t_total:.3f}s "
                f"| upload={timing['upload']:.3f} select={timing['select']:.3f} "
                f"coverage={timing['coverage']:.3f} edges={timing['edge_finding']:.3f}"
            )

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _auto_chunk(dev, N):
        """Pick a streaming chunk.  The intra-chunk MIS cost scales as B^2 (the
        first chunk is all-uncovered, the worst case), while the chunk-vs-
        landmark GEMM total is independent of chunk size, so a SMALL chunk
        minimises selection time.  ~4096 is the empirical sweet spot (large
        enough to amortise per-chunk launch overhead, small enough that the
        dense first-chunk MIS triangle stays a few MB).  Capped by VRAM."""
        try:
            free, _ = torch.cuda.mem_get_info(dev)
        except Exception:
            free = 8 * 1024**3
        cap = int(math.sqrt(0.10 * free / 4.0))  # B^2 fp32 triangle < 10% VRAM
        return int(min(N, max(1024, min(cap, 4096))))

    @staticmethod
    def _coverage_and_edges(rows, cols, L, N, lm_ids):
        """Group the (landmark, point) incidence by landmark (-> per-node
        coverage arrays) and compute the presence-only edge set (off-diagonal
        nonzero of M @ M^T).  Returns (cov_list, edges_list, backend_str).

        Self-coverage is injected and the incidence de-duplicated: every
        landmark provably covers its OWN point (distance 0 <= eps), but the
        fp32 Gram trick can miss that self-pair when the data scale dwarfs eps,
        so we add (v, landmark_point_id[v]) explicitly and drop duplicates via
        a composite-key unique (which also sorts by (landmark, point), giving
        the per-node grouping for free).

        GPU path (cupy + cuSPARSE) keeps the work on-device — at large N the
        host sort and scipy SpGEMM dominate, so this is a big win.  Falls back
        to scipy on the host if cupy / dlpack interop is unavailable.
        """
        if L == 0:
            return [], [], "empty"
        lm_arr = np.asarray(lm_ids, dtype=np.int64)
        nnz_est = int(rows.numel()) + L

        def _host():
            rows_h = np.concatenate([rows.cpu().numpy(), np.arange(L, dtype=np.int64)])
            cols_h = np.concatenate([cols.cpu().numpy(), lm_arr])
            key = np.unique(rows_h * (N + 1) + cols_h)
            rk = key // (N + 1)
            ck = key % (N + 1)
            counts = np.bincount(rk, minlength=L)
            bounds = np.zeros(L + 1, dtype=np.int64)
            np.cumsum(counts, out=bounds[1:])
            cov = [ck[bounds[v] : bounds[v + 1]] for v in range(L)]
            data = np.ones(rk.size, dtype=np.int32)  # int32: counts<=N<2^31
            M = csr_matrix((data, (rk, ck)), shape=(L, N))
            S = (M @ M.T).tocoo()
            m = S.row < S.col
            return cov, list(zip(S.row[m].tolist(), S.col[m].tolist())), "host/scipy"

        # Trivially-small problems: the scipy host path is sub-millisecond AND
        # skips cuSPARSE entirely, so a cold one-shot tiny build avoids the ~6 s
        # one-time cuSPARSE init.  The threshold is low (30k incidence entries)
        # so that any substantial build still takes the on-device cupy path,
        # which is ~2x faster than host once cuSPARSE is warm.
        if nnz_est < 30_000:
            return _host()

        try:
            import os

            import cupy as cp
            from cupyx.scipy.sparse import csr_matrix as cucsr

            dbg = os.environ.get("BM_DEBUG")

            def _t():
                cp.cuda.Stream.null.synchronize()
                return time.time()

            t = _t()
            rcp = cp.from_dlpack(rows)  # zero-copy views of torch buffers
            ccp = cp.from_dlpack(cols)
            # inject self-coverage, then de-dup via a sorted composite key
            self_r = cp.arange(L, dtype=cp.int64)
            self_c = cp.asarray(lm_arr)
            rall = cp.concatenate([rcp, self_r])
            call = cp.concatenate([ccp, self_c])
            key = cp.unique(
                rall * (N + 1) + call
            )  # sorted (lm major, pt minor), unique
            rk = (key // (N + 1)).astype(cp.int64)
            ck = (key % (N + 1)).astype(cp.int64)
            nnz = int(rk.size)
            counts = cp.bincount(rk, minlength=L)
            bounds = cp.zeros(L + 1, dtype=cp.int64)
            cp.cumsum(counts, out=bounds[1:])
            if dbg:
                t1 = _t()
                print(f"    [pp] group {1000*(t1-t):.1f}ms", flush=True)
                t = t1
            # presence-only edges via cuSPARSE SpGEMM.  cupyx sparse supports
            # only float/bool dtypes; float32 represents integer intersection
            # counts exactly (<= N < 2^24 here is not required since presence
            # only needs >0, and counts <= N keep fp32 exact up to 2^24).
            data = cp.ones(nnz, dtype=cp.float32)
            M = cucsr((data, (rk, ck)), shape=(L, N))
            S = (M @ M.T).tocoo()
            m = S.row < S.col
            if dbg:
                t1 = _t()
                print(f"    [pp] spgemm {1000*(t1-t):.1f}ms", flush=True)
                t = t1
            eu = cp.asnumpy(S.row[m])
            ev = cp.asnumpy(S.col[m])
            ck_h = cp.asnumpy(ck)
            bounds_h = cp.asnumpy(bounds)
            if dbg:
                t1 = _t()
                print(f"    [pp] d2h {1000*(t1-t):.1f}ms", flush=True)
                t = t1
            cov = [ck_h[bounds_h[v] : bounds_h[v + 1]] for v in range(L)]
            edges = list(zip(eu.tolist(), ev.tolist()))
            if dbg:
                t1 = _t()
                print(f"    [pp] build {1000*(t1-t):.1f}ms", flush=True)
                t = t1
            del rcp, ccp, rall, call, key, rk, ck, M, S, data
            cp.get_default_memory_pool().free_all_blocks()
            return cov, edges, "gpu/cupy"
        except Exception as _e:
            import os as _os

            if _os.environ.get("BM_DEBUG"):
                import traceback

                print("    [pp] cupy path FAILED -> host:", repr(_e))
                traceback.print_exc()
            return _host()

    def _verify_on_device(self, Xt, sq, Lpts, Lsq, eps2, N, L, dev):
        """C1 maximality (covered union == N) and C2 independence (min landmark
        pair distance > eps), both on-device and cheap."""
        # C1: every point within eps of some landmark
        covered = torch.zeros(N, dtype=torch.bool, device=dev)
        for s in range(0, N, self._chunk):
            e = min(s + self._chunk, N)
            xn2 = sq[s:e]
            c = torch.zeros(e - s, dtype=torch.bool, device=dev)
            for j0 in range(0, L, 8192):
                j1 = min(j0 + 8192, L)
                G = Xt[s:e] @ Lpts[j0:j1].T
                D2 = xn2[:, None] + Lsq[j0:j1][None, :] - 2.0 * G
                c |= (D2 <= eps2).any(1)
            covered[s:e] = c
        assert bool(
            covered.all()
        ), f"C1 maximality FAILED: {int((~covered).sum())} uncovered points"
        # C2: landmarks pairwise > eps (min off-diagonal squared distance > eps2)
        bad = 0
        for i0 in range(0, L, 4096):
            i1 = min(i0 + 4096, L)
            G = Lpts[i0:i1] @ Lpts.T
            D2 = Lsq[i0:i1][None, :].T + Lsq[None, :] - 2.0 * G
            # mask self-pairs
            idx = torch.arange(i0, i1, device=dev)
            D2[torch.arange(i1 - i0, device=dev), idx] = float("inf")
            bad += int((D2 <= eps2).sum())
        assert bad == 0, f"C2 independence FAILED: {bad} within-eps landmark pairs"

    # ── pyballmapper-compatible methods ─────────────────────────────────────
    def add_coloring(
        self, coloring_df, custom_function=np.mean, custom_name=None, add_std=False
    ):
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
        filtered_bm = copy.deepcopy(self)
        pt_set = set(list_of_points)
        for node in filtered_bm.Graph.nodes:
            kept = list(
                set(int(x) for x in filtered_bm.points_covered_by_landmarks[node])
                & pt_set
            )
            filtered_bm.points_covered_by_landmarks[node] = np.array(kept)
            filtered_bm.Graph.nodes[node]["points covered"] = np.array(kept)
            filtered_bm.Graph.nodes[node]["size"] = len(kept)
        filtered_bm.Graph.remove_nodes_from(
            [n for n in filtered_bm.Graph if filtered_bm.Graph.nodes[n]["size"] == 0]
        )
        return filtered_bm

    def points_and_balls(self):
        rows = []
        for ball, points in self.points_covered_by_landmarks.items():
            for p in points:
                rows.append([int(p), ball])
        return pd.DataFrame(rows, columns=["point", "ball"])


if __name__ == "__main__":
    # Smoke test on synthetic data; assert validity invariants.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 10)).astype(np.float32)
    bm = GpuBallMapper(X=X, eps=1.5, verbose=True, verify=True)
    print(f"nodes={bm.n_landmarks}  edges={bm.Graph.number_of_edges()}")
