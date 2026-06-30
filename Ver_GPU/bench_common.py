"""bench_common.py — shared 3-way Ball Mapper benchmark harness.

Compares three implementations under a uniform interface, each repeated --reps
times (mean ± std), recording per-phase timings, landmark/edge Jaccard vs the
BallTree reference, and independent V1–V4 validity:

  original    : pyballmapper.BallMapper        — the ORIGINAL algorithm,
                serial, naive O(N·L) distance loop (no spatial index).
  BallTree    : fast_ball_mapper.FastBallMapper — sklearn BallTree + joblib (CPU).
                Used as the Jaccard reference (the canonical lexicographic net).
  GPU-waveMIS : gpu_ball_mapper.GpuBallMapper   — parallel index-priority
                wavefront-MIS + GEMM coverage + cuSPARSE edges (GPU).

Two driver scripts use this module:
  bench_eps_sweep.py  (Test 1): N fixed (100k, 1M), sweep eps 0.32 -> 0.02.
  bench_n_sweep.py    (Test 2): eps fixed, sweep N.

The original (naive) is INFEASIBLE at large N (O(N·L)); driver scripts pre-skip
it past a threshold and the guard caps any cell that exceeds the time budget —
every skip/cap is logged (never silent).
"""

from __future__ import annotations

import json
import os
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Core backends.  These pull in heavy deps (sklearn / cupy / torch); wrap them
# so the module still imports on a machine without a GPU or sklearn — that lets
# the table/plot helpers (e.g. median-based PNG regeneration) run anywhere from
# a saved repeat_results.json.  A missing dep disables exactly that method, not
# the whole module (same policy as the optional backends below).
try:
    from fast_ball_mapper import N_PHYSICAL_CORES, FastBallMapper

    _FBM_ERR = None
except Exception as _e:  # pragma: no cover
    FastBallMapper = None
    N_PHYSICAL_CORES = os.cpu_count() or 1
    _FBM_ERR = _e
try:
    import gen_synthetic as gs

    _GS_ERR = None
except Exception as _e:  # pragma: no cover
    gs = None
    _GS_ERR = _e
try:
    import bm_validate as bv

    _BV_ERR = None
except Exception as _e:  # pragma: no cover
    bv = None
    _BV_ERR = _e
try:
    from gpu_ball_mapper import GpuBallMapper

    _GPU_ERR = None
except Exception as _e:  # pragma: no cover
    GpuBallMapper = None
    _GPU_ERR = _e

# Optional backends (import errors are turned into clear runtime errors so a
# missing dependency disables exactly one method, not the whole run).
try:
    from pyballmapper import BallMapper as RefBallMapper

    _ORIG_ERR = None
except Exception as _e:  # pragma: no cover
    RefBallMapper = None
    _ORIG_ERR = _e

# name, color, marker, valid-by-construction.  This is the FULL catalog; the
# ACTIVE subset (METHODS / NAMES) is chosen by select_methods() below.
_ALL_METHODS = [
    ("original", "#7f7f7f", "P", True),
    ("BallTree", "#d62728", "o", True),
    ("GPU-waveMIS", "#1f77b4", "D", True),
]
REF = "BallTree"  # Jaccard reference
GPU = "GPU-waveMIS"  # speed-up baseline
# COLOR/MARKER cover the full catalog (harmless for inactive methods); only
# METHODS/NAMES are filtered, and run loops / tables / plots iterate NAMES.
COLOR = {m[0]: m[1] for m in _ALL_METHODS}
MARKER = {m[0]: m[2] for m in _ALL_METHODS}


def select_methods():
    """Set the active methods for this run; returns the active NAMES.

    Three methods are compared: the naive ``original`` (pyballmapper), the
    ``BallTree`` CPU baseline (also the Jaccard reference), and the
    ``GPU-waveMIS`` accelerator.  Mutates the module-level METHODS / NAMES
    consumed downstream."""
    global METHODS, NAMES
    METHODS = list(_ALL_METHODS)
    NAMES = [m[0] for m in METHODS]
    return NAMES


select_methods()


# ── builders ────────────────────────────────────────────────────────────────
def build(method, X, eps, device):
    if method == "original":
        if RefBallMapper is None:
            raise RuntimeError(f"pyballmapper unavailable: {_ORIG_ERR}")
        # float64 so the naive reference matches the float64 BallTree net exactly;
        # then any lm_jaccard<1 vs BallTree at tiny eps is purely a GPU-fp32 effect,
        # not a float32 artifact leaking into the CPU reference.
        return RefBallMapper(
            X=np.ascontiguousarray(X, dtype=np.float64), eps=float(eps)
        )
    if method == "BallTree":
        if FastBallMapper is None:
            raise RuntimeError(f"fast_ball_mapper unavailable: {_FBM_ERR}")
        return FastBallMapper(X=X, eps=eps, n_jobs=N_PHYSICAL_CORES)
    if method == "GPU-waveMIS":
        if GpuBallMapper is None:
            raise RuntimeError(f"gpu_ball_mapper unavailable: {_GPU_ERR}")
        return GpuBallMapper(X=X, eps=eps, device=device)
    raise ValueError(method)


# ── graph accessors (uniform across all four backends) ───────────────────────
def _lm_pids(bm):
    return set(int(bm.Graph.nodes[n]["landmark"]) for n in bm.Graph.nodes)


def _n_land(bm):
    return int(bm.Graph.number_of_nodes())


def _n_edges(bm):
    return int(bm.Graph.number_of_edges())


def _pid_edges(bm):
    pid = {n: int(bm.Graph.nodes[n]["landmark"]) for n in bm.Graph.nodes}
    return set(frozenset((pid[u], pid[v])) for u, v in bm.Graph.edges)


def jaccard_vs(ref_bm, bm):
    lr, lg = _lm_pids(ref_bm), _lm_pids(bm)
    er, eg = _pid_edges(ref_bm), _pid_edges(bm)
    lj = len(lr & lg) / len(lr | lg) if (lr | lg) else 1.0
    ej = len(er & eg) / len(er | eg) if (er | eg) else 1.0
    return round(lj, 6), round(ej, 6)


def validate_result(X, eps, bm, N, atol=1e-4):
    if bv is None:  # validation backend unavailable
        return None, {"error": f"bm_validate unavailable: {_BV_ERR}"}
    lm = [int(bm.Graph.nodes[n]["landmark"]) for n in bm.Graph.nodes]
    cov = {
        n: np.asarray(bm.points_covered_by_landmarks[n], dtype=np.int64)
        for n in bm.Graph.nodes
    }
    edges = list(bm.Graph.edges)
    # The exact validate() cost is driven by L, not N: its V1/V3 checks loop in
    # pure Python over EVERY landmark, so at the L->N degeneracy (eps=0.03 gives
    # L~98.5k at N=100k) it runs for tens of minutes PER method — the bottleneck
    # that stalled the resume.  Switch to the sampled validate_large() once L is
    # large too, not only once N is large.
    try:
        if N > 100000 or len(lm) > 50000:
            vr = bv.validate_large(X, eps, lm, cov, edges, atol=atol)
        else:
            vr = bv.validate(X, eps, lm, cov, edges, atol=atol)
        return bool(vr["valid"]), vr
    except Exception as ex:  # pragma: no cover
        return None, {"error": f"{type(ex).__name__}: {str(ex)[:120]}"}


# ── timed run of one (method, X, eps) cell, with time guards ─────────────────
def run_cell(
    method,
    X,
    eps,
    device,
    reps,
    per_rep_ceiling,
    cell_budget,
    pre_skip=False,
    skip_reason="",
    log=print,
):
    """Run up to `reps` builds; stop early if a single rep exceeds
    `per_rep_ceiling` or the cumulative cell time exceeds `cell_budget`.
    Returns a dict with per-rep totals + phase timings and the last object."""
    if pre_skip:
        log(f"      [{method:>11}] SKIP — {skip_reason}")
        return {
            "runs": [],
            "n_reps": 0,
            "skipped": True,
            "note": skip_reason,
            "last": None,
        }
    runs, last, cum, note = [], None, 0.0, ""
    for r in range(reps):
        try:
            t0 = time.perf_counter()
            obj = build(method, X, eps, device)
            dt = time.perf_counter() - t0
        except Exception as ex:
            note = f"FAILED rep{r}: {type(ex).__name__}: {str(ex)[:110]}"
            log(f"      [{method:>11}] {note}")
            # free fragmented VRAM before the next cell (GPU OOM hygiene).  We
            # return on the FIRST failure (no same-cell retry), so an OOM costs
            # at most one attempt; descending-eps means it hits the last cell.
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
            return {
                "runs": runs,
                "n_reps": len(runs),
                "failed": True,
                "note": note,
                "last": last,
            }
        rec = {"rep": r, "total": dt}
        bt = getattr(obj, "build_time_s", None)
        if isinstance(bt, dict):
            rec["phases"] = {k: float(v) for k, v in bt.items()}
        runs.append(rec)
        last = obj
        cum += dt
        log(
            f"      [{method:>11}] rep {r + 1}/{reps}  {dt:8.3f}s  "
            f"(L={_n_land(obj)})"
        )
        if dt > per_rep_ceiling:
            note = (
                f"capped: single rep {dt:.0f}s > {per_rep_ceiling}s ceiling "
                f"({len(runs)} rep(s) done)"
            )
            log("        " + note)
            break
        if cum > cell_budget:
            note = (
                f"capped: cell budget {cell_budget}s reached "
                f"({len(runs)} rep(s) done)"
            )
            log("        " + note)
            break
    return {"runs": runs, "n_reps": len(runs), "last": last, "note": note}


class _ShimBM:
    """Lightweight stand-in for a Ball Mapper object, reconstructed from the
    portable summary a subprocess returns (see run_cell_isolated /
    _orig_worker.py).  Exposes exactly the attributes the graph accessors and
    the validator touch: ``.Graph`` (a networkx graph with the 'landmark' node
    attribute and the edges) and ``.points_covered_by_landmarks``."""

    def __init__(self, summary):
        import networkx as nx

        G = nx.Graph()
        for n, pid in summary["nodes"]:
            G.add_node(n, landmark=pid)
        G.add_edges_from(summary["edges"])
        self.Graph = G
        self.points_covered_by_landmarks = summary["covers"]
        ph = summary.get("phases")
        self.build_time_s = ph if isinstance(ph, dict) else None


def run_cell_isolated(
    method,
    X,
    eps,
    device,
    reps,
    per_rep_ceiling,
    cell_budget,
    pre_skip=False,
    skip_reason="",
    log=print,
    worker="_orig_worker.py",
    hard_timeout=None,
):
    """Subprocess-isolated variant of run_cell (used for the naive 'original').

    Each build runs in a FRESH python process (no GPU/CUDA) so a NATIVE segfault
    at the L->N degeneracy region is *contained*: the parent records a failed
    cell and the sweep continues, instead of the whole process dying with rc=139
    (the failure mode that killed the original eps-sweep at eps=0.03).  Same
    return contract as run_cell.  ``hard_timeout`` (s, optional) kills a single
    rep that runs away — None keeps run_cell's "cap only after completion"
    semantics (so a legitimately slow naive rep is still recorded, not lost)."""
    if pre_skip:
        log(f"      [{method:>11}] SKIP — {skip_reason}")
        return {
            "runs": [],
            "n_reps": 0,
            "skipped": True,
            "note": skip_reason,
            "last": None,
        }

    import pickle
    import subprocess
    import sys
    import tempfile

    runs, last, cum, note = [], None, 0.0, ""
    xfd, xpath = tempfile.mkstemp(suffix="_X.npy")
    os.close(xfd)
    np.save(xpath, np.ascontiguousarray(X))
    wpath = os.path.join(HERE, worker)
    try:
        for r in range(reps):
            ofd, opath = tempfile.mkstemp(suffix="_out.pkl")
            os.close(ofd)
            try:
                try:
                    proc = subprocess.run(
                        [sys.executable, wpath, xpath, repr(float(eps)), opath],
                        cwd=HERE,
                        capture_output=True,
                        text=True,
                        timeout=hard_timeout,
                    )
                except subprocess.TimeoutExpired:
                    note = (
                        f"FAILED rep{r}: isolated build exceeded hard_timeout "
                        f"{hard_timeout}s"
                    )
                    log(f"      [{method:>11}] {note}")
                    return {
                        "runs": runs,
                        "n_reps": len(runs),
                        "failed": True,
                        "note": note,
                        "last": last,
                    }
                if proc.returncode != 0:
                    err = (proc.stderr or "").strip().splitlines()
                    tailmsg = (": " + err[-1][:100]) if err else ""
                    sig = (
                        f" (killed by signal {-proc.returncode})"
                        if proc.returncode < 0
                        else " (SIGSEGV)" if proc.returncode == 139 else ""
                    )
                    note = (
                        f"FAILED rep{r}: isolated build rc={proc.returncode}"
                        f"{sig}{tailmsg}"
                    )
                    log(f"      [{method:>11}] {note}")
                    return {
                        "runs": runs,
                        "n_reps": len(runs),
                        "failed": True,
                        "note": note,
                        "last": last,
                    }
                with open(opath, "rb") as fh:
                    summary = pickle.load(fh)
            finally:
                try:
                    os.remove(opath)
                except OSError:
                    pass
            dt = float(summary["build_dt"])
            obj = _ShimBM(summary)
            rec = {"rep": r, "total": dt}
            if isinstance(summary.get("phases"), dict):
                rec["phases"] = {k: float(v) for k, v in summary["phases"].items()}
            runs.append(rec)
            last = obj
            cum += dt
            log(
                f"      [{method:>11}] rep {r + 1}/{reps}  {dt:8.3f}s  "
                f"(L={_n_land(obj)})  [isolated]"
            )
            if dt > per_rep_ceiling:
                note = (
                    f"capped: single rep {dt:.0f}s > {per_rep_ceiling}s ceiling "
                    f"({len(runs)} rep(s) done)"
                )
                log("        " + note)
                break
            if cum > cell_budget:
                note = (
                    f"capped: cell budget {cell_budget}s reached "
                    f"({len(runs)} rep(s) done)"
                )
                log("        " + note)
                break
    finally:
        try:
            os.remove(xpath)
        except OSError:
            pass
    return {"runs": runs, "n_reps": len(runs), "last": last, "note": note}


def mean_std(vals):
    if not vals:
        return None, None
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


def summarize_cell(method, x, N, eps, cell, ref_bm, X):
    """Build the per-(x, method) result record from a run_cell() output."""
    totals = [r["total"] for r in cell.get("runs", [])]
    tm, ts = mean_std(totals)
    # phase aggregation
    phase_keys = set()
    for r in cell.get("runs", []):
        phase_keys |= set(r.get("phases", {}).keys())
    phase_means, phase_stds = {}, {}
    for k in sorted(phase_keys):
        vals = [
            r["phases"][k]
            for r in cell.get("runs", [])
            if "phases" in r and k in r["phases"]
        ]
        m, s = mean_std(vals)
        if m is not None:
            phase_means[k] = m
            phase_stds[k] = s
    rec = {
        "x": x,
        "N": N,
        "eps": eps,
        "method": method,
        "n_reps": cell.get("n_reps", 0),
        "totals": totals,
        "t_mean": tm,
        "t_std": ts,
        "phase_means": phase_means,
        "phase_stds": phase_stds,
        "skipped": bool(cell.get("skipped", False)),
        "failed": bool(cell.get("failed", False)),
        "note": cell.get("note", ""),
        "L": None,
        "E": None,
        "valid": None,
        "lm_jaccard": None,
        "edge_jaccard": None,
    }
    last = cell.get("last")
    if last is not None:
        rec["L"] = _n_land(last)
        rec["E"] = _n_edges(last)
        if X is not None:  # X is None => --no-validate
            v, _vr = validate_result(X, eps, last, N)
            rec["valid"] = v
        if ref_bm is not None:
            lj, ej = jaccard_vs(ref_bm, last)
            rec["lm_jaccard"] = lj
            rec["edge_jaccard"] = ej
        elif method == REF:
            rec["lm_jaccard"] = 1.0
            rec["edge_jaccard"] = 1.0
    return rec


# ── output: JSON, tables, plots ──────────────────────────────────────────────
def flush_json(results, path):
    with open(path, "w") as fh:
        json.dump(results, fh, indent=1)


def _cells_at(results, x):
    return {c["method"]: c for c in results["cells"] if c["x"] == x}


def _fmt_ms(m, s):
    return f"{m:.4f} ± {s:.4f}" if m is not None else "—"


def write_timing_table(results, xlabel, path):
    xs = results["xs"]
    L = [
        f"{len(NAMES)}-way Ball Mapper build time — mean ± std over up to "
        f"{results['reps']} reps  ({results['fixed_str']}; host={results['host']})\n",
        "  "
        + f"{xlabel:>10} | "
        + " | ".join(f"{n + ' (s)':>22}" for n in NAMES)
        + " | GPU-waveMIS speed-up over "
        + "/".join(n for n in NAMES if n != GPU),
    ]
    L.append("  " + "-" * 150)
    for x in xs:
        cm = _cells_at(results, x)
        cells = []
        for n in NAMES:
            c = cm.get(n)
            cells.append(
                _fmt_ms(c["t_mean"], c["t_std"])
                if c and c["t_mean"] is not None
                else "—"
            )
        gpu = cm.get(GPU)
        gm = gpu["t_mean"] if gpu else None
        sp = []
        for n in NAMES:
            if n == GPU:
                continue
            c = cm.get(n)
            m = c["t_mean"] if c else None
            sp.append(f"{m / gm:.1f}x" if (m is not None and gm) else "—")
        L.append(
            "  "
            + f"{x:>10} | "
            + " | ".join(f"{c:>22}" for c in cells)
            + " | "
            + " / ".join(sp)
        )
    txt = "\n".join(L)
    print("\n" + txt, flush=True)
    with open(path, "w") as fh:
        fh.write(txt + "\n")


def write_validity_table(results, xlabel, path):
    xs = results["xs"]
    L = [
        "\nGraph size, validity & Jaccard-vs-BallTree per (%s, method)\n" % xlabel,
        "  " + f"{xlabel:>10} | {'method':>12} | {'L':>7} | {'E':>8} | "
        f"{'valid':>6} | {'lm_jacc':>8} | {'edge_jacc':>9} | {'reps':>4} | note",
    ]
    L.append("  " + "-" * 110)
    for x in xs:
        cm = _cells_at(results, x)
        for n in NAMES:
            c = cm.get(n)
            if not c:
                continue
            Ls = "—" if c["L"] is None else str(c["L"])
            Es = "—" if c["E"] is None else str(c["E"])
            vs = "—" if c["valid"] is None else str(c["valid"])
            lj = "—" if c["lm_jaccard"] is None else f"{c['lm_jaccard']:.4f}"
            ej = "—" if c["edge_jaccard"] is None else f"{c['edge_jaccard']:.4f}"
            L.append(
                "  " + f"{x:>10} | {n:>12} | {Ls:>7} | {Es:>8} | "
                f"{vs:>6} | {lj:>8} | {ej:>9} | {c['n_reps']:>4} | {c['note']}"
            )
    txt = "\n".join(L)
    print(txt, flush=True)
    with open(path, "w") as fh:
        fh.write(txt + "\n")


def write_phase_table(results, xlabel, path):
    """Per-phase mean±std for the instrumented methods (original has none)."""
    xs = results["xs"]
    L = ["\nPer-phase build time mean±std (s) — methods that expose build_time_s\n"]
    for x in xs:
        cm = _cells_at(results, x)
        L.append(f"  {xlabel} = {x}")
        for n in NAMES:
            c = cm.get(n)
            if not c or not c["phase_means"]:
                if c and c["t_mean"] is not None:
                    L.append(
                        f"    {n:>12}: total {c['t_mean']:.4f} ± {c['t_std']:.4f}  (no phase breakdown)"
                    )
                continue
            parts = [
                f"{k}={c['phase_means'][k]:.4f}±{c['phase_stds'][k]:.4f}"
                for k in c["phase_means"]
            ]
            L.append(
                f"    {n:>12}: total {c['t_mean']:.4f} ± {c['t_std']:.4f} | "
                + "  ".join(parts)
            )
    txt = "\n".join(L)
    with open(path, "w") as fh:
        fh.write(txt + "\n")


def _central(c, how="mean"):
    """Central build-time estimate for a cell, with asymmetric error bars.
    how='mean'   -> (mean, std, std)            (default; matches t_mean±t_std)
    how='median' -> (median, median-p25, p75-median)   robust to one-off cold
                    GPU reps (e.g. the N=10k cuSPARSE first-call spike) that
                    inflate the mean by >10x.  Returns None if no timing."""
    if not c or c.get("t_mean") is None:
        return None
    tot = c.get("totals") or []
    if how == "median" and tot:
        a = np.asarray(tot, dtype=float)
        med = float(np.median(a))
        lo = float(np.percentile(a, 25))
        hi = float(np.percentile(a, 75))
        return med, max(0.0, med - lo), max(0.0, hi - med)
    m = c["t_mean"]
    s = c.get("t_std") or 0.0
    return m, s, s


def plot_sweep(results, xlabel, include_original, path, xlog=True, central="mean"):
    """Two-panel: (left) build time vs x; (right) ×slower-than-GPU vs x.
    central='mean' uses mean±std; central='median' uses median+IQR — robust to
    the one-off cold-start GPU reps (cuSPARSE/CUDA first-call init) that inflate
    the GPU mean (most visibly the N=10k spike, mean 0.41s vs median 0.025s)."""
    xs = results["xs"]
    names = NAMES if include_original else [n for n in NAMES if n != "original"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for n in names:
        px, pm, plo, phi = [], [], [], []
        for x in xs:
            ct = _central(_cells_at(results, x).get(n), central)
            if ct is not None:
                m, lo, hi = ct
                px.append(x)
                pm.append(m)
                plo.append(lo)
                phi.append(hi)
        if not px:
            continue
        ax1.errorbar(
            px,
            pm,
            yerr=[plo, phi],
            fmt=MARKER[n] + "-",
            color=COLOR[n],
            lw=2,
            capsize=4,
            label=n,
        )
    if xlog:
        ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("wall-clock build time (s)")
    ax1.set_title(f"Build time vs {xlabel}  ({results['fixed_str']})")
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    ax1.legend()

    for n in names:
        if n == GPU:
            continue
        px, pm, plo, phi = [], [], [], []
        for x in xs:
            cm = _cells_at(results, x)
            ct = _central(cm.get(n), central)
            gt = _central(cm.get(GPU), central)
            if ct is not None and gt is not None and gt[0]:
                cmv, clo, chi = ct
                gmv, glo, ghi = gt
                ratio = cmv / gmv
                rel_c = ((clo + chi) / 2) / cmv if cmv else 0.0
                rel_g = ((glo + ghi) / 2) / gmv if gmv else 0.0
                rel = (rel_c**2 + rel_g**2) ** 0.5
                px.append(x)
                pm.append(ratio)
                plo.append(ratio * rel)
                phi.append(ratio * rel)
        if not px:
            continue
        ax2.errorbar(
            px,
            pm,
            yerr=[plo, phi],
            fmt=MARKER[n] + "-",
            color=COLOR[n],
            lw=2,
            capsize=4,
            label=f"{n} / GPU-waveMIS",
        )
    ax2.axhline(1.0, color=COLOR[GPU], ls="--", alpha=0.6, label="GPU-waveMIS (1×)")
    if xlog:
        ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("× slower than GPU-waveMIS")
    ax2.set_title("GPU-waveMIS speed-up")
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    ax2.legend()

    stat = "median + IQR" if central == "median" else "mean ± std"
    n_shown = len(names)
    if include_original:
        has_orig = any(
            (_cells_at(results, x).get("original") or {}).get("t_mean") is not None
            for x in xs
        )
        tag = (
            f"{n_shown} methods (incl. original)"
            if has_orig
            else f"{n_shown} methods — original infeasible/skipped, not shown"
        )
    else:
        tag = f"{n_shown} optimized methods"
    fig.suptitle(
        f"{results['title']} — {tag}  [{stat}]", fontsize=13, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote", path, flush=True)


def plot_diagnostic(results, xlabel, path, xlog=True):
    """L and L/N vs x (left), and lm/edge Jaccard-vs-BallTree vs x (right) —
    surfaces the L->N degeneracy and the fp32 eps-boundary break-even."""
    xs = results["xs"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    # left: L per method + L/N for the reference
    for n in NAMES:
        px, pL = [], []
        for x in xs:
            c = _cells_at(results, x).get(n)
            if c and c["L"] is not None:
                px.append(x)
                pL.append(c["L"])
        if px:
            ax1.plot(px, pL, MARKER[n] + "-", color=COLOR[n], lw=2, label=f"L ({n})")
    if xlog:
        ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("landmark count L")
    ax1.set_title("Landmark count vs %s" % xlabel)
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    ax1.legend(fontsize=8)
    # right: jaccard of GPU & original vs BallTree
    for n in NAMES:
        if n == REF:
            continue
        px, plj, pej = [], [], []
        for x in xs:
            c = _cells_at(results, x).get(n)
            if c and c["lm_jaccard"] is not None:
                px.append(x)
                plj.append(c["lm_jaccard"])
                pej.append(c["edge_jaccard"])
        if px:
            ax2.plot(
                px, plj, MARKER[n] + "-", color=COLOR[n], lw=2, label=f"lm_jacc ({n})"
            )
            ax2.plot(
                px,
                pej,
                MARKER[n] + ":",
                color=COLOR[n],
                lw=1.5,
                alpha=0.7,
                label=f"edge_jacc ({n})",
            )
    if xlog:
        ax2.set_xscale("log")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Jaccard vs BallTree")
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_title("Graph identity vs BallTree (fp32 break-even)")
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    ax2.legend(fontsize=8)
    fig.suptitle(
        f"{results['title']} — diagnostics (L and Jaccard)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote", path, flush=True)


def warmup(device, ref, eps, big_n, log=print):
    """Pay one-time GPU init (cuBLAS/cuSPARSE) and FastBallMapper import cost so
    no timed rep eats it."""
    log("warming up backends ...")
    try:
        _ = GpuBallMapper(X=ref, eps=eps, device=device)
        Xw = gs.make_bootstrap(ref, min(big_n, 200000), seed=7).astype(np.float32)
        _ = GpuBallMapper(X=Xw, eps=eps, device=device)
        log("  GPU/cuSPARSE warmed")
    except Exception as ex:
        log(f"  GPU warmup note: {type(ex).__name__} {ex}")
    try:
        _ = FastBallMapper(X=ref[:2000], eps=eps, n_jobs=N_PHYSICAL_CORES)
    except Exception:
        pass
