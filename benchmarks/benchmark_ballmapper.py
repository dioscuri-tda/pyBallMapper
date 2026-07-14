"""benchmark_ballmapper.py — benchmark BallMapper scaling behaviour.

Measures how BallMapper construction time and memory usage scale with:
  - N (dataset size) at fixed eps
  - eps (ball radius) at fixed N
across different landmark methods (greedy, nearest, adaptive).

Outputs (default ``--out .``):
    report.html      single self-contained report (tables + embedded plots)
    results.json     raw per-cell timings / sizes / memory

Usage
-----
    python benchmarks/benchmark_ballmapper.py
    python benchmarks/benchmark_ballmapper.py --ns 500 1000 2000 --reps 5
    python benchmarks/benchmark_ballmapper.py --data path/to/data.npy --methods greedy nearest
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import tracemalloc
from collections.abc import Callable
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Ensure parent directory is importable when running as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))

import gen_synthetic as gs  # noqa: E402
from html_report import Report  # noqa: E402

from pyballmapper import BallMapper  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────────
def _landmark_ids(bm: BallMapper) -> list[int]:
    """Ordered landmark point-ids from a BallMapper graph."""
    return [int(bm.Graph.nodes[n]["landmark"]) for n in bm.Graph.nodes]


def _edge_pid_set(bm: BallMapper) -> set[frozenset[int]]:
    """Set of unordered landmark-id pairs for every edge."""
    pid = {n: int(bm.Graph.nodes[n]["landmark"]) for n in bm.Graph.nodes}
    return set(frozenset((pid[u], pid[v])) for u, v in bm.Graph.edges)


def _identical(bm_a: BallMapper, bm_b: BallMapper) -> bool:
    """Check two BallMapper graphs are identical (same landmarks + edges)."""
    return _landmark_ids(bm_a) == _landmark_ids(bm_b) and _edge_pid_set(
        bm_a
    ) == _edge_pid_set(bm_b)


def _timed_build(
    X: np.ndarray, eps: float, method: str | None, reps: int
) -> tuple[list[float], list[float], BallMapper]:
    """Build BallMapper *reps* times.

    Returns (all_times_s, all_peak_rss_mb, last_object).
    Peak RSS is measured via ``tracemalloc``.
    """
    ts: list[float] = []
    peak_mbs: list[float] = []
    obj: BallMapper | None = None
    for _ in range(reps):
        tracemalloc.start()
        t0 = time.perf_counter()
        obj = BallMapper(X=X, eps=eps, method=method)
        elapsed = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        ts.append(elapsed)
        peak_mbs.append(peak_bytes / (1024 * 1024))
    assert obj is not None  # noqa: S101
    return ts, peak_mbs, obj


# ── test 1: N-scaling ───────────────────────────────────────────────────────
def run_n_scaling(
    methods: list[str | None],
    ns: list[int],
    eps: float,
    reps: int,
    d: int,
    log: Callable[..., Any] = print,
) -> list[dict]:
    """For each method, build BallMapper on datasets of increasing N."""
    rows: list[dict] = []
    for method in methods:
        label = method or "greedy"
        log(f"  method={label}  eps={eps:.4f}", flush=True)
        for n in ns:
            X = gs.make_highd(n, d=d, seed=n)
            all_times, all_mems, bm = _timed_build(X, eps, method, reps)
            time_arr = np.array(all_times)
            mem_arr = np.array(all_mems)
            row = {
                "method": label,
                "N": n,
                "eps": eps,
                "L": int(bm.Graph.number_of_nodes()),
                "E": int(bm.Graph.number_of_edges()),
                "time_mean": float(time_arr.mean()),
                "time_std": float(time_arr.std()),
                "peak_rss_mean_mb": float(mem_arr.mean()),
                "peak_rss_std_mb": float(mem_arr.std()),
                "times": all_times,
                "peak_rss_mb": all_mems,
            }
            rows.append(row)
            log(
                f"    N={n:>6d}  L={row['L']:>5d}  E={row['E']:>5d}  "
                f"t={row['time_mean']:.3f}+/-{row['time_std']:.3f}s  "
                f"mem={row['peak_rss_mean_mb']:.1f}+/-{row['peak_rss_std_mb']:.1f}MB"
            )
    return rows


# ── test 2: eps-scaling ─────────────────────────────────────────────────────
def run_eps_scaling(
    methods: list[str | None],
    eps_list: list[float],
    n: int,
    reps: int,
    d: int,
    log: Callable[..., Any] = print,
) -> list[dict]:
    """For each method, build BallMapper with decreasing eps at fixed N."""
    rows: list[dict] = []
    X = gs.make_highd(n, d=d, seed=42)
    for method in methods:
        label = method or "greedy"
        log(f"  method={label}  N={n}", flush=True)
        for eps in eps_list:
            all_times, all_mems, bm = _timed_build(X, eps, method, reps)
            time_arr = np.array(all_times)
            mem_arr = np.array(all_mems)
            row = {
                "method": label,
                "N": n,
                "eps": eps,
                "L": int(bm.Graph.number_of_nodes()),
                "E": int(bm.Graph.number_of_edges()),
                "time_mean": float(time_arr.mean()),
                "time_std": float(time_arr.std()),
                "peak_rss_mean_mb": float(mem_arr.mean()),
                "peak_rss_std_mb": float(mem_arr.std()),
                "times": all_times,
                "peak_rss_mb": all_mems,
            }
            rows.append(row)
            log(
                f"    eps={eps:.4f}  L={row['L']:>5d}  E={row['E']:>5d}  "
                f"t={row['time_mean']:.3f}+/-{row['time_std']:.3f}s  "
                f"mem={row['peak_rss_mean_mb']:.1f}+/-{row['peak_rss_std_mb']:.1f}MB"
            )
    return rows


# ── plots ────────────────────────────────────────────────────────────────────
_COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#e11d48"]
_MARKERS = ["o", "s", "D", "^", "v", "P"]


def _apply_ax_style(ax: matplotlib.axes.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(ax.get_title(), fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, which="major", ls="-", color="#e8e8e8", alpha=0.7)
    ax.grid(True, which="minor", ls="--", color="#f0f0f0", alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9, fontsize=10)


def _plot_method(
    ax: matplotlib.axes.Axes, xs: list, ys: list, yerr: list, label: str, idx: int
) -> None:
    color = _COLORS[idx % len(_COLORS)]
    marker = _MARKERS[idx % len(_MARKERS)]
    ax.plot(
        xs,
        ys,
        marker=marker,
        color=color,
        lw=2.5,
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label=label,
        zorder=3,
    )
    ax.fill_between(
        xs,
        [y - e for y, e in zip(ys, yerr)],
        [y + e for y, e in zip(ys, yerr)],
        color=color,
        alpha=0.15,
        zorder=2,
    )


def _n_scaling_figure(rows: list[dict], eps: float) -> matplotlib.figure.Figure:
    methods = sorted({r["method"] for r in rows})
    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(14, 5.5))
    for i, m in enumerate(methods):
        subset = [r for r in rows if r["method"] == m]
        ns = [r["N"] for r in subset]
        ts = [r["time_mean"] for r in subset]
        ts_err = [r["time_std"] for r in subset]
        mems = [r["peak_rss_mean_mb"] for r in subset]
        mem_err = [r["peak_rss_std_mb"] for r in subset]
        _plot_method(ax_time, ns, ts, ts_err, m, i)
        _plot_method(ax_mem, ns, mems, mem_err, m, i)
    _apply_ax_style(ax_time, "N (points)", "build time (s)")
    _apply_ax_style(ax_mem, "N (points)", "peak RSS (MB)")
    ax_time.set_title(
        f"Build time vs N (eps={eps:.4f})", fontsize=13, fontweight="bold", pad=10
    )
    ax_mem.set_title(
        f"Peak memory vs N (eps={eps:.4f})", fontsize=13, fontweight="bold", pad=10
    )
    fig.tight_layout(pad=1.5)
    return fig


def _eps_scaling_figure(rows: list[dict], n: int) -> matplotlib.figure.Figure:
    methods = sorted({r["method"] for r in rows})
    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(14, 5.5))
    for i, m in enumerate(methods):
        subset = [r for r in rows if r["method"] == m]
        eps_vals = [r["eps"] for r in subset]
        ts = [r["time_mean"] for r in subset]
        ts_err = [r["time_std"] for r in subset]
        mems = [r["peak_rss_mean_mb"] for r in subset]
        mem_err = [r["peak_rss_std_mb"] for r in subset]
        _plot_method(ax_time, eps_vals, ts, ts_err, m, i)
        _plot_method(ax_mem, eps_vals, mems, mem_err, m, i)
    _apply_ax_style(ax_time, "eps (ball radius)", "build time (s)")
    _apply_ax_style(ax_mem, "eps (ball radius)", "peak RSS (MB)")
    ax_time.set_title(
        f"Build time vs eps (N={n})", fontsize=13, fontweight="bold", pad=10
    )
    ax_mem.set_title(
        f"Peak memory vs eps (N={n})", fontsize=13, fontweight="bold", pad=10
    )
    fig.tight_layout(pad=1.5)
    return fig


def _landmarks_figure(rows: list[dict], eps: float) -> matplotlib.figure.Figure:
    methods = sorted({r["method"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, m in enumerate(methods):
        subset = [r for r in rows if r["method"] == m]
        ns = [r["N"] for r in subset]
        landmarks = [r["L"] for r in subset]
        color = _COLORS[i % len(_COLORS)]
        marker = _MARKERS[i % len(_MARKERS)]
        ax.plot(
            ns,
            landmarks,
            marker=marker,
            color=color,
            lw=2.5,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=m,
            zorder=3,
        )
    _apply_ax_style(ax, "N (points)", "number of landmarks")
    ax.set_title(
        f"Number of landmarks vs N (eps={eps:.4f})",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    fig.tight_layout(pad=1.5)
    return fig


# ── report ───────────────────────────────────────────────────────────────────
def build_report(
    n_rows: list[dict],
    eps_rows: list[dict],
    meta: dict,
    out_path: str,
) -> str:
    rep = Report(
        "BallMapper scaling benchmark",
        subtitle=(
            f"host={meta['host']} · reps={meta['reps']} · "
            f"d={meta['d']} · {meta['timestamp']}"
        ),
    )

    # ── Test 1: N-scaling ──
    rep.h2("Test 1 — N-scaling at fixed eps")
    rep.p(
        f"Eps = <code>{meta['scaling_eps']:.4f}</code>.  "
        f"Landmark methods: <code>{', '.join(meta['methods'])}</code>."
    )
    eps = meta["scaling_eps"]
    headers = ["method", "N", "L", "E", "time (s)", "peak RSS (MB)"]
    table_rows: list[list[str]] = []
    for r in n_rows:
        table_rows.append(
            [
                r["method"],
                f"{r['N']:,}",
                str(r["L"]),
                str(r["E"]),
                f"{r['time_mean']:.3f} +/- {r['time_std']:.3f}",
                f"{r['peak_rss_mean_mb']:.1f} +/- {r['peak_rss_std_mb']:.1f}",
            ]
        )
    rep.table(headers, table_rows)
    rep.figure(
        _n_scaling_figure(n_rows, eps),
        caption="Left: build time vs N (log-log). Right: peak RSS vs N (log-log).",
    )
    rep.figure(
        _landmarks_figure(n_rows, eps),
        caption="Number of landmarks vs N (log-log).",
    )

    # ── Test 2: eps-scaling ──
    rep.h2("Test 2 — eps-scaling at fixed N")
    rep.p(f"N = <code>{meta['scaling_n']:,}</code>.")
    headers = ["method", "eps", "L", "E", "time (s)", "peak RSS (MB)"]
    table_rows = []
    for r in eps_rows:
        table_rows.append(
            [
                r["method"],
                f"{r['eps']:.4f}",
                str(r["L"]),
                str(r["E"]),
                f"{r['time_mean']:.3f} +/- {r['time_std']:.3f}",
                f"{r['peak_rss_mean_mb']:.1f} +/- {r['peak_rss_std_mb']:.1f}",
            ]
        )
    rep.table(headers, table_rows)
    rep.figure(
        _eps_scaling_figure(eps_rows, meta["scaling_n"]),
        caption="Left: build time vs eps (log-log). Right: peak RSS vs eps (log-log).",
    )

    # ── Notes ──
    rep.h2("Notes")
    rep.html(
        "<ul>"
        "<li>Euclidean metric. Times and memory reported as mean +/- std over the repeats.</li>"
        "<li>Peak RSS measured via <code>tracemalloc</code> (tracks Python allocations).</li>"
        "<li>Synthetic data: Gaussian mixture, min-max normalised to [0, 1].</li>"
        "</ul>"
    )

    return rep.save(out_path)


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    matplotlib.use("Agg")

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data", default=None, help="path to an (N, D) .npy file")
    ap.add_argument("--d", type=int, default=100, help="features for synthetic data")
    ap.add_argument(
        "--ns",
        type=int,
        nargs="+",
        default=[500, 1000, 2000, 4000, 8000],
        help="dataset sizes for N-scaling test",
    )
    ap.add_argument(
        "--eps-list",
        type=float,
        nargs="+",
        default=None,
        help="eps values for eps-scaling test (default: auto-calibrated)",
    )
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["greedy"],
        help="landmark methods to benchmark (default: greedy)",
    )
    ap.add_argument(
        "--scaling-eps",
        type=float,
        default=None,
        help="eps for N-scaling test (default: auto-calibrated)",
    )
    ap.add_argument(
        "--scaling-n",
        type=int,
        default=None,
        help="N for eps-scaling test (default: 2000)",
    )
    ap.add_argument("--reps", type=int, default=3, help="repetitions per timing")
    ap.add_argument("--out", default=".", help="output directory")
    args = ap.parse_args()

    # Load or generate reference data
    if args.data:
        ref_full = np.load(args.data).astype(np.float32)
        d = ref_full.shape[1]
        data_src = os.path.basename(args.data)
    else:
        d = args.d
        ref_full = gs.default_reference(n=max(args.ns), d=d)
        data_src = "synthetic"

    # Calibrate eps if not provided
    if args.scaling_eps is None:
        cal_eps = gs.calibrate_eps(ref_full[: min(4000, ref_full.shape[0])])
        scaling_eps = cal_eps[1]  # middle quantile
    else:
        scaling_eps = args.scaling_eps

    if args.eps_list is not None:
        eps_list = args.eps_list
    else:
        eps_list = gs.calibrate_eps(ref_full[: min(4000, ref_full.shape[0])])

    scaling_n = args.scaling_n or min(2000, max(args.ns))

    methods = args.methods

    print(
        f"benchmark_ballmapper | host={socket.gethostname().split('.')[0]} "
        f"reps={args.reps} d={d} data={data_src}",
        flush=True,
    )
    print(
        f"  methods={methods}  scaling_eps={scaling_eps:.4f}  "
        f"scaling_n={scaling_n:,}  eps_list={[f'{e:.4f}' for e in eps_list]}",
        flush=True,
    )

    print("\n=== Test 1: N-scaling ===", flush=True)
    n_rows = run_n_scaling(methods, args.ns, scaling_eps, args.reps, d)

    print("\n=== Test 2: eps-scaling ===", flush=True)
    eps_rows = run_eps_scaling(methods, eps_list, scaling_n, args.reps, d)

    meta = {
        "host": socket.gethostname().split(".")[0],
        "reps": args.reps,
        "d": d,
        "data": data_src,
        "methods": methods,
        "scaling_eps": scaling_eps,
        "scaling_n": scaling_n,
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
    }

    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "results.json")
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"meta": meta, "n_scaling": n_rows, "eps_scaling": eps_rows},
            fh,
            indent=1,
        )
    print(f"\nwrote {results_path}", flush=True)

    report_path = build_report(
        n_rows, eps_rows, meta, os.path.join(args.out, "report.html")
    )
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
