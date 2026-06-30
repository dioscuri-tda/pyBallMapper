"""benchmark_balltree.py — compare the BallTree CPU Ball Mapper
(``fast_ball_mapper.FastBallMapper``) against the reference
``pyballmapper.BallMapper`` and emit a single self-contained HTML report.

What it does
------------
Test 1 — correctness + per-eps speed-up
    On a subsample of the bundled wine GC x GC-HRMS sample, for each eps it
    builds both Mappers and checks they are IDENTICAL (same ordered landmark
    point-ids AND the same edge set, compared by landmark point-id), then
    reports the speed-up ``t_original / t_balltree``.

Test 2 — N-scaling
    On bootstrap-from-wine data of growing N at a fixed eps, it times both
    methods.  The naive ``original`` is O(N * L) and becomes infeasible, so it
    is skipped above ``--orig-max-n`` (every skip is shown in the table, never
    silent); BallTree continues to the largest N.

Outputs (default ``--out .``):
    report.html      single self-contained report (tables + embedded plots)
    results.json     raw per-cell timings / sizes / identity flags

Usage
-----
    python benchmark_balltree.py                 # fast defaults (~1-2 min, CPU)
    python benchmark_balltree.py --reps 5 --sample 5000 \\
        --ns 5000 10000 20000 50000 --orig-max-n 20000
"""

from __future__ import annotations

import argparse
import json
import os
import time

import gen_synthetic as gs
import numpy as np
from fast_ball_mapper import N_PHYSICAL_CORES, FastBallMapper
from html_report import Report

try:
    from pyballmapper import BallMapper as RefBallMapper

    _ORIG_ERR = None
except Exception as _e:  # pragma: no cover - depends on the environment
    RefBallMapper = None
    _ORIG_ERR = _e

HERE = os.path.dirname(os.path.abspath(__file__))


# ── graph accessors (uniform across both backends) ──────────────────────────
def _landmark_ids(bm):
    return [int(bm.Graph.nodes[n]["landmark"]) for n in bm.Graph.nodes]


def _edge_pid_set(bm):
    pid = {n: int(bm.Graph.nodes[n]["landmark"]) for n in bm.Graph.nodes}
    return set(frozenset((pid[u], pid[v])) for u, v in bm.Graph.edges)


def _build(method, X, eps):
    if method == "original":
        if RefBallMapper is None:
            raise RuntimeError(f"pyballmapper unavailable: {_ORIG_ERR}")
        # float64 so the naive reference is the canonical exact net.
        return RefBallMapper(
            X=np.ascontiguousarray(X, dtype=np.float64), eps=float(eps)
        )
    if method == "BallTree":
        return FastBallMapper(X=X, eps=eps, n_jobs=N_PHYSICAL_CORES)
    raise ValueError(method)


def _timed(method, X, eps, reps):
    """Return (median_time, last_object).  Median is robust to a cold rep."""
    ts, obj = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        obj = _build(method, X, eps)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), obj


def _identical(ref_bm, bm):
    """Exact identity: same ordered landmark point-ids AND same edge set."""
    same_lm = _landmark_ids(ref_bm) == _landmark_ids(bm)
    er, eg = _edge_pid_set(ref_bm), _edge_pid_set(bm)
    return bool(same_lm and er == eg), (len(er - eg), len(eg - er))


# ── tests ───────────────────────────────────────────────────────────────────
def run_correctness(ref_X, eps_list, reps, log=print):
    rows = []
    for eps in eps_list:
        log(f"  [Test 1] eps={eps} ...", flush=True)
        t_o, bm_o = _timed("original", ref_X, eps, reps)
        t_b, bm_b = _timed("BallTree", ref_X, eps, reps)
        same, (only_ref, only_bt) = _identical(bm_o, bm_b)
        rows.append(
            {
                "eps": eps,
                "L": int(bm_o.Graph.number_of_nodes()),
                "E": int(bm_o.Graph.number_of_edges()),
                "t_original": t_o,
                "t_balltree": t_b,
                "speedup": (t_o / t_b) if t_b else None,
                "identical": same,
                "edges_only_ref": only_ref,
                "edges_only_balltree": only_bt,
            }
        )
        log(
            f"     L={rows[-1]['L']} E={rows[-1]['E']} "
            f"t_orig={t_o:.3f}s t_bt={t_b:.3f}s "
            f"speedup={rows[-1]['speedup']:.1f}x identical={same}",
            flush=True,
        )
    return rows


def run_scaling(ref_X, ns, eps, reps, orig_max_n, log=print):
    rows = []
    for n in ns:
        X = gs.make_bootstrap(ref_X, n, seed=n).astype(np.float32)
        log(f"  [Test 2] N={n} eps={eps} ...", flush=True)
        t_b, bm_b = _timed("BallTree", X, eps, reps)
        if n <= orig_max_n:
            t_o, bm_o = _timed("original", X, eps, reps)
            same, _ = _identical(bm_o, bm_b)
        else:
            t_o, same = None, None
        rows.append(
            {
                "N": n,
                "L": int(bm_b.Graph.number_of_nodes()),
                "E": int(bm_b.Graph.number_of_edges()),
                "t_original": t_o,
                "t_balltree": t_b,
                "speedup": (t_o / t_b) if (t_o and t_b) else None,
                "identical": same,
            }
        )
        msg = f"     L={rows[-1]['L']} t_bt={t_b:.3f}s"
        msg += (
            f" t_orig={t_o:.3f}s speedup={rows[-1]['speedup']:.1f}x"
            if t_o
            else (f" (original skipped: N>{orig_max_n})")
        )
        log(msg, flush=True)
    return rows


# ── plots + report ───────────────────────────────────────────────────────────
def _scaling_figures(scaling):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [r["N"] for r in scaling]
    bt = [r["t_balltree"] for r in scaling]
    on = [(r["N"], r["t_original"]) for r in scaling if r["t_original"] is not None]
    sp = [(r["N"], r["speedup"]) for r in scaling if r["speedup"] is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax1.plot(ns, bt, "o-", color="#d62728", lw=2, label="BallTree (FastBallMapper)")
    if on:
        ax1.plot(
            [a for a, _ in on],
            [b for _, b in on],
            "P-",
            color="#7f7f7f",
            lw=2,
            label="original (pyballmapper)",
        )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("N (points)")
    ax1.set_ylabel("build time (s)")
    ax1.set_title("Build time vs N (log-log)")
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    ax1.legend()

    if sp:
        ax2.plot([a for a, _ in sp], [b for _, b in sp], "D-", color="#2563eb", lw=2)
    ax2.set_xscale("log")
    ax2.set_xlabel("N (points)")
    ax2.set_ylabel("speed-up  (t_original / t_balltree)")
    ax2.set_title("BallTree speed-up over the naive original")
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    return fig


def _flag(ok):
    if ok is None:
        return "<span class='meta'>—</span>"
    return "<span class='good'>yes</span>" if ok else "<span class='bad'>NO</span>"


def build_report(correctness, scaling, meta, out_path):
    rep = Report(
        "BallTree Ball Mapper — benchmark vs pyballmapper",
        subtitle=(
            f"host={meta['host']} · cores={meta['cores']} · "
            f"reps={meta['reps']} (median) · {meta['timestamp']}"
        ),
    )
    rep.p(
        "<code>FastBallMapper</code> replaces the reference algorithm's serial "
        "O(N&middot;L) distance loop with an sklearn <code>BallTree</code> "
        "(landmark selection &amp; coverage) and a sparse incidence-matrix "
        "product <code>M&nbsp;@&nbsp;M<sup>T</sup></code> for edges. It is a "
        "drop-in replacement: same public API and a "
        "<b>mathematically identical graph</b>."
    )

    all_identical = all(r["identical"] for r in correctness)
    rep.callout(
        (
            (
                "All tested &epsilon; produce a graph <b>identical</b> to "
                "<code>pyballmapper.BallMapper</code> (same landmark point-ids and "
                "same edge set) &mdash; the speed-up is free of any accuracy cost."
            )
            if all_identical
            else "Some &epsilon; produced a non-identical graph &mdash; see the table."
        ),
        kind="good" if all_identical else "warn",
    )

    data_desc = (
        "the bundled wine GC&times;GC-HRMS sample"
        if "wine" in meta["data"]
        else f"<code>{meta['data']}</code>"
    )
    rep.h2("Test 1 — correctness + per-&epsilon; speed-up")
    rep.p(
        f"Subsample of <code>{meta['sample']}</code> points "
        f"({meta['dims']} features) from {data_desc}."
    )
    rows = []
    for r in correctness:
        rows.append(
            [
                f"{r['eps']:.2f}",
                str(r["L"]),
                str(r["E"]),
                f"{r['t_original']:.3f}",
                f"{r['t_balltree']:.3f}",
                f"<b>{r['speedup']:.1f}&times;</b>" if r["speedup"] else "—",
                _flag(r["identical"]),
            ]
        )
    rep.table(
        ["eps", "L", "E", "t_original (s)", "t_balltree (s)", "speed-up", "identical"],
        rows,
    )

    rep.h2("Test 2 — N-scaling at fixed &epsilon;")
    rep.p(
        f"Bootstrap-from-wine data, &epsilon; = <code>{meta['scaling_eps']}</code>. "
        f"The naive <code>original</code> is skipped above N = "
        f"<code>{meta['orig_max_n']}</code> (its O(N&middot;L) cost is infeasible)."
    )
    rows = []
    for r in scaling:
        rows.append(
            [
                f"{r['N']:,}",
                str(r["L"]),
                str(r["E"]),
                f"{r['t_original']:.3f}" if r["t_original"] is not None else "—",
                f"{r['t_balltree']:.3f}",
                f"<b>{r['speedup']:.1f}&times;</b>" if r["speedup"] else "—",
                _flag(r["identical"]),
            ]
        )
    rep.table(
        ["N", "L", "E", "t_original (s)", "t_balltree (s)", "speed-up", "identical"],
        rows,
    )
    rep.figure(
        _scaling_figures(scaling),
        caption="Left: build time vs N (log-log). Right: BallTree speed-up over "
        "the naive original. The original curve stops where it becomes infeasible.",
    )

    rep.h2("Notes")
    rep.html(
        "<ul>"
        "<li>Euclidean metric. The landmark order is the natural index order "
        "<code>0..N-1</code>, matching the reference's greedy rule exactly.</li>"
        "<li>Times are the median of the repeats (robust to a cold first rep). "
        "BallTree coverage / edge phases release the GIL and run on all "
        f"{meta['cores']} physical cores.</li>"
        "<li>For a richer interactive write-up see "
        "<a href='doc/index.html'>doc/index.html</a>.</li>"
        "</ul>"
    )
    rep.save(out_path)
    return out_path


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    import socket

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sample",
        type=int,
        default=4000,
        help="wine subsample size for Test 1 (default 4000)",
    )
    ap.add_argument("--eps-list", type=float, nargs="+", default=[0.04, 0.08, 0.16])
    ap.add_argument(
        "--ns", type=int, nargs="+", default=[2000, 4000, 8000, 16000, 32000]
    )
    ap.add_argument("--scaling-eps", type=float, default=0.08)
    ap.add_argument(
        "--orig-max-n",
        type=int,
        default=8000,
        help="skip the naive original for N above this (default 8000)",
    )
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument(
        "--data",
        default=None,
        help="path to an (N, D) .npy point cloud; default: the "
        "bundled wine sample if present, else synthetic data",
    )
    ap.add_argument("--out", default=HERE, help="output directory")
    args = ap.parse_args()

    if RefBallMapper is None:
        raise SystemExit(f"pyballmapper is required for the comparison: {_ORIG_ERR}")

    # Use the bundled wine sample if present, else deterministic synthetic data
    # (the repo .gitignore excludes *.npy, so a fresh clone has no data file).
    if args.data:
        ref_full = np.load(args.data).astype(np.float32)
        data_src = os.path.basename(args.data)
    else:
        ref_full = gs.default_reference(HERE, n=max(20000, max(args.ns)))
        wine = os.path.join(HERE, "wine_massspec_sample.npy")
        data_src = "wine_massspec_sample.npy" if os.path.exists(wine) else "synthetic"
    sample = min(args.sample, ref_full.shape[0])
    ref_X = ref_full[:sample]

    print(
        f"benchmark_balltree | host={socket.gethostname()} "
        f"cores={N_PHYSICAL_CORES} reps={args.reps} sample={sample} "
        f"data={data_src} eps={args.eps_list} ns={args.ns}",
        flush=True,
    )

    correctness = run_correctness(ref_X, args.eps_list, args.reps)
    scaling = run_scaling(
        ref_full, args.ns, args.scaling_eps, args.reps, args.orig_max_n
    )

    meta = {
        "host": socket.gethostname().split(".")[0],
        "cores": N_PHYSICAL_CORES,
        "reps": args.reps,
        "sample": sample,
        "dims": int(ref_X.shape[1]),
        "data": data_src,
        "scaling_eps": args.scaling_eps,
        "orig_max_n": args.orig_max_n,
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {"meta": meta, "correctness": correctness, "scaling": scaling}, fh, indent=1
        )
    out = build_report(
        correctness, scaling, meta, os.path.join(args.out, "report.html")
    )
    print(f"\nwrote {out}\nwrote {os.path.join(args.out, 'results.json')}", flush=True)


if __name__ == "__main__":
    main()
