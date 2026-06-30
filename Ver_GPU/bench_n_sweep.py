"""bench_n_sweep.py — TEST 2: eps fixed, sweep N.

Compares original (pyballmapper) / BallTree / GPU-waveMIS as N grows at a fixed
eps, --reps each (mean ± std), with per-phase timings, lm/edge Jaccard vs
BallTree, and V1–V4 validity.  The naive `original` is pre-skipped for
N > --orig-max-n (O(N·L) infeasible at scale); every skip/cap is logged.

Outputs (./n_sweep/):
  repeat_results.json   raw per-rep + summary for every (N, method)
  timing_table.txt      mean ± std + GPU speed-up, per N
  validity_table.txt    L / E / valid / lm_jacc / edge_jacc, per N
  phase_table.txt       per-phase mean ± std
  nsweep_4way.png       time & speed-up vs N, 4 methods
  nsweep_3way.png       same, original excluded (finer scale)
  nsweep_diag.png       L vs N + Jaccard-vs-BallTree vs N

Usage:
  python3 bench_n_sweep.py --reps 10 --eps 0.08 \
      --ns 5000 10000 20000 50000 100000 200000 500000 1000000 --device 0
"""

from __future__ import annotations

import argparse
import os
import socket

import bench_common as bc
import gen_synthetic as gs
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--eps", type=float, default=0.08)
    ap.add_argument(
        "--ns",
        type=int,
        nargs="+",
        default=[5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000],
    )
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--per-rep-ceiling", type=float, default=300.0)
    ap.add_argument(
        "--cell-budget",
        type=float,
        default=1500.0,
        help="larger than Test 1: eps is fixed/moderate so full 10 "
        "reps fit even at N=1e6 (CPU ~108 s/rep)",
    )
    ap.add_argument(
        "--orig-max-n",
        type=int,
        default=200000,
        help="pre-skip the naive `original` for N > this",
    )
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--out",
        default="n_sweep",
        help="output subdirectory under the script dir (default: n_sweep)",
    )
    args = ap.parse_args()

    bc.select_methods()

    if args.smoke:
        args.ns = [2000, 4000]
        args.reps = 2
        args.per_rep_ceiling = 60.0
        args.cell_budget = 120.0
        args.orig_max_n = 10**9

    out = os.path.join(HERE, args.out)
    os.makedirs(out, exist_ok=True)
    host = socket.gethostname().split(".")[0]
    log = print

    log(
        f"bench_n_sweep | host={host} device={args.device} reps={args.reps} "
        f"eps={args.eps} ns={args.ns} per_rep_ceiling={args.per_rep_ceiling}s "
        f"cell_budget={args.cell_budget}s orig_max_n={args.orig_max_n} "
        f"validate={not args.no_validate} cores={bc.N_PHYSICAL_CORES} "
        f"methods={bc.NAMES}",
        flush=True,
    )
    if bc.RefBallMapper is None:
        log(
            f"  WARNING: pyballmapper unavailable ({bc._ORIG_ERR}) — 'original' will fail"
        )

    ref = gs.default_reference(HERE).astype(np.float32)
    bc.warmup(args.device, ref, args.eps, max(args.ns), log=log)

    json_path = os.path.join(out, "repeat_results.json")
    res = {
        "xs": list(args.ns),
        "reps": args.reps,
        "host": host,
        "fixed_str": f"eps={args.eps}, {host}",
        "title": f"N-sweep at eps={args.eps} ({host})",
        "cells": [],
        "test": "n_sweep",
        "device": args.device,
        "eps": args.eps,
        "guards": {
            "per_rep_ceiling": args.per_rep_ceiling,
            "cell_budget": args.cell_budget,
            "orig_max_n": args.orig_max_n,
        },
    }

    for N in args.ns:
        log("\n" + "=" * 88 + f"\nN = {N:,}  (eps={args.eps})\n" + "=" * 88, flush=True)
        X = gs.make_bootstrap(ref, N, seed=N).astype(np.float32)
        raw = {}
        for method in bc.NAMES:
            pre = method == "original" and N > args.orig_max_n
            reason = f"naive O(N·L) infeasible at N={N:,} (> {args.orig_max_n:,})"
            raw[method] = bc.run_cell(
                method,
                X,
                args.eps,
                args.device,
                args.reps,
                args.per_rep_ceiling,
                args.cell_budget,
                pre_skip=pre,
                skip_reason=reason,
                log=log,
            )
        ref_bm = raw.get("BallTree", {}).get("last")
        for method in bc.NAMES:
            rbm = None if method == bc.REF else ref_bm
            rec = bc.summarize_cell(
                method,
                N,
                N,
                args.eps,
                raw[method],
                rbm,
                None if args.no_validate else X,
            )
            res["cells"].append(rec)
            if rec["t_mean"] is not None:
                log(
                    f"  => {method:>12}  t={rec['t_mean']:.4f}±{rec['t_std']:.4f}s "
                    f"L={rec['L']} E={rec['E']} valid={rec['valid']} "
                    f"lm_jacc={rec['lm_jaccard']} ({rec['n_reps']} reps)",
                    flush=True,
                )
            bc.flush_json(res, json_path)

    bc.write_timing_table(res, "N", os.path.join(out, "timing_table.txt"))
    bc.write_validity_table(res, "N", os.path.join(out, "validity_table.txt"))
    bc.write_phase_table(res, "N", os.path.join(out, "phase_table.txt"))
    bc.plot_sweep(res, "N", True, os.path.join(out, "nsweep_4way.png"))
    bc.plot_sweep(res, "N", False, os.path.join(out, "nsweep_3way.png"))
    # median + IQR variants (robust to one-off cold-start GPU reps that inflate
    # the mean, e.g. the N=10k cuSPARSE first-call spike) — the recommended view.
    bc.plot_sweep(
        res, "N", True, os.path.join(out, "nsweep_4way_median.png"), central="median"
    )
    bc.plot_sweep(
        res, "N", False, os.path.join(out, "nsweep_3way_median.png"), central="median"
    )
    bc.plot_diagnostic(res, "N", os.path.join(out, "nsweep_diag.png"))
    bc.flush_json(res, json_path)
    log("\nALL DONE", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
