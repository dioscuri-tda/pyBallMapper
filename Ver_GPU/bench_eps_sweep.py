"""bench_eps_sweep.py — TEST 1: N fixed, sweep eps (0.32 -> 0.02).

Goal (a): confirm "smaller eps -> larger GPU-waveMIS speed-up" holds at large N.
Goal (b): locate the break-even / degeneracy region — where L -> N, where fp32
          breaks the lm_jaccard=1.0 identity with BallTree, and where the GPU
          runs out of memory.

Compares original (pyballmapper) / BallTree / GPU-waveMIS, --reps each
(mean ± std), with per-phase timings, lm/edge Jaccard vs BallTree, and
V1–V4 validity.  The naive `original` is pre-skipped for N >= --orig-max-n
(O(N·L) infeasible); every skip/cap is logged.

Outputs (./eps_sweep/):
  repeat_results.json            raw per-rep + summary for every (N, eps, method)
  timing_table_N{N}.txt          mean ± std + GPU speed-up, per eps
  validity_table_N{N}.txt        L / E / valid / lm_jacc / edge_jacc, per eps
  phase_table_N{N}.txt           per-phase mean ± std
  eps_N{N}_4way.png              time & speed-up vs eps, 4 methods
  eps_N{N}_3way.png              same, original excluded (finer scale)
  eps_N{N}_diag.png              L vs eps + Jaccard-vs-BallTree vs eps

Usage:
  python3 bench_eps_sweep.py --reps 10 --ns 100000 1000000 \
      --eps-list 0.32 0.24 0.16 0.12 0.08 0.06 0.04 0.03 0.02 --device 0
"""

from __future__ import annotations

import argparse
import json
import os
import socket

import bench_common as bc
import gen_synthetic as gs
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _eps_match(a, b):
    return a is not None and b is not None and abs(float(a) - float(b)) < 1e-9


def _done_methods(res, eps):
    """Methods that already have a recorded cell at this eps (for --resume)."""
    return {c["method"] for c in res.get("cells", []) if _eps_match(c.get("x"), eps)}


def _put_cell(res, rec):
    """Insert rec, replacing any existing cell with the same (x, method) so a
    --resume re-run never leaves duplicates."""
    cells = res.setdefault("cells", [])
    for i, c in enumerate(cells):
        if c.get("method") == rec["method"] and _eps_match(c.get("x"), rec["x"]):
            cells[i] = rec
            return
    cells.append(rec)


def _log_cell(log, method, rec):
    if rec["t_mean"] is not None:
        log(
            f"    => {method:>12}  t={rec['t_mean']:.4f}±{rec['t_std']:.4f}s "
            f"L={rec['L']} E={rec['E']} valid={rec['valid']} "
            f"lm_jacc={rec['lm_jaccard']} ({rec['n_reps']} reps)",
            flush=True,
        )
    else:
        log(
            f"    => {method:>12}  (no timing: {rec.get('note') or 'skipped/failed'})",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--ns", type=int, nargs="+", default=[100000, 1000000])
    ap.add_argument(
        "--eps-list",
        type=float,
        nargs="+",
        default=[0.32, 0.24, 0.16, 0.12, 0.08, 0.06, 0.04, 0.03, 0.02],
    )
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument(
        "--per-rep-ceiling",
        type=float,
        default=300.0,
        help="skip remaining reps if one rep exceeds this (s)",
    )
    ap.add_argument(
        "--cell-budget",
        type=float,
        default=900.0,
        help="stop a (method, eps) cell after this cumulative time (s)",
    )
    ap.add_argument(
        "--orig-max-n",
        type=int,
        default=200000,
        help="pre-skip the naive `original` for N >= this",
    )
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny preflight run (N=2000, 2 eps, 2 reps)",
    )
    ap.add_argument(
        "--orig-max-l",
        type=int,
        default=40000,
        help="skip the naive 'original' once the BallTree landmark "
        "count L exceeds this — the L->N degeneracy region where "
        "the naive float64 build segfaults (rc=139)",
    )
    ap.add_argument(
        "--robust-max-l",
        type=int,
        default=10**9,
        help="skip the CPU baseline (BallTree) when the GPU "
        "landmark count L exceeds this — the L->N degeneracy "
        "region where they take HOURS at large N while the GPU "
        "takes seconds (GPU is still measured). Default: off.",
    )
    ap.add_argument(
        "--no-isolate-original",
        action="store_true",
        help="run 'original' in-process (default: isolate it in a "
        "subprocess so a native segfault can't kill the sweep)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="load existing <out>/repeat_results.json and skip "
        "already-recorded (N, eps, method) cells",
    )
    ap.add_argument(
        "--out",
        default="eps_sweep",
        help="output subdirectory under the script dir (default: eps_sweep)",
    )
    args = ap.parse_args()

    bc.select_methods()

    if args.smoke:
        args.ns = [2000]
        args.eps_list = [0.16, 0.08]
        args.reps = 2
        args.per_rep_ceiling = 60.0
        args.cell_budget = 120.0
        args.orig_max_n = 10**9  # let original run in the smoke test
        args.orig_max_l = 10**9  # ... and never trip the L threshold
        if args.out == "eps_sweep":
            args.out = "eps_sweep_smoke"  # never clobber the real results

    out = os.path.join(HERE, args.out)
    os.makedirs(out, exist_ok=True)
    host = socket.gethostname().split(".")[0]
    log = print

    log(
        f"bench_eps_sweep | host={host} device={args.device} reps={args.reps} "
        f"ns={args.ns} eps={args.eps_list} per_rep_ceiling={args.per_rep_ceiling}s "
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
    bc.warmup(args.device, ref, max(args.eps_list), max(args.ns), log=log)

    json_path = os.path.join(out, "repeat_results.json")
    results_all = None
    if args.resume and os.path.exists(json_path):
        try:
            with open(json_path) as fh:
                results_all = json.load(fh)
            n_prior = sum(
                len(v.get("cells", [])) for v in results_all.get("by_N", {}).values()
            )
            log(
                f"  RESUME: loaded {json_path} ({n_prior} prior cells) — "
                f"already-recorded (N, eps, method) cells will be skipped"
            )
        except Exception as ex:
            log(
                f"  RESUME: could not read {json_path} "
                f"({type(ex).__name__}: {ex}); starting fresh"
            )
            results_all = None
    if results_all is None:
        results_all = {
            "test": "eps_sweep",
            "host": host,
            "device": args.device,
            "reps": args.reps,
            "eps_list": args.eps_list,
            "ns": args.ns,
            "guards": {
                "per_rep_ceiling": args.per_rep_ceiling,
                "cell_budget": args.cell_budget,
                "orig_max_n": args.orig_max_n,
                "orig_max_l": args.orig_max_l,
            },
            "by_N": {},
        }
    results_all.setdefault("by_N", {})

    robust = [m for m in bc.NAMES if m != "original"]
    has_orig = "original" in bc.NAMES

    for N in args.ns:
        log("\n" + "=" * 88 + f"\nN = {N:,}\n" + "=" * 88, flush=True)
        res = results_all["by_N"].get(str(N)) or {
            "xs": list(args.eps_list),
            "reps": args.reps,
            "host": host,
            "fixed_str": f"N={N:,}, {host}",
            "title": f"eps-sweep at N={N:,} ({host})",
            "cells": [],
        }
        res["xs"] = list(args.eps_list)  # refresh axis on resume
        results_all["by_N"][str(N)] = res
        X = None  # generated lazily, only if a cell in this N actually runs

        for eps in args.eps_list:
            done = _done_methods(res, eps)
            if all(m in done for m in bc.NAMES):
                log(
                    f"\n  --- eps = {eps} (N={N:,}) --- "
                    f"RESUME-SKIP (all methods recorded)",
                    flush=True,
                )
                continue
            log(f"\n  --- eps = {eps} (N={N:,}) ---", flush=True)
            if X is None:
                X = gs.make_bootstrap(ref, N, seed=N).astype(np.float32)
            Xv = None if args.no_validate else X
            raw = {}

            # (1) robust methods, GPU FIRST so its landmark count L can gate the
            #     expensive CPU baselines: at the L->N degeneracy (e.g. N=1M,
            #     eps<=0.04) BallTree takes HOURS while the GPU takes seconds,
            #     so skip them once L exceeds --robust-max-l (GPU still measured).
            gpu = bc.GPU
            run_order = (
                ([gpu] + [m for m in robust if m != gpu]) if gpu in robust else robust
            )
            L_gpu = None
            for method in run_order:
                if method in done:
                    log(f"      [{method:>11}] RESUME-SKIP (already recorded)")
                    continue
                if method != gpu and L_gpu is not None and L_gpu > args.robust_max_l:
                    reason = (
                        f"L={L_gpu} > robust_max_l={args.robust_max_l:,} "
                        f"(L->N degeneracy: CPU baseline takes hours at "
                        f"N={N:,})"
                    )
                    raw[method] = bc.run_cell(
                        method,
                        X,
                        eps,
                        args.device,
                        args.reps,
                        args.per_rep_ceiling,
                        args.cell_budget,
                        pre_skip=True,
                        skip_reason=reason,
                        log=log,
                    )
                    continue
                raw[method] = bc.run_cell(
                    method,
                    X,
                    eps,
                    args.device,
                    args.reps,
                    args.per_rep_ceiling,
                    args.cell_budget,
                    log=log,
                )
                if method == gpu:
                    glast = (raw[method] or {}).get("last")
                    L_gpu = bc._n_land(glast) if glast is not None else None
            ref_bm = (raw.get("BallTree") or {}).get("last")

            # (2) summarize + flush the robust methods NOW, before the risky
            #     naive 'original' — so even if it hard-crashes the process,
            #     these results are already on disk.
            for method in robust:
                if method not in raw:
                    continue
                rbm = None if method == bc.REF else ref_bm
                rec = bc.summarize_cell(method, eps, N, eps, raw[method], rbm, Xv)
                _put_cell(res, rec)
                _log_cell(log, method, rec)
                results_all["by_N"][str(N)] = res
                bc.flush_json(results_all, json_path)

            # (3) naive 'original' LAST, guarded + isolated.  Pre-skip it past
            #     the L->N degeneracy threshold (where it segfaults); otherwise
            #     run it in a subprocess so any crash is contained.
            if has_orig and "original" not in done:
                L_bt = bc._n_land(ref_bm) if ref_bm is not None else None
                if N >= args.orig_max_n:
                    pre = True
                    reason = (
                        f"naive O(N·L) infeasible at N={N:,} "
                        f"(>= {args.orig_max_n:,})"
                    )
                elif L_bt is None:
                    pre = True
                    reason = (
                        "BallTree reference unavailable — cannot verify L "
                        "is below the segfault threshold"
                    )
                elif L_bt > args.orig_max_l:
                    pre = True
                    reason = (
                        f"L={L_bt} > orig_max_l={args.orig_max_l:,} "
                        f"(L->N degeneracy: naive float64 build segfaults)"
                    )
                else:
                    pre = False
                    reason = ""
                runner = (
                    bc.run_cell if args.no_isolate_original else bc.run_cell_isolated
                )
                raw["original"] = runner(
                    "original",
                    X,
                    eps,
                    args.device,
                    args.reps,
                    args.per_rep_ceiling,
                    args.cell_budget,
                    pre_skip=pre,
                    skip_reason=reason,
                    log=log,
                )
                rec = bc.summarize_cell(
                    "original", eps, N, eps, raw["original"], ref_bm, Xv
                )
                _put_cell(res, rec)
                _log_cell(log, "original", rec)
                results_all["by_N"][str(N)] = res
                bc.flush_json(results_all, json_path)

        # per-N outputs (regenerated from the full, merged res)
        bc.write_timing_table(res, "eps", os.path.join(out, f"timing_table_N{N}.txt"))
        bc.write_validity_table(
            res, "eps", os.path.join(out, f"validity_table_N{N}.txt")
        )
        bc.write_phase_table(res, "eps", os.path.join(out, f"phase_table_N{N}.txt"))
        bc.plot_sweep(res, "eps", True, os.path.join(out, f"eps_N{N}_4way.png"))
        bc.plot_sweep(res, "eps", False, os.path.join(out, f"eps_N{N}_3way.png"))
        bc.plot_diagnostic(res, "eps", os.path.join(out, f"eps_N{N}_diag.png"))

    bc.flush_json(results_all, json_path)
    log("\nALL DONE", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
