"""benchmark_gpu.py — GPU-waveMIS Ball Mapper benchmark + HTML report.

Compares three Ball Mappers under one interface:
  original    pyballmapper.BallMapper        (serial, naive O(N*L))
  BallTree    fast_ball_mapper.FastBallMapper (sklearn BallTree + joblib, CPU)
  GPU-waveMIS gpu_ball_mapper.GpuBallMapper   (parallel wavefront-MIS, GPU)

across an eps-sweep (N fixed: 100k and 1M) and an N-sweep (eps fixed: 5k -> 1M),
and writes a single self-contained ``report.html`` with both sweeps' tables and
plots.

Two modes
---------
default (no GPU needed)
    Render the report from the SAVED sweep results bundled under ``results/``
    (these were produced on the FJFI GPU servers; see doc/index.html). Only
    numpy + matplotlib are needed, so the report builds on any machine.

--live  (needs a CUDA GPU + torch)
    Re-run both sweeps here via bench_eps_sweep.py / bench_n_sweep.py, refresh
    ``results/``, then render.

The naive ``original`` is skipped on the large examples (its O(N*L) cost is
infeasible) — it appears as a dash in the tables, never silently dropped. FAISS
is intentionally not part of this comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from html_report import Report

HERE = os.path.dirname(os.path.abspath(__file__))

METHODS = ["original", "BallTree", "GPU-waveMIS"]
GPU = "GPU-waveMIS"
COLOR = {"original": "#7f7f7f", "BallTree": "#d62728", "GPU-waveMIS": "#1f77b4"}
MARKER = {"original": "P", "BallTree": "o", "GPU-waveMIS": "D"}


# ── result access ────────────────────────────────────────────────────────────
def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _x_eq(a, b):
    return abs(float(a) - float(b)) < 1e-9


def _cell(cells, x, method):
    for c in cells:
        if c["method"] == method and _x_eq(c["x"], x):
            return c
    return None


def _central(cell):
    """Robust central build time for a cell: median of the per-rep totals
    (insensitive to a cold first GPU rep), falling back to the recorded mean."""
    if not cell or cell.get("t_mean") is None:
        return None
    tot = cell.get("totals") or []
    return float(np.median(tot)) if tot else float(cell["t_mean"])


def _fmt_t(v):
    if v is None:
        return "—"
    return f"{v:.4f}" if v < 1 else f"{v:.2f}"


def _speedup(cell_other, cell_gpu):
    a, g = _central(cell_other), _central(cell_gpu)
    if a is None or not g:
        return "—"
    return f"<b>{a / g:.0f}&times;</b>"


def _valid_flag(cell):
    if not cell or cell.get("valid") is None:
        return "<span class='meta'>—</span>"
    return (
        "<span class='good'>OK</span>"
        if cell["valid"]
        else "<span class='bad'>INVALID</span>"
    )


# ── plotting ─────────────────────────────────────────────────────────────────
def _sweep_figure(res, xlabel, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = res["xs"]
    cells = res["cells"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.9))

    for m in METHODS:
        px, py = [], []
        for x in xs:
            v = _central(_cell(cells, x, m))
            if v is not None:
                px.append(x)
                py.append(v)
        if px:
            ax1.plot(px, py, MARKER[m] + "-", color=COLOR[m], lw=2, label=m)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("build time (s, median)")
    ax1.set_title(f"Build time vs {xlabel}")
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    ax1.legend()

    for m in METHODS:
        if m == GPU:
            continue
        px, py = [], []
        for x in xs:
            a = _central(_cell(cells, x, m))
            g = _central(_cell(cells, x, GPU))
            if a is not None and g:
                px.append(x)
                py.append(a / g)
        if px:
            ax2.plot(px, py, MARKER[m] + "-", color=COLOR[m], lw=2, label=f"{m} / GPU")
    ax2.axhline(1.0, color=COLOR[GPU], ls="--", alpha=0.6, label="GPU (1×)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("× slower than GPU-waveMIS")
    ax2.set_title("GPU-waveMIS speed-up")
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    ax2.legend()
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ── report sections ──────────────────────────────────────────────────────────
def _eps_section(rep, eps_json):
    rep.h2("Eps-sweep — N fixed, &epsilon; varied")
    by_n = eps_json.get("by_N", {})
    for N in sorted(by_n, key=int):
        res = by_n[N]
        rep.html(f"<h3 style='margin:22px 0 8px'>N = {int(N):,}</h3>")
        cells = res["cells"]
        rows = []
        for eps in res["xs"]:
            cg = _cell(cells, eps, GPU)
            cb = _cell(cells, eps, "BallTree")
            co = _cell(cells, eps, "original")
            L = (cg or cb or {}).get("L")
            rows.append(
                [
                    f"{eps:g}",
                    "—" if L is None else f"{L:,}",
                    _fmt_t(_central(co)),
                    _fmt_t(_central(cb)),
                    _fmt_t(_central(cg)),
                    _speedup(co, cg),
                    _speedup(cb, cg),
                    _valid_flag(cg),
                ]
            )
        rep.table(
            [
                "eps",
                "L",
                "original (s)",
                "BallTree (s)",
                "GPU (s)",
                "GPU vs orig",
                "GPU vs BallTree",
                "GPU valid",
            ],
            rows,
        )
        rep.figure(
            _sweep_figure(res, "eps", f"eps-sweep at N={int(N):,}"),
            caption=f"N={int(N):,}: build time and GPU-waveMIS speed-up vs &epsilon;.",
        )
    rep.callout(
        "At N = 1,000,000 the naive <code>original</code> is skipped at every "
        "&epsilon; (its O(N&middot;L) cost is infeasible), so its column is all "
        "dashes — the comparison there is GPU-waveMIS vs the BallTree CPU baseline."
    )


def _n_section(rep, n_json):
    rep.h2("N-sweep — &epsilon; fixed, N varied")
    res = n_json
    eps = res.get("eps", "?")
    rep.p(f"&epsilon; = <code>{eps}</code>, bootstrap-from-wine data.")
    cells = res["cells"]
    rows = []
    for N in res["xs"]:
        cg = _cell(cells, N, GPU)
        cb = _cell(cells, N, "BallTree")
        co = _cell(cells, N, "original")
        L = (cg or cb or {}).get("L")
        rows.append(
            [
                f"{int(N):,}",
                "—" if L is None else f"{L:,}",
                _fmt_t(_central(co)),
                _fmt_t(_central(cb)),
                _fmt_t(_central(cg)),
                _speedup(co, cg),
                _speedup(cb, cg),
                _valid_flag(cg),
            ]
        )
    rep.table(
        [
            "N",
            "L",
            "original (s)",
            "BallTree (s)",
            "GPU (s)",
            "GPU vs orig",
            "GPU vs BallTree",
            "GPU valid",
        ],
        rows,
    )
    rep.figure(
        _sweep_figure(res, "N", f"N-sweep at eps={eps}"),
        caption="Build time and GPU-waveMIS speed-up vs N. The original curve "
        "stops where it becomes infeasible.",
    )


def render(results_dir, out_path):
    eps_json = _load(os.path.join(results_dir, "eps_sweep", "repeat_results.json"))
    n_json = _load(os.path.join(results_dir, "n_sweep", "repeat_results.json"))

    host = n_json.get("host", eps_json.get("host", "?"))
    rep = Report(
        "GPU-waveMIS Ball Mapper — benchmark",
        subtitle=(
            f"three-way comparison (original / BallTree / GPU-waveMIS) · "
            f"results host={host} · rendered {time.strftime('%Y-%m-%d %H:%M')}"
        ),
    )
    rep.p(
        "<code>GpuBallMapper</code> replaces the CPU bottleneck — the sequential "
        "greedy landmark selection — with a parallel index-priority "
        "<b>wavefront maximal-independent-set</b> on the GPU, plus GEMM coverage "
        "and a cuSPARSE presence-only edge product. It is a drop-in replacement "
        "for <code>FastBallMapper</code> and produces a validated maximal "
        "&epsilon;-net (independently checked V1&ndash;V4)."
    )
    rep.callout(
        "Times are the <b>median</b> over repeats (robust to a one-off cold GPU "
        "rep). Validity is the independent V1&ndash;V4 check; <code>GPU valid = "
        "OK</code> means the GPU graph is a genuine Ball Mapper. FAISS is not "
        "part of this comparison."
    )
    _eps_section(rep, eps_json)
    _n_section(rep, n_json)
    rep.h2("Notes")
    rep.html(
        "<ul>"
        "<li>On the bundled data the GPU landmark set is in fact identical to "
        "the BallTree (CPU) result (lm/edge Jaccard = 1.0), so the comparison is "
        "pure wall-clock.</li>"
        "<li>GPU figures need a CUDA machine; this report renders from the saved "
        "<code>results/</code> by default. Re-run live with <code>--live</code> "
        "on a GPU host.</li>"
        "<li>For the math + an interactive CPU-vs-GPU graph viewer, see "
        "<a href='doc/index.html'>doc/index.html</a>.</li>"
        "</ul>"
    )
    rep.save(out_path)
    return out_path


# ── live re-run (GPU host only) ──────────────────────────────────────────────
def _have_cuda():
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _run_live(reps, device, results_dir):
    if not _have_cuda():
        raise SystemExit(
            "--live needs a CUDA GPU and torch. None detected. Omit --live to "
            "render the report from the bundled results/ instead."
        )
    eps_out = os.path.join("results", "eps_sweep")
    n_out = os.path.join("results", "n_sweep")
    print("[live] eps-sweep ...", flush=True)
    subprocess.run(
        [
            sys.executable,
            "bench_eps_sweep.py",
            "--reps",
            str(reps),
            "--device",
            str(device),
            "--out",
            eps_out,
        ],
        cwd=HERE,
        check=True,
    )
    print("[live] N-sweep ...", flush=True)
    subprocess.run(
        [
            sys.executable,
            "bench_n_sweep.py",
            "--reps",
            str(reps),
            "--device",
            str(device),
            "--out",
            n_out,
        ],
        cwd=HERE,
        check=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results",
        default=os.path.join(HERE, "results"),
        help="directory with eps_sweep/ and n_sweep/ result JSONs",
    )
    ap.add_argument("--out", default=os.path.join(HERE, "report.html"))
    ap.add_argument(
        "--live",
        action="store_true",
        help="re-run both sweeps on a CUDA GPU before rendering",
    )
    ap.add_argument("--reps", type=int, default=10, help="reps for --live")
    ap.add_argument("--device", type=int, default=0, help="CUDA device for --live")
    args = ap.parse_args()

    if args.live:
        _run_live(args.reps, args.device, args.results)
        args.results = os.path.join(HERE, "results")

    out = render(args.results, args.out)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
