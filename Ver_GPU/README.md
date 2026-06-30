# Ver_GPU — a fast, validated GPU Ball Mapper (wavefront-MIS)

`GpuBallMapper` is a **drop-in replacement** for `FastBallMapper` (and so for
`pyballmapper.BallMapper`) that removes the CPU bottleneck — the *sequential*
greedy landmark selection — by recognising it as a lexicographically-first
**maximal independent set** on the ε-proximity graph and computing it in
parallel as an **index-priority wavefront MIS** on the GPU, with GEMM coverage
and a cuSPARSE presence-only edge product. It scales to **N = 1,000,000** and
produces a **validated maximal ε-net** every time (independent V1–V4 checks).

This folder is self-contained and the comparison here is strictly three-way —
`original` (pyballmapper) / `BallTree` (CPU) / `GPU-waveMIS`. **FAISS is not
part of this comparison.**

## Files

| file | purpose |
| --- | --- |
| `gpu_ball_mapper.py` | the `GpuBallMapper` class (PyTorch + cupy/cuSPARSE) |
| `fast_ball_mapper.py` | the `BallTree` CPU baseline / Jaccard reference |
| `bm_validate.py` | independent validity checker (V1–V4) + large-N sampled check |
| `gen_synthetic.py` | bootstrap-from-reference + synthetic blob generators |
| `bench_common.py` | shared 3-way benchmark harness (timing, validity, plots) |
| `bench_eps_sweep.py` | Test 1: N fixed, sweep ε (live, needs a GPU) |
| `bench_n_sweep.py` | Test 2: ε fixed, sweep N (live, needs a GPU) |
| `benchmark_gpu.py` | entry point: render `report.html` (saved or `--live`) |
| `html_report.py` | tiny self-contained HTML-report builder |
| `results/` | **saved** sweep results, so the report builds with no GPU |
| `tests/test_gpu_ball_mapper.py` | validity + render tests (GPU tests auto-skip) |
| `doc/index.html` | math + interactive CPU-vs-GPU viewer (English; `index.ko.html`) |

## Quick start

### Render the report (no GPU needed)

```bash
pip install numpy matplotlib          # all that the renderer needs
python benchmark_gpu.py               # -> report.html from results/
```

`report.html` shows both sweeps — **eps-sweep** (N = 100k and 1M) and
**N-sweep** (5k → 1M) — as 3-way tables and plots, with the GPU-waveMIS
speed-up over each baseline. The bundled results were produced on the FJFI GPU
servers (RTX 5060 Ti / RTX 3060).

### Reproduce live on a CUDA machine

```bash
pip install -r requirements.txt       # adds torch (+ optional cupy)
python benchmark_gpu.py --live --reps 10 --device 0
# or run a single sweep driver directly:
python bench_eps_sweep.py --reps 10 --ns 100000 1000000 --device 0
python bench_n_sweep.py   --reps 10 --eps 0.08 --device 0
```

```bash
pytest -q     # render + validity tests on CPU; GPU build tests run if CUDA
```

## "Too-large examples → the original is skipped"

The naive `original` (`pyballmapper`) is `O(N·L)` and infeasible at scale, so it
is **pre-skipped** on the large examples — at **N = 1,000,000 it is skipped at
every ε**, and in the N-sweep it is skipped above `--orig-max-n`. Skipped cells
appear as a dash in the tables (never silently dropped). At the deepest ε on the
1M example the `BallTree` baseline is skipped too (it would take hours), and only
the GPU is measured.

## Validity

Every GPU result is an independently-checked maximal ε-net:
V1 independence, V2 maximality, V3 coverage-exact, V4 edges-exact
(`bm_validate.py`, sklearn `BallTree` ground truth ≤ 200k, BallTree-free sampled
check beyond). On the bundled data the GPU landmark set is in fact *identical* to
the CPU result (lm/edge Jaccard = 1.0), so the comparison is pure wall-clock.

## Data source

The bundled `wine_massspec_sample.npy` is a 20000×100 `float32` sample of a
GC×GC-HRMS feature matrix of botrytized wines (min–max normalised per feature),
published with:

> Koljančić, N.; Park, S. A.; Gurnari, D.; Dłotko, P.; Hahn, J.; Špánik, I.
> "Untargeted Chemical Profiling of Two-Dimensional Gas Chromatography Coupled
> with High-Resolution Mass Spectrometry Data for Botrytized Wines via
> Topological Data Analysis." *Analytical Chemistry* **2025**, *97*(46),
> 25853–25867. DOI: [10.1021/acs.analchem.5c05342](https://doi.org/10.1021/acs.analchem.5c05342)

The sample is bundled here (the folder's `.gitignore` re-includes it past the
repository-wide `*.npy` rule). The live drivers fall back to deterministic
synthetic data if it is ever absent. The **saved** `results/` are independent of
the data file, so `report.html` always builds.

## Requirements

Render-only: `numpy`, `matplotlib`. Live runs additionally need `torch`
(CUDA build) and the CPU stack (`pandas`, `scikit-learn`, `joblib`, `networkx`,
`scipy`, `pyballmapper`); `cupy` is optional (faster post-processing).
