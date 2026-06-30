# Ver_BallTree — a fast, CPU `BallTree` Ball Mapper

`FastBallMapper` is a **drop-in replacement** for `pyballmapper.BallMapper` that
replaces the reference algorithm's serial `O(N·L)` distance loop with an sklearn
[`BallTree`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html)
(landmark selection + coverage) and a sparse incidence-matrix product
`M @ Mᵀ` for the edges. It produces a **mathematically identical graph** —
the same ordered landmark point-ids and the same edge set — while being many
times faster, with the gap widening as `N` grows.

This folder is self-contained: same public API (`eps`, `eps_dict`,
`points_covered_by_landmarks`, `Graph`, `add_coloring`, `filter_by`,
`points_and_balls`), no project-internal imports.

## Files

| file | purpose |
| --- | --- |
| `fast_ball_mapper.py` | the `FastBallMapper` class (BallTree + joblib) |
| `benchmark_balltree.py` | runs `original` vs `BallTree`, writes `report.html` |
| `html_report.py` | tiny self-contained HTML-report builder (numpy/matplotlib) |
| `gen_synthetic.py` | deterministic synthetic / bootstrap data generators |
| `tests/test_fast_ball_mapper.py` | correctness tests (identical to the reference) |
| `doc/index.html` | interactive write-up (English; `index.ko.html` Korean) |
| `wine_massspec_sample.npy` | optional 20000×100 example sample (see *Data*) |

## Quick start

```bash
pip install -r requirements.txt

python benchmark_balltree.py     # ~1–2 min on CPU -> report.html + results.json
pytest -q                        # correctness tests (vs pyballmapper)
```

Open `report.html` in any browser. It contains:

* **Test 1** — for each `eps`, both Mappers are built and checked to be
  *identical*, and the per-`eps` speed-up `t_original / t_balltree` is reported.
* **Test 2** — N-scaling at a fixed `eps`. The naive `original` is `O(N·L)` and
  becomes infeasible, so it is skipped above `--orig-max-n` (every skip is shown
  in the table, never silent); `BallTree` continues to the largest `N`.

Useful flags: `--reps`, `--sample`, `--eps-list`, `--ns`, `--orig-max-n`,
`--data path.npy`, `--out DIR`.

## Why the graphs are identical

`FastBallMapper` reproduces the reference greedy rule step for step: a point
becomes a landmark iff it is not already inside an existing landmark's ball
(visited in natural index order `0..N-1`); coverage lists every point within
`eps` of a landmark; and two balls are adjacent iff they share a covered point.
The edge step is a single C++-level SpGEMM (`M @ Mᵀ`, nonzero off-diagonal ⇒
shared point) instead of a per-pair Python loop. See `doc/index.html` for the
full write-up.

## Data source

The bundled `wine_massspec_sample.npy` is a 20000×100 `float32` sample of a
two-dimensional gas chromatography / high-resolution mass spectrometry
(GC×GC-HRMS) feature matrix of botrytized wines, min–max normalised per feature
to `[0, 1]`. It is published with:

> Koljančić, N.; Park, S. A.; Gurnari, D.; Dłotko, P.; Hahn, J.; Špánik, I.
> "Untargeted Chemical Profiling of Two-Dimensional Gas Chromatography Coupled
> with High-Resolution Mass Spectrometry Data for Botrytized Wines via
> Topological Data Analysis." *Analytical Chemistry* **2025**, *97*(46),
> 25853–25867. DOI: [10.1021/acs.analchem.5c05342](https://doi.org/10.1021/acs.analchem.5c05342)

The sample is bundled here (the folder's `.gitignore` re-includes it past the
repository-wide `*.npy` rule). If the file is ever absent, the benchmark
**automatically falls back to deterministic synthetic data**
(`gen_synthetic.make_highd`) and still runs end-to-end. Pass `--data path.npy`
to use any `(N, D)` point cloud.

## Requirements

`numpy`, `pandas`, `scikit-learn`, `joblib`, `networkx`, `scipy`, `matplotlib`,
and `pyballmapper` (for the reference comparison). `pytest` to run the tests.
