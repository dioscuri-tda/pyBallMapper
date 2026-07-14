# Benchmarks

pyBallMapper ships with a benchmarking script that measures how `BallMapper` construction scales with dataset size, ball radius, and landmark method.

## Running the benchmark

Make sure you have the development dependencies installed:

```bash
uv sync
```

Then run the benchmark with:

```bash
uv run python benchmarks/benchmark_ballmapper.py
```

This generates two files in the current directory:

- `report.html` — self-contained HTML report with tables and embedded plots
- `results.json` — raw timing and memory data

## CLI options

| Flag | Default | Description |
|---|---|---|
| `--data PATH` | synthetic | Path to an `(N, D)` `.npy` point cloud |
| `--d INT` | 100 | Number of features for synthetic data |
| `--ns INT ...` | `500 1000 2000 4000 8000` | Dataset sizes for N-scaling test |
| `--eps-list FLOAT ...` | auto-calibrated | Eps values for eps-scaling test |
| `--methods STR ...` | `greedy` | Landmark methods to benchmark (`greedy`, `nearest`, `adaptive`) |
| `--scaling-eps FLOAT` | auto-calibrated | Fixed eps for the N-scaling test |
| `--scaling-n INT` | 2000 | Fixed N for the eps-scaling test |
| `--reps INT` | 3 | Repetitions per timing (mean +/- std is reported) |
| `--out DIR` | `.` | Output directory |

## What is measured

The benchmark runs two tests:

**Test 1 — N-scaling** builds BallMapper on datasets of increasing size at a fixed eps. This shows how construction time and memory grow with `N`.

**Test 2 — eps-scaling** builds BallMapper with decreasing eps at a fixed `N`. Smaller eps means more landmarks and more edges, so both time and memory increase.

Both tests report:

- **Build time** (seconds, mean +/- std over `--reps` runs)
- **Peak RSS** (MB, mean +/- std, measured via `tracemalloc`)
- **Number of landmarks** (`L`)
- **Number of edges** (`E`)
- **Raw per-run values** in the JSON output

## Examples

Compare greedy and nearest methods on larger data:

```bash
uv run python benchmarks/benchmark_ballmapper.py \
    --ns 1000 2000 4000 8000 \
    --methods greedy nearest \
    --reps 5
```

Use your own data:

```bash
uv run python benchmarks/benchmark_ballmapper.py --data my_data.npy
```

Test the adaptive method with a fixed eps:

```bash
uv run python benchmarks/benchmark_ballmapper.py \
    --methods adaptive \
    --scaling-eps 0.1 \
    --ns 500 1000 2000
```

## Output

The HTML report (`report.html`) contains:

- Tables with per-configuration timing, memory, landmarks, and edges
- Log-log plots of build time and peak memory vs N
- Log-log plots of build time and peak memory vs eps
- A plot of the number of landmarks vs N

The JSON file (`results.json`) contains mean, std, and raw per-run values in machine-readable form, suitable for CI integration or further analysis.
