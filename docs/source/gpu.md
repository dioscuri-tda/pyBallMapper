# GPU landmark selection

`method="gpu"` runs the landmark search on a CUDA device through
[PyTorch](https://pytorch.org). On large point clouds it is the fastest option
pyBallMapper offers, by a wide margin.

```python
from pyballmapper import BallMapper

bm = BallMapper(X=my_pointcloud, eps=0.08, method="gpu")
```

## Requirements

PyTorch is **not** a dependency of pyBallMapper — it is imported lazily, only
when `method="gpu"` is actually used. Install a CUDA build separately, e.g.

```bash
pip install torch
```

If PyTorch is missing, or if `torch.cuda.is_available()` is `False`, the method
emits a warning and falls back to `method="balltree"`. It is therefore safe to
leave `method="gpu"` in code that also has to run on CPU-only machines.

Only the Euclidean metric is supported and `orbits` are not handled; for any
other `metric`, or when `orbits` are given, the method falls back to the default
greedy search instead.

The data must be finite and representable as `float32`. Input containing `NaN`,
infinity, or a magnitude that overflows the cast is rejected with a `ValueError`
rather than quietly producing a graph in which every point is its own ball.

## The same graph, computed in parallel

The greedy rule — walk the points in order, keep one iff it is not already
covered — is not merely *a* way to build a cover. It is exactly the
**lexicographically-first maximal independent set** of the ε-proximity graph.
The "maximal independent set" half is what makes the output a valid BallMapper
cover; the "lexicographically-first" half is what forces it to run one point at
a time.

The GPU method computes that same set, in parallel rounds. In each round, every
still-active point that has no active lower-indexed neighbour is selected at
once — two points within ε can never both win, because the higher-indexed one
sees the lower one as a neighbour. Streaming the points in chunks does not
change the answer either, since each chunk is first tested against every
landmark already accumulated.

So `method="gpu"` is expected to return **the same landmarks and the same edges
as the default greedy method**, and measurement agrees: identical landmark
sequence and identical edge set at every configuration tested, from
`n_samples = 5,000` to `1,000,000` and ε from `0.24` down to `0.02`, across 1 to
245 streaming chunks.

That said, equality is *expected*, not *guaranteed* — and the gap is arithmetic,
not algorithmic. Distances here are resolved in `float32` (see below), so a
point sitting within rounding of the ε boundary can land on the other side of it
than the CPU's `float64` test would put it. When that happens the result is a
different maximal ε-net: still valid — landmarks pairwise farther apart than ε,
every point covered — just not the greedy one. If you need the greedy graph with
a hard guarantee, use `method="balltree"`, which reproduces it exactly and is
still much faster than the default.

## How it works

The build runs in three passes:

1. **selection** — *on the device*. Points are streamed in chunks. Each chunk is
   tested against every landmark accumulated so far, which drops the
   already-covered rows; the survivors are then resolved against one another by
   the parallel independent-set step described above.
2. **coverage** — *on the device*. Every point is tested against the **final**
   landmark set. This is recomputed from scratch rather than reused from pass 1,
   where the coverage mask is only partial.
3. **grouping** — *on the host*. The (landmark, point) incidence pairs are
   sorted into per-landmark coverage lists.

Edges are then found by the same sparse incidence product `method="balltree"`
uses, on the host: two balls are adjacent iff they share at least one covered
point.

Passes 1 and 2 put the same (point, landmark) pair through matrix products of
*different shapes*, and BLAS does not promise the same float32 result for
different shapes — the kernel and the accumulation order change with them. For a
pair whose true distance sits within a few ulps of ε the two passes can
therefore disagree: pass 1 calls the point covered, so it never becomes a
landmark, and pass 2 then leaves it out of that ball. Left alone, such a point
would belong to no ball at all and would vanish from the cover without a word.
Pass 2 therefore also tracks each point's nearest landmark, and attaches any
point it orphaned to it — the point's true distance is within ε, which is why
pass 1 dropped it, so this restores maximality without weakening independence.

## Numerics

Distances use the Gram identity `‖a − b‖² = ‖a‖² + ‖b‖² − 2⟨a, b⟩` on squared
distances compared against `eps²`, so no square roots are taken.

The data is mean-centred **in float64 on the host, before the float32 cast**.
The order matters: the cast is the lossy step and it is irreversible. It
quantises every coordinate onto the float32 grid at the data's own offset
magnitude — for coordinates around `5·10⁶` that grid step is about `0.5`, which
is larger than most useful ε — and no later centring can recover what it threw
away. Centring first spends the full float32 relative precision on the *spread*,
which is what ε is actually measured against.

Everything then runs in float32. TF32 would be roughly twice as fast, but its
10-bit mantissa cannot resolve the ε boundary: the absolute error swamps `eps²`
and the resulting net is silently invalid. TF32 is therefore not offered as an
option, and it is switched off for the duration of the build and restored
afterwards, so a global PyTorch setting is neither honoured nor clobbered.

The same argument bounds float32 itself. The Gram identity cancels two terms of
size `max‖x‖²` down to a distance of size `eps²`, and a 24-bit mantissa leaves an
absolute error of about `max‖x‖² · 2⁻²³`. Once that is not small against `eps²`,
the comparison is noise. Centring removes the data's offset but not its
**extent**, so the method checks the ratio directly and, when float32 cannot
resolve ε on the data it was given, warns and falls back to `method="balltree"`
rather than return a net that merely looks valid. In practice the method needs
`max‖x − mean‖ / eps` to stay below roughly `10³`; rescale the data, or raise ε,
if you hit the fallback.

## Scaling

The distance products cost `O(n_samples × n_landmarks × n_features)`. The
advantage over the CPU methods is largest when the landmark count stays
*sublinear* in `n_samples` — the usual case, since sampling a fixed point cloud
ever more densely leaves the net size to saturate. If the landmark count instead
grows linearly with `n_samples`, because there is genuinely new structure at
every scale, the cost approaches quadratic and the advantage narrows. Comparing
`n_samples` against `Graph.number_of_nodes()` tells you which regime you are in.

Note that the two device passes are not necessarily where the wall-clock goes.
On a 1,000,000-point cloud with ~1,800 landmarks the device work takes about
`0.6 s`, while turning roughly 7.6 million incidence pairs into per-ball index
lists on the host takes several times that. The larger the coverage, the more
the host post-processing dominates — which is worth knowing before reaching for
a faster GPU.

## Tuning

| Keyword | Default | Meaning |
|---|---|---|
| `device` | `"cuda:0"` | CUDA device to run on; an `int` is read as `cuda:<int>` |
| `chunk` | auto | Points streamed per batch |
| `lblk` | `8192` | Column-tile width along the landmark axis |

```python
bm = BallMapper(X=my_pointcloud, eps=0.08, method="gpu", device=1)
```

Neither `chunk` nor `lblk` changes the result — they trade memory against launch
overhead only. The defaults are usually right.

`chunk` is chosen from the free VRAM and capped at 4096. That cap is deliberate:
the within-chunk independent-set step costs `O(chunk²)` — the very first chunk is
entirely uncovered, which is the worst case — while the total cost of the
chunk-versus-landmark products does not depend on the chunk size at all. A
*smaller* chunk is therefore generally the faster one. Raise it only if
per-chunk launch overhead is visibly dominating on a small dataset.

`lblk` tiles the landmark axis so the intermediate distance block stays a fixed
size no matter how many landmarks have accumulated. Lower it if you run out of
VRAM on a device with a very large net.

## Verifying a result

The two properties that define the cover are cheap to check directly, which is
worth doing once on a new dataset:

```python
import numpy as np

X = my_pointcloud
eps = 0.08
bm = BallMapper(X=X, eps=eps, method="gpu")
landmarks = np.array([bm.Graph.nodes[n]["landmark"] for n in bm.Graph.nodes])

# maximality: every point is covered
covered = set()
for pts in bm.points_covered_by_landmarks.values():
    covered.update(int(p) for p in pts)
assert covered == set(range(len(X)))

# independence: no two landmarks are within eps
centers = X[landmarks]
gaps = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
np.fill_diagonal(gaps, np.inf)
assert gaps.min() > eps
```

To check the stronger property — that you got the greedy graph — compare against
`method="balltree"` on a subsample:

```python
reference = BallMapper(X=X[:20000], eps=eps, method="balltree")
device = BallMapper(X=X[:20000], eps=eps, method="gpu")
assert [reference.Graph.nodes[n]["landmark"] for n in reference.Graph.nodes] == [
    device.Graph.nodes[n]["landmark"] for n in device.Graph.nodes
]
```

## Benchmarking

`benchmarks/benchmark_ballmapper.py` accepts `gpu` like any other method:

```bash
uv run python benchmarks/benchmark_ballmapper.py \
    --methods greedy balltree gpu \
    --ns 5000 20000 100000 500000 \
    --reps 5 \
    --skip-over 300
```

`--skip-over` drops a method from the remaining, larger cells once a single
build exceeds the given number of seconds, which keeps a run that includes the
quadratic `greedy` method finite. Anything left out is listed in the report, and
so is a `gpu` run that fell back for want of a device — a fallback is never
tabulated as if it were a device result. See [Benchmarks](benchmarks.md) for the
rest of the options.
