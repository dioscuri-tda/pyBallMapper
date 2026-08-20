# Getting started

[![version](https://img.shields.io/badge/version-0.3.1-blue)](https://pypi.org/project/pyBallMapper)
[![Documentation Status](https://readthedocs.org/projects/pyballmapper/badge/?version=latest)](https://pyballmapper.readthedocs.io/en/latest/?badge=latest)

Python version of the BallMapper algorithm described in [arXiv:1901.07410 ](https://arxiv.org/abs/1901.07410) .  

## Install the package 📦
```
pip install pyballmapper
```

### Basic usage
```
from pyballmapper import BallMapper
bm = BallMapper(X = my_pointcloud,    # the pointcloud, as a array-like of shape (n_samples, n_features)
                eps = 4.669)          # the radius of the covering balls
```

### Faster construction on large datasets ⚡

The default landmark search compares every point against every landmark, which
scales as `O(n_samples**2)` and becomes slow for large point clouds. For
Euclidean data you can instead pass `method="balltree"`, which uses a
[scikit-learn `BallTree`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html)
to select landmarks and compute coverage, and a sparse matrix product to find
the edges:

```
from pyballmapper import BallMapper
bm = BallMapper(X = my_pointcloud,
                eps = 4.669,
                method = "balltree")   # fast, euclidean-only
```

`method="balltree"` produces **the same graph** (identical landmarks and edges)
as the default method — it is only faster. It supports the Euclidean metric
without orbits; for any other metric, or when `orbits` are given, it
automatically falls back to the default greedy search.

### Running on a GPU 🚀

For very large point clouds there is also `method="gpu"`, which runs the
landmark search on a CUDA device via [PyTorch](https://pytorch.org):

```
from pyballmapper import BallMapper
bm = BallMapper(X = my_pointcloud,
                eps = 4.669,
                method = "gpu")        # CUDA, euclidean-only
```

This is not an approximation. The greedy rule is exactly the
lexicographically-first maximal independent set of the `eps`-proximity graph,
and the GPU method computes that same set in parallel rounds instead of one
point at a time — so it returns **the same landmarks and the same edges** as the
default method. Measured identical from 5,000 to 1,000,000 points. (Distances
are resolved in `float32` there, so a point sitting within rounding of the `eps`
boundary could in principle be classified differently; the result is then a
different, still perfectly valid cover. `method="balltree"` is the one that
guarantees equality.)

PyTorch is not a dependency of pyBallMapper. Without it, or without a working
CUDA device, `method="gpu"` warns and falls back to `method="balltree"`, so it
is safe to leave in code that also runs on CPU-only machines. See
[the GPU page](gpu.md) for details.

For more info check out the [example notebooks](https://github.com/dgurnari/pyBallMapper/tree/main/notebooks) or the [documentation](https://pyballmapper.readthedocs.io).