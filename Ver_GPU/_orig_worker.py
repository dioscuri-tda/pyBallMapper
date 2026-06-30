"""_orig_worker.py — isolated subprocess build of the naive 'original'
(pyballmapper) Ball Mapper.

Run as:  python3 _orig_worker.py <X.npy> <eps> <out.pkl>

It builds ``pyballmapper.BallMapper`` in a FRESH process (numpy + pyballmapper
only — no GPU/CUDA imported) and writes a portable graph summary (pickle) the
parent reconstructs into a shim object (bench_common._ShimBM).

The whole point is ISOLATION: at the L->N degeneracy region the naive float64
build can SEGFAULT inside native code (this is what killed the eps-sweep at
eps=0.03 with rc=139).  Here that only kills this subprocess — the parent reads
a non-zero returncode, records a failed cell, and the sweep continues.
faulthandler is on so a crash prints a C-level traceback to stderr, which the
parent captures and logs.
"""

from __future__ import annotations

import faulthandler
import pickle
import sys
import time

import numpy as np

faulthandler.enable()


def main():
    if len(sys.argv) != 4:
        print("usage: _orig_worker.py <X.npy> <eps> <out.pkl>", file=sys.stderr)
        return 2
    x_path, eps_s, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    eps = float(eps_s)
    X = np.load(x_path)

    # float64 so the naive reference matches the float64 BallTree net exactly
    # (same policy as bench_common.build); no GPU backend is imported here.
    from pyballmapper import BallMapper

    t0 = time.perf_counter()
    bm = BallMapper(X=np.ascontiguousarray(X, dtype=np.float64), eps=eps)
    dt = time.perf_counter() - t0

    G = bm.Graph
    bt = getattr(bm, "build_time_s", None)
    summary = {
        "build_dt": dt,
        "phases": (dict(bt) if isinstance(bt, dict) else None),
        # node -> landmark point-id, edges as node pairs, and per-node coverage:
        # exactly what bench_common's accessors + bm_validate need downstream.
        "nodes": [(n, int(G.nodes[n]["landmark"])) for n in G.nodes],
        "edges": [(u, v) for u, v in G.edges],
        "covers": {
            n: np.asarray(bm.points_covered_by_landmarks[n], dtype=np.int64)
            for n in G.nodes
        },
    }
    with open(out_path, "wb") as fh:
        pickle.dump(summary, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
