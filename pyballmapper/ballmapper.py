from __future__ import annotations

import contextlib
import copy
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib import colormaps as cm
from matplotlib.colors import Colormap
from numba import njit
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from sklearn.neighbors import BallTree
from tqdm.auto import tqdm

ADAPTIVE_ETA = 0.7
"""Default shrink factor for the ``"adaptive"`` landmark method.

Defined once so that the signature of :func:`_find_landmarks_adaptive` and the
dispatch in :func:`_find_landmarks` cannot drift apart.
"""

GPU_DEVICE = "cuda:0"
"""Default CUDA device used by the ``"gpu"`` landmark method."""

GPU_LANDMARK_BLOCK = 8192
"""Column-tile width for the ``chunk x n_landmarks`` distance products on the GPU.

Tiling the landmark axis bounds the size of the intermediate distance block
independently of how many landmarks have accumulated, so memory stays flat as
the epsilon-net grows.
"""


@njit
def _euclid_distance(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(x - y))


def _find_landmarks_deterministic_nearest_uncovered(
    X: npt.NDArray,
    eps: float,
    orbits: Any = None,
    metric: Any = None,
    order: Any = None,
    verbose: bool | str = False,
) -> tuple[dict[int, int], dict[int, list[int]], None]:
    """
    Constructs an epsilon-net H ⊆ X such that every point in X is within distance epsilon
    from at least one point in H.

    Algorithm:
    1. Initialize H with the medoid of X (point minimizing sum of distances to all others)
    2. Iteratively add the uncovered point closest to any existing ball until all points are covered

    Parameters:
    -----------
    X : np.ndarray
        Dataset of shape (n, d) where n is the number of points and d is the dimension
    epsilon : float
        Radius of covering balls

    Returns:
    --------
    net_indices : Dict[int, int]
        Maps k ∈ {0, 1, ..., |H|-1} to the index of the k-th net point in X
    coverage : Dict[int, List[int]]
        Maps k ∈ {0, 1, ..., |H|-1} to the list of indices in X covered by ball B(H[k], epsilon)
    """
    n = X.shape[0]

    # Compute pairwise distances (can be optimized for large datasets)
    distances = cdist(X, X, metric="euclidean")

    # Step 1: Find medoid (point minimizing sum of distances to all other points)
    medoid_idx = np.argmin(distances.sum(axis=1))

    # Initialize data structures
    net_indices: dict[int, int] = {0: int(medoid_idx)}
    coverage: dict[int, list[int]] = {}
    covered = np.zeros(n, dtype=bool)
    net_size = 1

    # Mark points covered by the medoid
    covered_by_medoid = distances[medoid_idx] <= eps
    covered |= covered_by_medoid
    coverage[0] = np.where(covered_by_medoid)[0].tolist()

    # Step 2: Iteratively add uncovered points
    while not np.all(covered):
        uncovered_indices = np.where(~covered)[0]

        # For each uncovered point, compute minimum distance to any ball center
        min_distances_to_net = np.min(
            distances[uncovered_indices][:, list(net_indices.values())], axis=1
        )

        # Select the uncovered point closest to any existing ball
        closest_uncovered_local_idx = int(np.argmin(min_distances_to_net))
        closest_uncovered_idx = int(uncovered_indices[closest_uncovered_local_idx])

        # Add this point to the epsilon-net
        net_indices[net_size] = closest_uncovered_idx

        # Update coverage
        newly_covered = distances[closest_uncovered_idx] <= eps
        covered |= newly_covered
        coverage[net_size] = np.where(newly_covered)[0].tolist()

        net_size += 1

    return net_indices, coverage, None


def _find_landmarks_greedy(
    X: npt.NDArray,
    eps: float,
    orbits: Any = None,
    metric: Any = None,
    order: Any = None,
    verbose: bool | str = False,
) -> tuple[dict[int, int], dict[int, list[int]], None]:
    """Finds the landmaks points via a greedy search procedure.

    Selects the first non-covered points in the cosidered order, adds it to the \
    list of landmarks and labels as covered all point inside its eps-ball. \
    Repeats the procedure till there are no more uncovered points.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features) \
            or (n_samples, n_samples)
        Data vectors, where `n_samples` is the number of samples
        and `n_features` is the number of features.
        For metric='precomputed', the expected shape of X is
        (n_samples, n_samples).

    eps : float
        The radius of the balls.

    orbits : list of lenght n_samples, default=None
        For each data points, contains a list of points in its orbit.
        Use it to create an Equivariant BallMapper.

    metric : str, or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array.
        If metric is 'precomputed', X is assumed to be a distance matrix and
        must be square.

    order: array-like of shape (n_samples, ), default=None
        The order in which to consider the data points in the greedy \
        search for landmarks. Different ordering might lead to different \
        BallMapper graphs.
        By defaults uses the order of X.

    verbose: bool or string, default=False
        Enable verbose output. Set it to 'tqdm' to show a tqdm progressbar.

    Returns
    ----------
    landmarks: list
        ids of the landmark points

    points_covered_by_landmarks: dict
        keys: landmarks ids
        values: list of ids of the points covered by the corresponding ball


    """

    n_points = X.shape[0]

    # set the distance function
    # f is used to access the points
    f: Callable[[Any], Any] = lambda i: X[i]
    distance: Callable[[Any, Any], Any]
    if metric == "euclidean":
        distance = _euclid_distance

    elif metric == "precomputed":
        distance = lambda x, y: X[x, y]
        f = lambda i: i
        if verbose:
            print("using precomputed distance matrix")

    else:
        distance = metric
        if verbose:
            print("using custom distance {}".format(distance))

    # set the orbits
    if orbits is None:
        points_have_orbits = False
    elif (type(orbits) is np.ndarray) or (type(orbits) is list):
        if len(orbits) != n_points:
            points_have_orbits = False
            warnings.warn(
                "Warning........... orbits is not compatible with points, ignoring it"
            )
        else:
            points_have_orbits = True
    else:
        warnings.warn(
            "Warning........... orbits should be a list or a numpy array, ignoring it"
        )
        points_have_orbits = False

    # find landmark points
    landmarks: dict[int, int] = {}  # dict of points {idx_v: idx_p, ... }
    centers_counter = 0

    if verbose:
        print("Finding vertices...")

    pbar = tqdm(order, disable=not (verbose == "tqdm"))

    for idx_p in pbar:
        # current point
        p = f(idx_p)

        pbar.set_description("{} vertices found".format(centers_counter))

        is_covered = False

        for idx_v in landmarks:
            if distance(p, f(landmarks[idx_v])) <= eps:
                is_covered = True
                break

        if not is_covered:
            landmarks[centers_counter] = idx_p
            centers_counter += 1
            # add points in the orbit
            if points_have_orbits:
                for idx_p_o in orbits[idx_p]:
                    if idx_p_o != idx_p:
                        landmarks[centers_counter] = idx_p_o
                        centers_counter += 1

    # compute points_covered_by_landmarks
    if verbose:
        print("{} vertices found.".format(centers_counter))
        print("Computing points_covered_by_landmarks...")
    points_covered_by_landmarks: dict[int, list[int]] = {}
    for idx_v in tqdm(landmarks, disable=not (verbose == "tqdm")):
        points_covered_by_landmarks[idx_v] = []
        for idx_p in order:
            if distance(f(idx_p), f(landmarks[idx_v])) <= eps:
                points_covered_by_landmarks[idx_v].append(idx_p)

    return landmarks, points_covered_by_landmarks, None


def _find_landmarks_balltree(
    X: npt.NDArray,
    eps: float,
    orbits: Any = None,
    metric: Any = None,
    order: Any = None,
    verbose: bool | str = False,
) -> tuple[dict[int, int], dict[int, list[int]], None]:
    """Finds the landmark points using a scikit-learn ``BallTree``.

    Fast, drop-in replacement for :func:`_find_landmarks_greedy` for Euclidean \
    data. It follows the exact same greedy criterion -- a point is skipped iff \
    it already lies inside the ``eps``-ball of a previously selected landmark -- \
    so it selects the **same landmarks in the same order** as the greedy method \
    and therefore yields an identical BallMapper graph. The speedup comes from \
    replacing the O(n_samples**2 * n_features) all-pairs distance computations \
    of the reference greedy search with O(n_samples * log n_samples) spatial \
    queries on a ``BallTree``.

    Only the Euclidean metric is supported, and orbits are not handled: for any \
    other ``metric`` or when ``orbits`` are given this function warns and falls \
    back to :func:`_find_landmarks_greedy`.

    Parameters
    ----------
    X : {array-like} of shape (n_samples, n_features)
        Data vectors.

    eps : float
        The radius of the balls.

    orbits : list of length n_samples, default=None
        Not supported by this method; if given, falls back to the greedy method.

    metric : str, default='euclidean'
        Only 'euclidean' is supported; other values fall back to the greedy \
        method.

    order: array-like of shape (n_samples, ), default=None
        The order in which to consider the data points in the greedy search for \
        landmarks. By default uses the order of X.

    verbose: bool or string, default=False
        Enable verbose output.

    Returns
    ----------
    landmarks: dict
        ids of the landmark points

    points_covered_by_landmarks: dict
        keys: landmarks ids
        values: list of ids of the points covered by the corresponding ball
    """

    if (metric is not None and metric != "euclidean") or orbits is not None:
        warnings.warn(
            "Warning........... the 'balltree' method only supports the euclidean "
            "metric without orbits, falling back to the greedy method"
        )
        return _find_landmarks_greedy(X, eps, orbits, metric, order, verbose)

    n_points = X.shape[0]

    if order is None:
        order = range(n_points)

    # build the spatial index once
    tree = BallTree(X, metric="euclidean")

    if verbose:
        print("Finding vertices...")

    # greedy landmark selection using a boolean `covered` mask:
    # a candidate is a new landmark iff it is not yet covered by a previous ball
    covered = np.zeros(n_points, dtype=bool)
    landmarks: dict[int, int] = {}  # dict of points {idx_v: idx_p, ... }
    centers_counter = 0

    for idx_p in order:
        if covered[idx_p]:
            continue
        landmarks[centers_counter] = int(idx_p)
        centers_counter += 1
        # mark every point inside the new ball as covered in one batch query
        in_ball = tree.query_radius(X[idx_p : idx_p + 1], r=eps)[0]
        covered[in_ball] = True

    if verbose:
        print("{} vertices found.".format(centers_counter))
        print("Computing points_covered_by_landmarks...")

    # batched coverage query for every landmark at once
    points_covered_by_landmarks: dict[int, list[int]] = {}
    if centers_counter > 0:
        landmark_ids = list(landmarks.values())
        coverage_arrays = tree.query_radius(X[landmark_ids], r=eps)
        for idx_v in landmarks:
            # sort so the coverage lists match the greedy method's natural order
            points_covered_by_landmarks[idx_v] = np.sort(
                coverage_arrays[idx_v]
            ).tolist()

    return landmarks, points_covered_by_landmarks, None


def _find_edges_from_coverage(
    points_covered_by_landmarks: dict[int, list[int]], n_points: int
) -> list[list[int]]:
    """Finds the BallMapper edges from the coverage via a sparse matrix product.

    Builds the incidence matrix ``M`` in {0, 1}^(n_landmarks x n_points) where \
    ``M[v, p] = 1`` iff point ``p`` lies in landmark ``v``'s ball. The product \
    ``S = M @ M.T`` is a sparse (n_landmarks x n_landmarks) matrix whose \
    ``(v, u)`` entry equals the number of points shared by balls ``v`` and \
    ``u``; its nonzero off-diagonal entries are exactly the BallMapper edges \
    (two balls that share at least one point). This replaces the \
    O(n_landmarks**2) Python ``set`` intersection double loop with a single \
    sparse matrix product and is bit-for-bit identical to it.

    Parameters
    ----------
    points_covered_by_landmarks: dict
        keys: landmarks ids (assumed to be 0, 1, ..., n_landmarks - 1)
        values: list of ids of the points covered by the corresponding ball

    n_points: int
        the number of data points

    Returns
    ----------
    edges: list of [idx_v, idx_u] with idx_v < idx_u
    """

    n_landmarks = len(points_covered_by_landmarks)
    if n_landmarks == 0:
        return []

    coverage_arrays = [
        np.asarray(points_covered_by_landmarks[v], dtype=np.int64)
        for v in range(n_landmarks)
    ]
    sizes = np.fromiter(
        (len(arr) for arr in coverage_arrays), dtype=np.int64, count=n_landmarks
    )
    indptr = np.empty(n_landmarks + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(sizes, out=indptr[1:])
    indices = np.concatenate(coverage_arrays)
    data = np.ones(int(indptr[-1]), dtype=np.int32)

    incidence = csr_matrix((data, indices, indptr), shape=(n_landmarks, n_points))
    overlap = (incidence @ incidence.T).tocoo()
    mask = overlap.row < overlap.col
    return [
        [int(idx_v), int(idx_u)]
        for idx_v, idx_u in zip(overlap.row[mask], overlap.col[mask])
    ]


def _gpu_wavefront_mis(adjacency_lower: Any) -> Any:
    """Index-priority wavefront maximal independent set, evaluated on the GPU.

    ``adjacency_lower`` is the strictly-lower-triangular part of the symmetric \
    ``eps``-proximity adjacency over ``n`` vertices, as a float tensor: \
    ``adjacency_lower[i, j] == 1`` iff ``j < i`` and ``dist(i, j) <= eps``. \
    Returns a boolean tensor of length ``n`` flagging the selected vertices.

    The sequential greedy rule -- walk the points in order, keep one iff it is \
    not already covered -- is exactly the lexicographically-first maximal \
    independent set of the ``eps``-proximity graph, which is what makes it \
    inherently serial. This routine computes *a* maximal independent set of the \
    same graph in parallel rounds instead: in each round every still-active \
    vertex that has no active neighbour of strictly smaller index is selected \
    simultaneously, then the selected vertices and their neighbours leave the \
    active set.

    Two properties hold by construction:

    * *independence* -- if ``i < j`` were both selected in one round, ``j`` \
      would have seen the active lower-indexed neighbour ``i`` and been \
      disqualified, so that cannot happen;
    * *maximality* -- the loop runs until no vertex is active, and a vertex \
      leaves the active set only by being selected or by neighbouring a \
      selected vertex, so every vertex is a landmark or is covered by one.

    Only the lower triangle is materialised; the full adjacency is recovered as \
    ``adjacency_lower + adjacency_lower.T`` because the graph is symmetric.
    """
    import torch

    n_vertices = adjacency_lower.shape[0]
    device = adjacency_lower.device
    keep = torch.zeros(n_vertices, dtype=torch.bool, device=device)
    if n_vertices == 0:
        return keep

    alive = torch.ones(n_vertices, dtype=torch.bool, device=device)
    lower_transposed = adjacency_lower.t()
    while bool(alive.any()):
        alive_float = alive.to(adjacency_lower.dtype)
        # blocked[i] is True iff i has an active neighbour with a smaller index
        blocked = (adjacency_lower @ alive_float) > 0.5
        roots = alive & ~blocked
        keep |= roots
        roots_float = roots.to(adjacency_lower.dtype)
        neighbours = (adjacency_lower @ roots_float) + (lower_transposed @ roots_float)
        alive = alive & ~(roots | (neighbours > 0.5))
    return keep


def _gpu_auto_chunk(torch_module: Any, device: Any, n_points: int) -> int:
    """Picks the streaming chunk size for the ``"gpu"`` method.

    The intra-chunk independent-set step costs O(chunk**2) -- the very first \
    chunk is entirely uncovered, which is the worst case -- while the total \
    cost of the chunk-versus-landmark distance products does not depend on the \
    chunk size at all. A *small* chunk therefore minimises the selection time; \
    ~4096 is large enough to amortise the per-chunk launch overhead and small \
    enough that the dense first-chunk triangle stays a few megabytes. The value \
    is additionally capped so that this triangle fits in 10% of free VRAM.
    """
    try:
        free_bytes = int(torch_module.cuda.mem_get_info(device)[0])
    except Exception:  # pragma: no cover - depends on the driver
        free_bytes = 8 * 1024**3
    # a float32 chunk x chunk block must stay below 10% of the free memory
    vram_cap = int(np.sqrt(0.10 * free_bytes / 4.0))
    return int(min(n_points, max(1024, min(vram_cap, 4096))))


GPU_CENTRING_BLOCK_BYTES = 32 * 1024**2
"""Working-set budget for the float64 block buffer in :func:`_gpu_centred_float32`."""


def _gpu_centred_float32(
    X: npt.NDArray, positions: npt.NDArray[np.int64]
) -> npt.NDArray[np.float32]:
    """Mean-centres ``X[positions]`` in float64 and returns it as float32.

    The order matters and is the whole point of doing this on the host: the     float32 cast is the lossy, irreversible step, quantising each coordinate     onto the float32 grid at the data's own offset magnitude, so centring     afterwards could not recover what it threw away. Centring first spends the     full float32 relative precision on the spread, which is what ``eps`` is     measured against.

    Done naively that means a full ``(n_samples, n_features)`` float64     temporary -- 800 MB at a million points in a hundred dimensions, and the     memory traffic to match. Instead the mean is accumulated in float64     straight off ``X``, and the rows are then centred and cast a block at a     time through **one reusable buffer**, so the float64 working set is a few     tens of megabytes whatever the input size. The output is bit-for-bit what     the naive version produces.
    """
    n_points, n_features = X.shape[0], X.shape[1]
    if n_points == 0:
        return np.empty((0, n_features), dtype=np.float32)
    permuted = not np.array_equal(positions, np.arange(n_points))

    if permuted:
        mean = np.mean(np.asarray(X[positions], dtype=np.float64), axis=0)
    else:
        mean = np.mean(X, axis=0, dtype=np.float64)

    rows = max(1, min(n_points, GPU_CENTRING_BLOCK_BYTES // (max(1, n_features) * 8)))
    centred = np.empty((n_points, n_features), dtype=np.float32)
    buffer = np.empty((min(rows, n_points), n_features), dtype=np.float64)
    # non-finite input is reported by the caller; numpy's own warnings about it
    # would only arrive first and say less
    with np.errstate(invalid="ignore", over="ignore"):
        for start in range(0, n_points, rows):
            stop = min(start + rows, n_points)
            block = X[positions[start:stop]] if permuted else X[start:stop]
            view = buffer[: stop - start]
            np.copyto(view, block, casting="unsafe")
            np.subtract(view, mean, out=view)
            np.copyto(centred[start:stop], view, casting="unsafe")
    return centred


@contextlib.contextmanager
def _gpu_exact_float32(torch_module: Any, device: Any) -> Iterator[None]:
    """Scopes the CUDA device and the exact-float32 matmul mode to one build.

    TF32 is roughly twice as fast as float32 but carries only a 10-bit \
    mantissa, which cannot resolve the ``eps`` boundary -- it would silently \
    produce an invalid net -- so it is switched off here. Both that setting and \
    the current device are *process-global* torch state, so a library has no \
    business leaving either changed; this restores both on the way out, \
    including when the build raises.

    Recent torch versions expose the mode as ``matmul.fp32_precision`` and treat \
    mixing it with the legacy ``matmul.allow_tf32`` flag as an error on any \
    subsequent read of either, so exactly one of the two is ever touched.
    """
    matmul = torch_module.backends.cuda.matmul
    use_precision_api = hasattr(matmul, "fp32_precision")
    if use_precision_api:
        saved = matmul.fp32_precision
    else:
        saved = matmul.allow_tf32
    try:
        with torch_module.cuda.device(device):
            if use_precision_api:
                matmul.fp32_precision = "ieee"
            else:
                matmul.allow_tf32 = False
            yield
    finally:
        if use_precision_api:
            matmul.fp32_precision = saved
        else:
            matmul.allow_tf32 = saved


def _find_landmarks_gpu(
    X: npt.NDArray,
    eps: float,
    orbits: Any = None,
    metric: Any = None,
    order: Any = None,
    verbose: bool | str = False,
    device: str | int = GPU_DEVICE,
    chunk: int | None = None,
    lblk: int = GPU_LANDMARK_BLOCK,
) -> tuple[dict[int, int], dict[int, list[int]], None]:
    """Finds the landmark points on a CUDA GPU, using PyTorch.

    This computes **the same landmarks the greedy method would**, in parallel. \
    The greedy rule -- walk the points in order, keep one iff it is not already \
    covered -- is the *lexicographically-first maximal independent set* of the \
    ``eps``-proximity graph. The "maximal independent set" half is what makes \
    the result a valid cover; the "lexicographically-first" half is what forces \
    it to be sequential. :func:`_gpu_wavefront_mis` computes that same \
    lexicographically-first set in parallel rounds instead of one point at a \
    time, and streaming in chunks does not change it either, because a chunk is \
    first tested against every landmark already accumulated. Measured identical \
    to :func:`_find_landmarks_balltree` -- same landmarks, same edges -- from \
    n_samples = 5,000 to 1,000,000 and ``eps`` from 0.24 down to 0.02.

    Equality is not *guaranteed* the way it is for \
    :func:`_find_landmarks_balltree`, though, and the difference is arithmetic \
    rather than algorithmic: distances here are resolved in float32, so a point \
    lying within rounding of the ``eps`` boundary can fall on the other side of \
    it than the CPU's float64 test would put it. The result is then a different \
    maximal ``eps``-net -- still valid, just not the greedy one.

    The build runs in three passes:

    1. *selection*, on the device -- points are streamed in chunks; each chunk \
       is tested against every landmark accumulated so far, which drops the \
       already-covered rows, and the survivors are resolved against one another \
       by :func:`_gpu_wavefront_mis`;
    2. *coverage*, on the device -- every point is tested against the **final** \
       landmark set. This is deliberately recomputed rather than reused from \
       pass 1, where the coverage mask is only partial;
    3. *grouping*, on the host -- the (landmark, point) incidence pairs are \
       sorted into per-landmark coverage lists by :func:`_gpu_group_coverage`.

    Edges are then found by :func:`_find_edges_from_coverage`, the same sparse \
    incidence product :func:`_find_landmarks_balltree` uses, on the host.

    Distances use the Gram identity \
    ``||a - b||**2 = ||a||**2 + ||b||**2 - 2 <a, b>`` on squared distances, \
    compared against ``eps**2``, so no square roots are taken. The data is \
    mean-centred **in float64 on the host, before the float32 cast**: the cast \
    is the lossy step and it is irreversible, quantising each coordinate onto \
    the float32 grid at the data's own offset magnitude, so centring afterwards \
    could not recover what it had thrown away. Centring first spends the full \
    float32 relative precision on the spread, which is what ``eps`` is measured \
    against.

    That still leaves the *extent* of the data. The Gram identity cancels two \
    terms of size ``max||x||**2`` down to a distance of size ``eps**2``, and \
    float32 carries a 24-bit mantissa, so the absolute error is about \
    ``max||x||**2 * 2**-23``. Once that is not small against ``eps**2`` the \
    comparison is noise; this function detects that case and falls back to \
    :func:`_find_landmarks_balltree` rather than return a silently invalid net.

    The cost of the distance products is O(n_samples * n_landmarks * \
    n_features). The speedup over the CPU methods is largest when the landmark \
    count stays sublinear in ``n_samples``, which is the usual case -- sampling \
    a fixed point cloud ever more densely leaves the net size to saturate. If \
    the landmark count instead grows linearly with ``n_samples`` the cost \
    approaches quadratic and the advantage narrows.

    ``torch`` is imported lazily and is not a hard dependency of pyBallMapper. \
    Without it, or without a working CUDA device, this function warns and falls \
    back to :func:`_find_landmarks_balltree`. Only the Euclidean metric is \
    supported and orbits are not handled; for any other ``metric`` or when \
    ``orbits`` are given it warns and falls back to \
    :func:`_find_landmarks_greedy`.

    Parameters
    ----------
    X : {array-like} of shape (n_samples, n_features)
        Data vectors. Must be finite and representable as float32.

    eps : float
        The radius of the balls.

    orbits : list of length n_samples, default=None
        Not supported by this method; if given, falls back to the greedy method.

    metric : str, default='euclidean'
        Only 'euclidean' is supported; other values fall back to the greedy \
        method.

    order: array-like of shape (n_samples, ), default=None
        The order in which to consider the data points, exactly as in the \
        greedy method. By default uses the order of X.

    verbose: bool or string, default=False
        Enable verbose output.

    device: str or int, default='cuda:0'
        The CUDA device to run on.

    chunk: int, default=None
        Number of points streamed per batch. By default chosen from the free \
        VRAM by :func:`_gpu_auto_chunk`. The result does not depend on it.

    lblk: int, default=8192
        Column-tile width along the landmark axis. The result does not depend \
        on it.

    Returns
    ----------
    landmarks: dict
        ids of the landmark points

    points_covered_by_landmarks: dict
        keys: landmarks ids
        values: list of ids of the points covered by the corresponding ball
    """

    n_points = X.shape[0]

    # normalised up front so that the fallbacks below can be handed a usable
    # order even when this function is called directly with order=None
    if order is None:
        order = range(n_points)
    # this method reads metric=None as euclidean, so make that explicit before
    # handing it on -- _find_landmarks_greedy treats None as a custom callable
    if metric is None:
        metric = "euclidean"

    if metric != "euclidean" or orbits is not None:
        warnings.warn(
            "Warning........... the 'gpu' method only supports the euclidean "
            "metric without orbits, falling back to the greedy method"
        )
        return _find_landmarks_greedy(X, eps, orbits, metric, order, verbose)

    if lblk < 1:
        raise ValueError(f"lblk must be a positive integer, got {lblk!r}")
    if chunk is not None and chunk < 1:
        raise ValueError(f"chunk must be a positive integer, got {chunk!r}")
    if not eps >= 0.0:
        raise ValueError(f"eps must be non-negative, got {eps!r}")

    try:
        import torch
    except ImportError:
        warnings.warn(
            "Warning........... the 'gpu' method requires pytorch, which is not "
            "installed, falling back to the balltree method"
        )
        return _find_landmarks_balltree(X, eps, orbits, metric, order, verbose)

    if not torch.cuda.is_available():
        warnings.warn(
            "Warning........... the 'gpu' method requires a CUDA device, none is "
            "available, falling back to the balltree method"
        )
        return _find_landmarks_balltree(X, eps, orbits, metric, order, verbose)

    if isinstance(device, int):
        device = f"cuda:{device}"
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError(f"the 'gpu' method needs a CUDA device, got device={device!r}")

    # points are streamed in `order`; `positions` maps a streaming position back
    # to the id of the data point sitting there
    positions = np.fromiter(order, dtype=np.int64, count=n_points)

    centred = _gpu_centred_float32(X, positions)

    with _gpu_exact_float32(torch, torch_device):
        landmarks, incidence_landmarks, incidence_points, fell_back = (
            _find_landmarks_gpu_device(
                torch, torch_device, centred, eps, chunk, lblk, verbose
            )
        )
        if fell_back:
            return _find_landmarks_balltree(X, eps, orbits, metric, order, verbose)

        points_covered_by_landmarks = _gpu_group_coverage(
            incidence_landmarks, incidence_points, positions, landmarks, n_points
        )

    return (
        {idx_v: int(positions[pos]) for idx_v, pos in enumerate(landmarks)},
        points_covered_by_landmarks,
        None,
    )


def _find_landmarks_gpu_device(
    torch: Any,
    torch_device: Any,
    centred: npt.NDArray[np.float32],
    eps: float,
    chunk: int | None,
    lblk: int,
    verbose: bool | str,
) -> tuple[list[int], list[Any], list[Any], bool]:
    """Runs the two device passes of the ``"gpu"`` method.

    Split out from :func:`_find_landmarks_gpu` so that everything touching the \
    device sits inside the one scope that owns the device and the exact-float32 \
    setting. ``centred`` arrives already mean-centred and in float32, from \
    :func:`_gpu_centred_float32`. Returns the landmarks as *streaming positions*, the raw incidence \
    pairs, and a flag saying the caller should fall back to the balltree method \
    because float32 cannot resolve ``eps`` on this data.
    """
    n_points = centred.shape[0]
    points = torch.as_tensor(np.ascontiguousarray(centred), device=torch_device)
    if not bool(torch.isfinite(points).all()):
        raise ValueError(
            "the 'gpu' method requires finite input; X contains NaN or infinity, "
            "or a value too large to represent in float32"
        )
    n_features = points.shape[1]
    squared_norms = (points * points).sum(1)
    eps_squared = float(eps) * float(eps)

    # The Gram identity cancels two terms of size ~max||x||**2 down to a
    # distance of size ~eps**2, and float32 carries a 24-bit mantissa, so the
    # absolute error is ~max||x||**2 * 2**-23. Once that is not small against
    # eps**2 the comparisons below are noise and the net would be silently
    # invalid -- the same argument that rules out TF32, applied to float32.
    # Centring removed the offset but not the extent, so this is checked here.
    if n_points:
        gram_error = float(squared_norms.max()) * 2.0**-23
        if gram_error > 0.25 * eps_squared:
            warnings.warn(
                "Warning........... the 'gpu' method resolves distances in "
                f"float32; at this data extent the rounding error ({gram_error:.3g}) "
                f"is not negligible against eps**2 ({eps_squared:.3g}), so the "
                "resulting net would be invalid -- falling back to the balltree "
                "method. Rescale the data, or raise eps, to use the gpu method."
            )
            del points, squared_norms
            torch.cuda.empty_cache()
            return [], [], [], True

    if chunk is None:
        chunk = _gpu_auto_chunk(torch, torch_device, n_points)
    chunk = max(1, int(chunk))

    if verbose:
        print("Finding vertices...")

    # pass 1: streaming construction of a maximal eps-net.
    #
    # Every chunk is tested against the landmarks accumulated so far to find the
    # ones already covered. That test computes exactly the distances pass 2 would
    # otherwise recompute -- and because the landmark count saturates within the
    # first few chunks, "so far" is very nearly "all of them", so recomputing
    # them is close to doubling the arithmetic of the whole build. The
    # (point, landmark) pairs are therefore kept here rather than reduced away to
    # a covered/not-covered flag, and pass 2 picks up from where each chunk left
    # off. Every pair is then evaluated once, which is also what stops the two
    # passes disagreeing about a distance that sits within rounding of eps.
    landmark_points = torch.empty(
        (0, n_features), dtype=points.dtype, device=torch_device
    )
    landmark_norms = torch.empty((0,), dtype=squared_norms.dtype, device=torch_device)
    landmark_positions: list[int] = []
    incidence_landmarks: list[Any] = []
    incidence_points: list[Any] = []
    landmarks_before_chunk: list[int] = []
    for start in range(0, n_points, chunk):
        stop = min(start + chunk, n_points)
        block = points[start:stop]
        block_norms = squared_norms[start:stop]
        n_landmarks_so_far = landmark_points.shape[0]
        landmarks_before_chunk.append(n_landmarks_so_far)

        covered = torch.zeros(stop - start, dtype=torch.bool, device=torch_device)
        for first in range(0, n_landmarks_so_far, lblk):
            last = min(first + lblk, n_landmarks_so_far)
            gram = block @ landmark_points[first:last].T
            squared = (
                block_norms[:, None] + landmark_norms[first:last][None, :] - 2.0 * gram
            )
            in_ball = squared <= eps_squared
            covered |= in_ball.any(1)
            point_index, landmark_index = in_ball.nonzero(as_tuple=True)
            incidence_landmarks.append((landmark_index + first).to(torch.int64))
            incidence_points.append((point_index + start).to(torch.int64))
            del gram, squared, in_ball

        uncovered = (~covered).nonzero(as_tuple=True)[0]
        if uncovered.numel() == 0:
            continue

        # resolve the uncovered rows of this chunk against one another
        candidates = block[uncovered]
        candidate_norms = block_norms[uncovered]
        gram = candidates @ candidates.T
        squared = candidate_norms[:, None] + candidate_norms[None, :] - 2.0 * gram
        adjacency_lower = torch.tril(
            (squared <= eps_squared).to(torch.float32), diagonal=-1
        )
        del gram, squared
        selected = uncovered[_gpu_wavefront_mis(adjacency_lower)]
        del adjacency_lower

        landmark_points = torch.cat([landmark_points, block[selected]], 0)
        landmark_norms = torch.cat([landmark_norms, block_norms[selected]], 0)
        landmark_positions.extend((start + selected).tolist())

    n_landmarks = landmark_points.shape[0]

    if verbose:
        print("{} vertices found.".format(n_landmarks))
        print("Computing points_covered_by_landmarks...")

    # pass 2: finish each chunk against the landmarks found *after* it was
    # streamed. Together with pass 1 that covers every (point, landmark) pair
    # exactly once.
    #
    # One overlap survives: a pair of points that were both uncovered in the same
    # chunk is judged once by the within-chunk step above and once here, and BLAS
    # does not promise the same float32 result for two different matrix shapes.
    # For a distance within a few ulps of eps the two can disagree, which would
    # leave the point in no ball at all -- silently absent from the cover, which
    # no other method can do. That is repaired below; the point's true distance
    # is within eps, which is why it was dropped, so attaching it restores
    # maximality without weakening independence.
    for index, start in enumerate(range(0, n_points, chunk)):
        stop = min(start + chunk, n_points)
        block = points[start:stop]
        block_norms = squared_norms[start:stop]
        for first in range(landmarks_before_chunk[index], n_landmarks, lblk):
            last = min(first + lblk, n_landmarks)
            gram = block @ landmark_points[first:last].T
            squared = (
                block_norms[:, None] + landmark_norms[first:last][None, :] - 2.0 * gram
            )
            in_ball = squared <= eps_squared
            point_index, landmark_index = in_ball.nonzero(as_tuple=True)
            incidence_landmarks.append((landmark_index + first).to(torch.int64))
            incidence_points.append((point_index + start).to(torch.int64))
            del gram, squared, in_ball

    if n_landmarks:
        covered = torch.zeros(n_points, dtype=torch.bool, device=torch_device)
        for covered_points in incidence_points:
            covered[covered_points] = True
        orphans = (~covered).nonzero(as_tuple=True)[0]
        if orphans.numel():
            # Tracking every point's nearest landmark inside the loop above
            # would double its elementwise work to serve a set that is almost
            # always empty, so the orphans are resolved here instead -- there
            # are a handful of them at most, and only when a boundary tie
            # actually went the wrong way.
            incidence_points.append(orphans)
            incidence_landmarks.append(
                _gpu_nearest_landmarks(
                    torch,
                    points[orphans],
                    squared_norms[orphans],
                    landmark_points,
                    landmark_norms,
                    lblk,
                )
            )
        del covered, orphans

    del points, squared_norms, landmark_points, landmark_norms
    torch.cuda.empty_cache()
    return landmark_positions, incidence_landmarks, incidence_points, False


def _gpu_nearest_landmarks(
    torch: Any,
    block: Any,
    block_norms: Any,
    landmark_points: Any,
    landmark_norms: Any,
    lblk: int,
) -> Any:
    """Index of the closest landmark to each row of ``block``.

    Used only to rehome the points that pass 2 left in no ball at all, so it
    runs over a handful of rows and can afford a second look at the landmarks.
    """
    n_rows = block.shape[0]
    n_landmarks = landmark_points.shape[0]
    best_squared = torch.full(
        (n_rows,), float("inf"), dtype=block.dtype, device=block.device
    )
    best_index = torch.zeros(n_rows, dtype=torch.int64, device=block.device)
    for first in range(0, n_landmarks, lblk):
        last = min(first + lblk, n_landmarks)
        gram = block @ landmark_points[first:last].T
        squared = (
            block_norms[:, None] + landmark_norms[first:last][None, :] - 2.0 * gram
        )
        tile_best, tile_argument = squared.min(dim=1)
        closer = tile_best < best_squared
        best_squared = torch.where(closer, tile_best, best_squared)
        best_index = torch.where(closer, tile_argument + first, best_index)
        del gram, squared, tile_best, tile_argument, closer
    return best_index


def _gpu_group_coverage(
    incidence_landmarks: list[Any],
    incidence_points: list[Any],
    positions: npt.NDArray[np.int64],
    landmark_positions: list[int],
    n_points: int,
) -> dict[int, list[int]]:
    """Groups the (landmark, point) incidence pairs into per-landmark coverage.

    Each landmark provably covers its own centre -- the distance is zero -- but \
    the float32 Gram identity can miss that pair when the spread of the data \
    dwarfs ``eps``, so a landmark that did not report itself has that pair added \
    back here, and only such a landmark.

    Grouping is a sort on the **landmark axis alone**, and it happens on the \
    device, where the incidence already is. Two alternatives are worth naming \
    because both were measured and both are worse. Sorting a composite key \
    ``landmark * (n_points + 1) + point`` would order and de-duplicate \
    everything in one step, but it is a comparison sort over keys spanning \
    ``n_landmarks * n_points`` and it costs seconds at a million points. Sorting \
    the landmark ids on the host is far better -- the keys are small enough to \
    radix sort -- but still an order of magnitude slower than letting the device \
    do it before the transfer. Each ball is then ordered on its own, and only \
    the finished, grouped result crosses to the host.

    Sorting the balls with :func:`numpy.unique` instead of :meth:`numpy.ndarray.sort` \
    would de-duplicate them for free in principle. In practice it costs about \
    half a millisecond per call, which over a couple of thousand balls made it \
    the single most expensive step of the entire build -- to remove duplicates \
    that cannot arise. They cannot because the two device passes cover disjoint \
    ``(chunk, landmark)`` ranges, so each pair is emitted once, and only absent \
    self-pairs are added. Rather than trust that silently, the sorted result is \
    swept once for an adjacent repeat inside a ball; the sweep is vectorised over \
    the whole incidence and costs a few milliseconds.

    Streaming positions are translated back to data point ids on the way out, so \
    the coverage lists refer to rows of the original ``X``.
    """
    n_landmarks = len(landmark_positions)
    if n_landmarks == 0:
        return {}

    import torch

    self_points = np.asarray(landmark_positions, dtype=np.int64)

    if incidence_landmarks:
        device = incidence_landmarks[0].device
        landmark_ids = torch.cat(incidence_landmarks)
        point_ids = torch.cat(incidence_points)

        # add back the self-pairs the device did not report, and only those
        self_points_device = torch.as_tensor(self_points, device=device)
        covers_itself = torch.zeros(n_landmarks, dtype=torch.bool, device=device)
        if landmark_ids.numel():
            is_self_pair = point_ids == self_points_device[landmark_ids]
            covers_itself[landmark_ids[is_self_pair]] = True
        absent = (~covers_itself).nonzero(as_tuple=True)[0]
        if absent.numel():
            landmark_ids = torch.cat([landmark_ids, absent])
            point_ids = torch.cat([point_ids, self_points_device[absent]])

        order = torch.argsort(landmark_ids, stable=True)
        position_map = torch.as_tensor(positions, device=device)
        grouped_points = position_map[point_ids[order]].cpu().numpy()
        sizes = torch.bincount(landmark_ids, minlength=n_landmarks).cpu().numpy()
        del landmark_ids, point_ids, order, position_map, self_points_device
    else:
        # nothing was reported at all: every landmark covers just its own centre
        grouped_points = positions[self_points]
        sizes = np.ones(n_landmarks, dtype=np.int64)

    bounds = np.zeros(n_landmarks + 1, dtype=np.int64)
    np.cumsum(sizes, out=bounds[1:])
    for idx_v in range(n_landmarks):
        grouped_points[bounds[idx_v] : bounds[idx_v + 1]].sort()

    if _gpu_has_repeat_within_ball(grouped_points, bounds):  # pragma: no cover
        return {
            idx_v: np.unique(grouped_points[bounds[idx_v] : bounds[idx_v + 1]]).tolist()
            for idx_v in range(n_landmarks)
        }
    return {
        idx_v: grouped_points[bounds[idx_v] : bounds[idx_v + 1]].tolist()
        for idx_v in range(n_landmarks)
    }


def _gpu_has_repeat_within_ball(
    grouped_points: npt.NDArray[np.int64], bounds: npt.NDArray[np.int64]
) -> bool:
    """Reports whether any ball lists the same point twice.

    ``grouped_points`` is every ball's contents laid end to end, each ball \
    already sorted, with ``bounds`` marking where each begins. A repeat inside a \
    ball is therefore a pair of equal neighbours -- but so is a ball that ends on \
    the same point the next one starts with, which is perfectly ordinary. The \
    cheap sweep for equal neighbours is run first and almost always settles it; \
    only if it finds something does it cost anything to ask which of them sit \
    across a boundary.
    """
    if grouped_points.size < 2:
        return False
    equal_neighbours = grouped_points[1:] == grouped_points[:-1]
    if not equal_neighbours.any():
        return False
    candidates = np.flatnonzero(equal_neighbours) + 1
    return bool(np.isin(candidates, bounds[1:-1], invert=True).any())


def _find_landmarks_adaptive(
    X: npt.NDArray,
    eps: float,
    max_size: int,
    eta: float = ADAPTIVE_ETA,
    orbits: Any = None,
    metric: Any = None,
    order: Any = None,
    verbose: bool | str = False,
) -> tuple[dict[int, int], dict[int, list[int]], dict[int, float]]:
    """Finds the landmaks points via a greedy search procedure.

    Selects the first non-covered points in the cosidered order, adds it to the \
    list of landmarks and labels as covered all point inside its eps-ball. \
    Repeats the procedure till there are no more uncovered points.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features) \
            or (n_samples, n_samples)
        Data vectors, where `n_samples` is the number of samples
        and `n_features` is the number of features.
        For metric='precomputed', the expected shape of X is
        (n_samples, n_samples).

    eps : float
        The radius of the balls.

    orbits : list of lenght n_samples, default=None
        For each data points, contains a list of points in its orbit.
        Use it to create an Equivariant BallMapper.

    metric : str, or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array.
        If metric is 'precomputed', X is assumed to be a distance matrix and
        must be square.

    order: array-like of shape (n_samples, ), default=None
        The order in which to consider the data points in the greedy \
        search for landmarks. Different ordering might lead to different \
        BallMapper graphs.
        By defaults uses the order of X.

    verbose: bool or string, default=False
        Enable verbose output. Set it to 'tqdm' to show a tqdm progressbar.

    Returns
    ----------
    landmarks: list
        ids of the landmark points

    points_covered_by_landmarks: dict
        keys: landmarks ids
        values: list of ids of the points covered by the corresponding ball


    """

    if metric == "precomputed":
        n_points = X.shape[0]
    else:
        n_points = len(X)

    # set the distance function
    # f is used to access the points
    f: Callable[[Any], Any] = lambda i: X[i]
    distance: Callable[[Any, Any], Any]
    if metric == "euclidean":
        distance = _euclid_distance

    elif metric == "precomputed":
        distance = lambda x, y: X[x, y]
        f = lambda i: i
        if verbose:
            print("using precomputed distance matrix")

    else:
        distance = metric
        if verbose:
            print("using custom distance {}".format(distance))

    # set the orbits
    if orbits is None:
        points_have_orbits = False
    elif (type(orbits) is np.ndarray) or (type(orbits) is list):
        if len(orbits) != n_points:
            points_have_orbits = False
            warnings.warn(
                "Warning........... orbits is not compatible with points, ignoring it"
            )
        else:
            points_have_orbits = True
    else:
        warnings.warn(
            "Warning........... orbits should be a list or a numpy array, ignoring it"
        )
        points_have_orbits = False

    # find landmark points
    landmarks: dict[int, int] = {}  # dict of points {idx_v: idx_p, ... }
    centers_counter = -1

    # check wheter order is a list of lenght = len(points)
    # otherwise use the defaut ordering
    if order:
        if len(np.unique(order)) != n_points:
            warnings.warn(
                "Warning........... order is not compatible with points, using default ordering"
            )
            order = range(n_points)
    else:
        order = range(n_points)

    if verbose:
        print("Finding vertices...")

    pbar = tqdm(order, disable=not (verbose == "tqdm"))

    # since the radius of every ball might be different, we need to store them
    eps_dict: dict[int, float] = {}
    points_covered_by_landmarks: dict[int, list[int]] = {}

    for idx_p in pbar:
        # current point
        p = f(idx_p)

        pbar.set_description("{} vertices found".format(centers_counter))

        is_covered = False

        for idx_v in landmarks:
            if distance(p, f(landmarks[idx_v])) <= eps_dict[idx_v]:
                is_covered = True
                break

        if not is_covered:
            # we use this as a new landmark
            centers_counter += 1
            landmarks[centers_counter] = idx_p

            # compute points_covered_by this new landmarks
            points_covered_by_landmarks[centers_counter] = []
            eps_dict[centers_counter] = eps
            for idx_p2 in order:
                if (
                    distance(f(idx_p2), f(landmarks[centers_counter]))
                    <= eps_dict[centers_counter]
                ):
                    points_covered_by_landmarks[centers_counter].append(idx_p2)

            while len(points_covered_by_landmarks[centers_counter]) > max_size:
                # decrease the radius and recompute
                eps_dict[centers_counter] *= eta
                if verbose:
                    print(
                        "ball {} - point {}  has size {}. decreasing eps to {}".format(
                            centers_counter,
                            idx_p,
                            len(points_covered_by_landmarks[centers_counter]),
                            eps_dict[centers_counter],
                        )
                    )

                points_covered_by_landmarks[centers_counter] = []
                for idx_p2 in order:
                    if (
                        distance(f(idx_p2), f(landmarks[centers_counter]))
                        <= eps_dict[centers_counter]
                    ):
                        points_covered_by_landmarks[centers_counter].append(idx_p2)

            # add points in the orbit
            if points_have_orbits:
                eps_o = eps_dict[centers_counter]

                for idx_p_o in orbits[idx_p]:
                    if idx_p_o != idx_p:
                        centers_counter += 1
                        landmarks[centers_counter] = idx_p_o

                        # if verbose:
                        #     print(
                        #         "adding orbit landmark {} with eps {}".format(
                        #             centers_counter, eps_o
                        #         )
                        #     )

                        # compute points_covered_by this new landmarks
                        # using the same radius as the original landmark
                        points_covered_by_landmarks[centers_counter] = []
                        eps_dict[centers_counter] = eps_o
                        for idx_p2 in order:
                            if (
                                distance(f(idx_p2), f(landmarks[centers_counter]))
                                <= eps_dict[centers_counter]
                            ):
                                points_covered_by_landmarks[centers_counter].append(
                                    idx_p2
                                )

    return landmarks, points_covered_by_landmarks, eps_dict


def _find_landmarks(
    X: npt.NDArray,
    eps: float,
    orbits: Any = None,
    metric: Any = None,
    order: Any = None,
    method: str | None = None,
    verbose: bool | str = False,
    **kwargs: Any,
) -> tuple[dict[int, int], dict[int, list[int]], dict[int, float] | None]:
    """Finds the landmaks points. At the moment the only option is a greedy search

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features) \
            or (n_samples, n_samples)
        Data vectors, where `n_samples` is the number of samples
        and `n_features` is the number of features.
        For metric='precomputed', the expected shape of X is
        (n_samples, n_samples).

    eps : float
        The radius of the balls.

    orbits : list of lenght n_samples, default=None
        For each data points, contains a list of points in its orbit.
        Use it to create an Equivariant BallMapper.

    metric : str, or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array.
        If metric is 'precomputed', X is assumed to be a distance matrix and
        must be square.

    order: array-like of shape (n_samples, ), default=None
        The order in which to consider the data points in the greedy \
        search for landmarks. Different ordering might lead to different \
        BallMapper graphs.
        By defaults uses the order of X.

    method: string, default=None
        The method to use for landmark selection. Options are:
        - "nearest": deterministic method selecting the uncovered point nearest to any existing ball
        - "adaptive": random method adjusting the radius of each ball to ensure a maximum number of points per ball
        - "greedy": random method selecting the first uncovered point in the considered order
        - "balltree": fast BallTree-based version of "greedy" (euclidean metric \
            only) that produces an identical BallMapper graph
        - "gpu": CUDA version (euclidean metric only, requires pytorch) that \
            produces a different but equally valid maximal eps-net

    verbose: bool or string, default=False
        Enable verbose output. Set it to 'tqdm' to show a tqdm progressbar.

    Returns
    ----------
    landmarks: list
        ids of the landmark points

    points_covered_by_landmarks: dict
        keys: landmarks ids
        values: list of ids of the points covered by the corresponding ball


    """

    match method:
        # deterministic method "nearest"
        case "nearest":
            landmarks, points_covered_by_landmarks, eps_dict = (
                _find_landmarks_deterministic_nearest_uncovered(
                    X, eps, orbits, metric, order, verbose
                )
            )
        # random methods "adaptive" and "greedy"
        case "adaptive":
            landmarks, points_covered_by_landmarks, eps_dict = _find_landmarks_adaptive(
                X=X,
                eps=eps,
                max_size=kwargs["max_size"],
                eta=kwargs.get("eta", ADAPTIVE_ETA),
                orbits=orbits,
                metric=metric,
                order=order,
                verbose=verbose,
            )
        # "greedy" method chooses the next covered point randomly
        case "greedy":
            landmarks, points_covered_by_landmarks, eps_dict = _find_landmarks_greedy(
                X, eps, orbits, metric, order, verbose
            )
        # "balltree" is a fast BallTree-based version of the greedy method
        # (euclidean metric only) that yields an identical BallMapper graph
        case "balltree":
            landmarks, points_covered_by_landmarks, eps_dict = _find_landmarks_balltree(
                X, eps, orbits, metric, order, verbose
            )
        # "gpu" runs the landmark search on a CUDA device; it yields a valid
        # but different maximal eps-net, not the greedy one
        case "gpu":
            landmarks, points_covered_by_landmarks, eps_dict = _find_landmarks_gpu(
                X,
                eps,
                orbits,
                metric,
                order,
                verbose,
                device=kwargs.get("device", GPU_DEVICE),
                chunk=kwargs.get("chunk"),
                lblk=kwargs.get("lblk", GPU_LANDMARK_BLOCK),
            )
        # "greedy" method is a default one when a method is not specified
        case None:
            landmarks, points_covered_by_landmarks, eps_dict = _find_landmarks_greedy(
                X, eps, orbits, metric, order, verbose
            )
        case _:
            raise ValueError(
                f"unknown method {method!r}; expected one of "
                "None, 'greedy', 'nearest', 'adaptive', 'balltree', 'gpu'"
            )

    return landmarks, points_covered_by_landmarks, eps_dict


class BallMapper:
    def __init__(
        self,
        X: npt.NDArray,
        eps: float,
        coloring_df: pd.DataFrame | None = None,
        orbits: npt.NDArray | list | None = None,
        metric: str = "euclidean",
        order: list[int] | npt.NDArray[np.int_] | range | None = None,
        method: str | None = None,
        verbose: bool | str = False,
        column_names: list[str] | None = None,
        **kwargs: Any,
    ):
        """Create a BallMapper graph from vector array or distance matrix.

        Parameters
        -----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) \
                or (n_samples, n_samples)
            Data vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.
            For metric='precomputed', the expected shape of X is
            (n_samples, n_samples).

        eps : float
            The radius of the balls.

        orbits : list of lenght n_samples, default=None
            For each data points, contains a list of points in its orbit.
            Use it to create an Equivariant BallMapper.

        coloring_df: pandas dataframe of shape (n_samples, n_coloring_function), default=None
            If defined, uses the `add_coloring` method to compute the average value
            of each column for the points covered by each ball.

        metric : str, or callable, default='euclidean'
            The metric to use when calculating distance between instances in a
            feature array.
            If metric is 'precomputed', X is assumed to be a distance matrix and
            must be square.

        order: array-like of shape (n_samples, ), default=None
            The order in which to consider the data points in the greedy \
            search for landmarks. Different ordering might lead to different \
            BallMapper graphs.
            By defaults uses the order of X.

        method: string, default=None
            The method used to select the landmark points. Options are:
            - None or "greedy": selects the first uncovered point in the \
            considered order (default).
            - "balltree": a fast BallTree-based version of the greedy method \
            (euclidean metric, no orbits) that produces an identical BallMapper \
            graph, recommended for large datasets.
            - "gpu": runs the landmark search on a CUDA device via pytorch \
            (euclidean metric, no orbits). It does not reproduce the greedy \
            landmark sequence -- it returns a different, equally valid maximal \
            eps-net -- but it is by far the fastest option on large datasets. \
            Accepts the `device`, `chunk` and `lblk` keywords. Falls back to \
            "balltree" when pytorch or a CUDA device is unavailable.
            - "nearest": deterministic method selecting the uncovered point \
            nearest to any existing ball.
            - "adaptive": adjusts the radius of each ball to enforce a maximum \
            number of points per ball (requires the `max_size` keyword).

        verbose: bool or string, default=False
            Enable verbose output. Set it to 'tqdm' to show a tqdm progressbar.

        column_names: list of strings, default=None
            names of the columns in X, used for labeling purposes.
            If not given, names [x1, x2, ..., xd] assigned.

        Attributes
        ------------

        Graph: NetworkX Graph object
            The BallMapper graph. Each node correspond to a covering ball and has attributes: \
            'landmark' the id of the corresponding landmark point \
            'points covered' the ids of the points covered by the corresponding ball

        eps: float
            The input radius of the balls.

        landmarks_data: {array-like, sparse matrix} of shape (len(landmarks), n_features)
            landmark points selected from the input data X

        points_covered_by_landmarks: dict
            keys: landmarks ids \
            values: list of ids of the points covered by the corresponding ball

        Notes
        ----------
        https://arxiv.org/abs/1901.07410

        """

        self.eps: float = eps

        # If column names not given, [x1, x2, ..., xd] assigned
        if column_names is not None:
            self.column_names: list[str] = column_names
        else:
            self.column_names = ["x{}".format(i) for i in range(X.shape[1])]

        if not isinstance(X, np.ndarray):
            try:
                X = np.asanyarray(X, dtype=float)
            except (TypeError, ValueError):
                warnings.warn(
                    "the input is {} - cannot convert it to numpy array".format(type(X))
                )

        n_points = X.shape[0]

        # convert order to a list
        if order is None:
            order = range(n_points)

        elif isinstance(order, np.ndarray):
            order = order.tolist()

        elif not isinstance(order, list):
            warnings.warn(
                "Warning........... order is not a list or numpy array, using default ordering"
            )
            order = range(n_points)

        # check whether order is a list of lenght = len(points)
        # otherwise use the defaut ordering
        if len(np.unique(order)) != n_points:
            warnings.warn(
                "Warning........... order is not compatible with points, using default ordering"
            )
            order = range(n_points)

        # find landmarks
        landmarks, self.points_covered_by_landmarks, self.eps_dict = _find_landmarks(
            X, eps, orbits, metric, order, method, verbose, **kwargs
        )

        # store landmarks points (centers of the balls)
        self.landmarks_data: pd.DataFrame = pd.DataFrame(
            X[list(landmarks.values()), :], columns=self.column_names
        )

        # find edges
        if verbose:
            print("Running BallMapper ")
            print("Finding edges...")
        edges: list[list[int]] = []  # list of edges [[idx_v, idx_u], ...]
        if method in ("balltree", "gpu"):
            # fast sparse edge finding; both methods return coverage lists that
            # are already sorted, which is all this needs
            edges = _find_edges_from_coverage(
                self.points_covered_by_landmarks, n_points
            )
        else:
            for i, idx_v in tqdm(
                enumerate(list(landmarks.keys())[:-1]), disable=not (verbose == "tqdm")
            ):
                for idx_u in list(landmarks.keys())[i + 1 :]:
                    if (
                        len(
                            set(self.points_covered_by_landmarks[idx_v]).intersection(
                                self.points_covered_by_landmarks[idx_u]
                            )
                        )
                        != 0
                    ):
                        edges.append([idx_v, idx_u])

        # create Ball Mapper graph
        if verbose:
            print("Creating Ball Mapper graph...")
        self.Graph: nx.Graph = nx.Graph()
        self.Graph.add_nodes_from(landmarks.keys())
        self.Graph.add_edges_from(edges)

        for node in self.Graph.nodes:
            self.Graph.nodes[node]["landmark"] = landmarks[node]
            self.Graph.nodes[node]["points covered"] = np.array(
                self.points_covered_by_landmarks[node]
            )
            self.Graph.nodes[node]["size"] = len(
                self.Graph.nodes[node]["points covered"]
            )

        if isinstance(coloring_df, pd.DataFrame):
            if verbose:
                print("Computing coloring")
            self.add_coloring(coloring_df)

        if verbose:
            print("Done")

    def add_coloring(
        self,
        coloring_df: pd.DataFrame,
        custom_function: Callable[..., Any] = np.mean,
        custom_name: str | None = None,
        add_std: bool = False,
    ) -> None:
        """Takes pandas dataframe and compute the average and standard deviation \
        of each column for the subset of points colored by each ball.
        Add such values as attributes to each node in the BallMapper graph

        Parameters
        ----------
        coloring_df: pandas dataframe of shape (n_samples, n_coloring_function)
        custom_function : callable, optional
            a function to compute on the `coloring_df` columns, by default numpy.mean
        custom_name : string, optional
            sets the attributes naming scheme, by default None, the attribute names will be the column names
        add_std: bool, default=False
            Wheter to compute also the standard deviation on each ball
        """
        # for each column in the dataframe compute the mean across all nodes and add it as mean attributes
        for node in self.Graph.nodes:
            for col_name, avg in (
                coloring_df.loc[self.Graph.nodes[node]["points covered"]]
                .apply(custom_function, axis=0)
                .items()
            ):
                if custom_name:
                    name = "{}_{}".format(col_name, custom_name)
                else:
                    name = col_name
                self.Graph.nodes[node][name] = avg
            # option to add the standar deviation on each node
            if add_std:
                for col_name, std in (
                    coloring_df.loc[self.Graph.nodes[node]["points covered"]]
                    .std()
                    .items()
                ):
                    self.Graph.nodes[node]["{}_std".format(col_name)] = std

    def color_by_variable(
        self,
        my_variable: str | None,
        my_palette: Colormap,
        MIN_VALUE: float = np.inf,
        MAX_VALUE: float = -np.inf,
    ) -> tuple[float, float]:
        """Colors the BallMapper graph using a specified variable. The `add_coloring` method needs to be called first. Automatically computes the min and max value for the colormap.

        Parameters
        ----------
        my_variable : string
            the variable to color by
        my_palette : matplotlib.colors.Colormap
            a valid colormap
        MIN_VALUE : float, optional
            the value to be assigned to the lowest color in the cmap, by default np.inf
        MAX_VALUE : float, optional
            the value to be assigned to the highest color in the cmap, by default -np.inf

        Returns
        -------
        MIN_VALUE, MAX_VALUE
            the computed min and max values of `my_variable` on the BM nodes, useful to set the limits for a colorbar
        """

        # get the coloring variables
        for k in self.Graph.nodes:
            node_keys = self.Graph.nodes[k].keys()
            break

        if my_variable is None:
            for node in self.Graph.nodes:
                self.Graph.nodes[node]["color"] = cm.get_cmap("tab10")(0)

        elif my_variable not in node_keys:  # TODO find a better way to check
            warnings.warn(
                "Warning........... {} is not a valid coloring, add it using the `add_coloring` method".format(
                    my_variable
                )
            )

        else:
            for node in self.Graph.nodes:
                if self.Graph.nodes[node][my_variable] > MAX_VALUE:
                    MAX_VALUE = self.Graph.nodes[node][my_variable]
                if self.Graph.nodes[node][my_variable] < MIN_VALUE:
                    MIN_VALUE = self.Graph.nodes[node][my_variable]

            for node in self.Graph.nodes:
                if not pd.isna(self.Graph.nodes[node][my_variable]):
                    color_id = (self.Graph.nodes[node][my_variable] - MIN_VALUE) / (
                        MAX_VALUE - MIN_VALUE
                    )
                    self.Graph.nodes[node]["color"] = my_palette(color_id)
                else:
                    self.Graph.nodes[node]["color"] = "black"

        return MIN_VALUE, MAX_VALUE

    def filter_by(self, list_of_points: list[int]) -> BallMapper:
        """return a copy of the BallMapper object with only the nodes covering a subset of points

        Parameters
        ----------
        list_of_points : list
            list of the subset of points to keep

        Returns
        -------
        BallMapper
            the filtered BallMapper graph
        """

        filtered_bm = copy.deepcopy(self)

        for node in filtered_bm.Graph.nodes:
            filtered_bm.points_covered_by_landmarks[node] = list(
                set(filtered_bm.points_covered_by_landmarks[node]).intersection(
                    list_of_points
                )
            )
            filtered_bm.Graph.nodes[node]["points covered"] = np.array(
                filtered_bm.points_covered_by_landmarks[node]
            )

            filtered_bm.Graph.nodes[node]["size"] = len(
                filtered_bm.Graph.nodes[node]["points covered"]
            )

        filtered_bm.Graph.remove_nodes_from(
            [
                node
                for node in filtered_bm.Graph
                if filtered_bm.Graph.nodes[node]["size"] == 0
            ]
        )

        return filtered_bm

    def points_and_balls(self) -> pd.DataFrame:
        """returns a DataFrame with the `points_covered_by_landmarks` information

        Returns
        -------
        pandas.DataFrame

        """
        to_df: list[list[int]] = []
        for ball, points in self.points_covered_by_landmarks.items():
            for p in points:
                to_df.append([p, ball])

        return pd.DataFrame(to_df, columns=["point", "ball"])

    def ball_data(
        self, X: npt.NDArray, ball_numbers: int | list[int]
    ) -> dict[int, pd.DataFrame]:
        """returns the data points corresponding to the specified ball numbers

        Parameters
        ----------
        ball_numbers : list
            list of ball numbers

        Returns
        -------
        dict
            keys: ball numbers
            values: pandas DataFrame with the data points corresponding to the ball
        """

        nodes_number = len(self.Graph.nodes)

        if isinstance(ball_numbers, int):
            ball_numbers = [ball_numbers]

        if np.max(np.array(ball_numbers)) >= nodes_number:
            raise Exception(
                "Incorrect ball number(s). The ball numbers should be in the range [0, {}]".format(
                    nodes_number - 1
                )
            )

        pab = self.points_and_balls()
        ball_data_frames: dict[int, pd.DataFrame] = {}
        for ball_number in ball_numbers:
            df_of_a_ball = pd.DataFrame(
                X[pab[pab["ball"] == ball_number]["point"], :],
                columns=self.column_names,
            )
            ball_data_frames[ball_number] = df_of_a_ball

        return ball_data_frames

    def ball_data_index(self, ball_numbers: int | list[int]) -> dict[int, list[int]]:
        """returns the indices of data points corresponding to the specified ball numbers

        Parameters
        ----------
        ball_numbers : list
            list of ball numbers

        Returns
        -------
        dict
            keys: ball numbers
            values: list of indices of data points corresponding to the ball
        """

        nodes_number = len(self.Graph.nodes)

        if isinstance(ball_numbers, int):
            ball_numbers = [ball_numbers]

        if np.max(np.array(ball_numbers)) >= nodes_number:
            raise Exception(
                "Incorrect ball number(s). The ball numbers should be in the range [0, {}]".format(
                    nodes_number - 1
                )
            )

        pab = self.points_and_balls()
        ball_points_indices_lists: dict[int, list[int]] = {}
        for ball_number in ball_numbers:
            list_of_point_indices = list(pab[pab["ball"] == ball_number]["point"])
            ball_points_indices_lists[ball_number] = list_of_point_indices

        return ball_points_indices_lists

    def draw_networkx(
        self,
        coloring_variable: str | None = None,
        color_palette: Colormap | None = None,
        colorbar: bool = False,
        colorbar_label: str | None = None,
        ax: plt.Axes | None = None,
        MIN_VALUE: float = np.inf,
        MAX_VALUE: float = -np.inf,
        MIN_SCALE: int = 100,
        MAX_SCALE: int = 600,
        pos: dict[int, tuple[float, float]] | None = None,
        **kwargs: Any,
    ) -> plt.Axes:
        """Wrapper around the `networkx.draw_networkx` method with colorbar support.

        Parameters
        ----------
        coloring_variable : string, optional
            the variable to use for coloring the BM graph, by default None
        color_palette : matplotlib.colors.Colormap, optional
            the coloring palette to use, by default cm.get_cmap("Reds")
        colorbar : bool, optional
            the label on the colorbar's long axis.
        colorbar_label : str, optional
            whether to add a colorbar to the plot, by default False
        ax : matplotlib.axes.Axes, optional
            the matplotlib ax where to plot the graph. If None, the current ax is used. By default None
        MIN_VALUE : float, optional
            the value to be assigned to the lowest color in the cmap, by default np.inf
        MAX_VALUE : float, optional
            the value to be assigned to the highest color in the cmap, by default -np.inf
        MIN_SCALE : int, optional
            the minimum radius for the nodes, by default 100
        MIN_SCALE : int, optional
            the maximum radius for the nodes, by default 100
        pos : dictionary, optional
            A dictionary with nodes as keys and positions as values. If not specified a spring layout positioning will be computed. See `networkx.drawing.layout` for functions that compute node positions. By default None

        Returns
        -------
        ax
            the matplotlib ax
        """
        if color_palette is None:
            color_palette = cm.get_cmap("Reds")

        MAX_NODE_SIZE = max(
            [self.Graph.nodes[node]["size"] for node in self.Graph.nodes]
        )

        if ax is None:
            ax = plt.gca()

        MIN_VALUE, MAX_VALUE = self.color_by_variable(
            coloring_variable, color_palette, MIN_VALUE, MAX_VALUE
        )

        if pos is None:
            pos = nx.spring_layout(self.Graph, seed=24)

        nx.draw_networkx(
            self.Graph,
            pos=pos,
            node_color=[self.Graph.nodes[node]["color"] for node in self.Graph.nodes],
            node_size=[
                MAX_SCALE * self.Graph.nodes[node]["size"] / MAX_NODE_SIZE + MIN_SCALE
                for node in self.Graph.nodes
            ],
            alpha=0.8,
            ax=ax,
            **kwargs,
        )

        # plot a legend
        if colorbar:
            sm = plt.cm.ScalarMappable(
                cmap=color_palette,
                norm=plt.Normalize(
                    vmin=MIN_VALUE,
                    vmax=MAX_VALUE,
                ),
            )
            plt.colorbar(sm, label=colorbar_label, ax=ax)

        return ax
