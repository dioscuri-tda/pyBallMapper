from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyballmapper import BallMapper


@pytest.fixture
def simple_2d() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((20, 2))


@pytest.fixture
def three_clusters_2d() -> np.ndarray:
    rng = np.random.default_rng(42)
    return np.vstack(
        [
            rng.normal(loc=(0, 0), scale=0.1, size=(10, 2)),
            rng.normal(loc=(3, 3), scale=0.1, size=(10, 2)),
            rng.normal(loc=(6, 0), scale=0.1, size=(10, 2)),
        ]
    )


@pytest.fixture
def single_point() -> np.ndarray:
    return np.array([[1.0, 2.0]])


@pytest.fixture
def distance_matrix(simple_2d: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import cdist

    return cdist(simple_2d, simple_2d, metric="euclidean")


@pytest.fixture
def coloring_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "color_a": np.random.default_rng(0).random(20),
            "color_b": np.random.default_rng(1).random(20),
        }
    )


@pytest.fixture
def bm_instance(simple_2d: np.ndarray) -> BallMapper:
    return BallMapper(simple_2d, eps=0.3, verbose=False)


@pytest.fixture
def bm_with_coloring(simple_2d: np.ndarray, coloring_df: pd.DataFrame) -> BallMapper:
    bm = BallMapper(simple_2d, eps=0.3, verbose=False, coloring_df=coloring_df)
    return bm
