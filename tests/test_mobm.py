from __future__ import annotations

import numpy as np
import pytest

from pyballmapper import BallMapper
from pyballmapper.mobm import MapperonBallMapper


@pytest.fixture
def cover_bm(simple_2d: np.ndarray) -> BallMapper:
    return BallMapper(simple_2d, eps=0.3, verbose=False)


class TestMapperonBallMapper:
    def test_basic_construction(self, cover_bm, three_clusters_2d):
        mobm = MapperonBallMapper(
            cover_BM=cover_bm,
            target_space=three_clusters_2d,
            eps=0.2,
            min_samples=1,
            dbg=False,
        )
        assert hasattr(mobm, "Graph")
        assert len(mobm.Graph.nodes) > 0

    def test_node_attributes(self, cover_bm, three_clusters_2d):
        mobm = MapperonBallMapper(
            cover_BM=cover_bm,
            target_space=three_clusters_2d,
            eps=0.2,
            min_samples=1,
            dbg=False,
        )
        for node in mobm.Graph.nodes:
            assert "points covered" in mobm.Graph.nodes[node]
            assert "size" in mobm.Graph.nodes[node]

    def test_sparse_option(self, cover_bm, three_clusters_2d):
        mobm = MapperonBallMapper(
            cover_BM=cover_bm,
            target_space=three_clusters_2d,
            eps=0.2,
            min_samples=1,
            sparse=True,
            dbg=False,
        )
        assert len(mobm.Graph.nodes) > 0

    def test_labels_format(self, cover_bm, three_clusters_2d):
        mobm = MapperonBallMapper(
            cover_BM=cover_bm,
            target_space=three_clusters_2d,
            eps=0.2,
            min_samples=1,
            dbg=False,
        )
        for node in mobm.Graph.nodes:
            assert isinstance(node, int)
            assert "label" in mobm.Graph.nodes[node]
