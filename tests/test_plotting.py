from __future__ import annotations

from pyballmapper.plotting import generate_partitions


class TestGeneratePartitions:
    def test_basic_partition_count(self):
        node_contents = {
            0: [0, 1, 2, 3],
            1: [2, 3, 4],
            2: [5],
        }
        original_element_values = {
            0: "a",
            1: "a",
            2: "b",
            3: "b",
            4: "c",
            5: "a",
        }
        result = generate_partitions(node_contents, original_element_values)
        assert result == {0: {"a": 2, "b": 2}, 1: {"b": 2, "c": 1}, 2: {"a": 1}}

    def test_empty_node(self):
        node_contents = {0: []}
        original_element_values = {}
        result = generate_partitions(node_contents, original_element_values)
        assert result == {0: {}}

    def test_single_class(self):
        node_contents = {0: [0, 1, 2]}
        original_element_values = {0: "x", 1: "x", 2: "x"}
        result = generate_partitions(node_contents, original_element_values)
        assert result == {0: {"x": 3}}
