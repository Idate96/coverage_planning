from collections import defaultdict

from numpy.lib.shape_base import expand_dims
from cpp.dfs_tree import Graph, adj_matrix_to_dict
from cpp.bsd import create_global_adj_matrix
import numpy as np
import matplotlib.image as mpimg
from typing import Tuple, List


def create_graph():
    graph = Graph()
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 3)
    graph.add_edge(1, 4)
    graph.add_edge(2, 5)
    graph.add_edge(3, 6)
    return graph


def create_graph_2():
    graph = Graph()
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 3)
    graph.add_edge(1, 4)
    graph.add_edge(2, 5)
    graph.add_edge(3, 6)
    graph.add_edge(6, 4)
    graph.add_edge(4, 2)
    return graph


def create_graph_undirected():
    graph = Graph()
    graph.add_edge(0, 1, directed=False)
    graph.add_edge(0, 2, directed=False)
    graph.add_edge(1, 3, directed=False)
    graph.add_edge(1, 4, directed=False)
    graph.add_edge(2, 5, directed=False)
    graph.add_edge(3, 6, directed=False)
    return graph


def create_graph_undirected_2():
    graph = Graph()
    graph.add_edge(0, 1, directed=False)
    graph.add_edge(0, 2, directed=False)
    graph.add_edge(1, 3, directed=False)
    graph.add_edge(1, 4, directed=False)
    graph.add_edge(2, 5, directed=False)
    graph.add_edge(3, 6, directed=False)
    graph.add_edge(6, 4, directed=False)
    graph.add_edge(4, 2, directed=False)
    return graph


def test_graph():
    graph = create_graph()
    assert graph.adj[0] == [1, 2]


def test_dfs():
    graph = create_graph()
    assert graph.DFS(0) == [0, 1, 3, 6, 4, 2, 5]


def test_dfs_undirected():
    graph = create_graph_undirected()
    assert graph.DFS(0) == [0, 1, 3, 6, 4, 2, 5]


def test_dfs_tree():
    graph = create_graph()
    assert graph.DFS_tree(0) == [(0, 1), (1, 3), (3, 6), (1, 4), (0, 2), (2, 5)]


def test_dfs_tree_undirected():
    graph = create_graph_undirected()
    assert graph.DFS_tree(0) == [(0, 1), (1, 3), (3, 6), (1, 4), (0, 2), (2, 5)]


def test_dfs_graph():
    graph = create_graph_2()
    assert graph.DFS_tree(0) == [(0, 1), (1, 3), (3, 6), (6, 4), (4, 2), (2, 5)]


def test_dfs_graph_undirected():
    graph = create_graph_undirected_2()
    assert graph.DFS_tree(0) == [(0, 1), (1, 3), (3, 6), (6, 4), (4, 2), (2, 5)]


def test_find_height():
    graph = create_graph_2()
    tree_edges = graph.DFS_tree(0)
    graph_tree = Graph()
    graph_tree.add_edges(tree_edges)
    assert graph_tree.find_height(0) == (6, 5)


def test_find_diameter():
    tree = create_graph_undirected()
    assert tree.find_diameter(0) == (5, 6, 5)


def test_post_order_traversal():
    graph = create_graph()
    assert graph.post_order_traversal(0) == [6, 3, 4, 1, 5, 2, 0]


def test_matrix_to_dict():
    matrix = np.zeros((4, 4), dtype=int)
    matrix[0, 1:3] = 1
    matrix[1, 3] = 1
    expected_dict = defaultdict(list)
    expected_dict[0] = [1, 2]
    expected_dict[1] = [3]
    expected_dict[2] = []
    expected_dict[3] = []
    graph_dict = adj_matrix_to_dict(matrix)
    assert graph_dict == expected_dict


def test_create_graph_from_image():
    image = mpimg.imread("data/test/map.jpg")
    # original image is black and white anyway
    binary_image = image[:, :, 0] > 150
    graph_adj_matrix = create_global_adj_matrix(binary_image)
    graph_adj_dict = adj_matrix_to_dict(graph_adj_matrix)
    
    graph = Graph(graph_adj_dict)
    tree_edges = graph.DFS_tree(0)
    graph_tree = Graph()
    graph_tree.add_edges(tree_edges, directed=False)
    diameter, new_root, furthest_node = graph_tree.find_diameter(0)
    
    tree_edges = graph_tree.DFS_tree(new_root)
    longest_tree = Graph()
    longest_tree.add_edges(tree_edges)
    visited = longest_tree.post_order_traversal(new_root)
    expected_visit = [0, 1, 5, 3, 2, 4, 6, 7, 9, 8]
    assert visited == expected_visit


def add_num_to_list(list: List[int], num: int):  # O(n)
    for i in range(len(list)):
        list[i] += num
    return list


if __name__ == "__main__":
    test_create_graph_from_image()
