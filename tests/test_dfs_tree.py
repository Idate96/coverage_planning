from collections import defaultdict

from numpy.lib.shape_base import expand_dims
from cpp.dfs_tree import Graph, adj_matrix_to_dict
from cpp.bsd import *
import numpy as np
import matplotlib.image as mpimg
from cpp.dfs_tree import *
from typing import Tuple, List
from cpp.helpers import *
from cpp.cells import *


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
    graph_adj_matrix, _ = create_global_adj_matrix(binary_image)
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


def test_create_arcs_from_image():
    binary_image = recurvise_H_map((400, 600), invert=True, num_recursions=0)
    graph_adj_matrix, decomposed_im = get_directed_global_adj_matrix(binary_image)
    graph_adj_dict = adj_matrix_to_dict(graph_adj_matrix)
    graph = Graph(graph_adj_dict)
    arc_edges = graph.graph_to_arcs()
    list_expected_arcs = [
        Arc(tail=0, weight=1, head=1),
        Arc(tail=0, weight=1, head=3),
        Arc(tail=1, weight=1, head=0),
        Arc(tail=1, weight=1, head=5),
        Arc(tail=2, weight=1, head=0),
        Arc(tail=3, weight=1, head=0),
        Arc(tail=3, weight=1, head=5),
        Arc(tail=4, weight=1, head=5),
        Arc(tail=5, weight=1, head=1),
        Arc(tail=5, weight=1, head=3),
    ]
    for arc in arc_edges:
        assert arc in list_expected_arcs


def test_create_arcs_dfs():
    binary_image = recurvise_H_map((400, 600), invert=True, num_recursions=0)
    graph_adj_matrix, decomposed_im = get_directed_global_adj_matrix(binary_image)
    graph_adj_dict = adj_matrix_to_dict(graph_adj_matrix)
    graph = Graph(graph_adj_dict)
    arc_edges = graph.graph_to_arcs()
    arc_edges_dsf = graph.graph_to_arcs_with_dfs(0)


    for arc in arc_edges:
        assert arc in arc_edges_dsf
    for arc in arc_edges_dsf:
        assert arc in arc_edges


def test_create_graph_from_edges():
    binary_image = recurvise_H_map((400, 600), invert=True, num_recursions=0)
    graph_adj_matrix, decomposed_im = get_directed_global_adj_matrix(binary_image)
    graph_adj_dict = adj_matrix_to_dict(graph_adj_matrix)
    graph = Graph(graph_adj_dict)

    arc_edges = graph.graph_to_arcs()
    graph_from_arcs = Graph()
    graph_from_arcs.arcs_to_graph(arc_edges)

    assert graph_from_arcs.adj == graph.adj


def add_num_to_list(list: List[int], num: int):  # O(n)
    for i in range(len(list)):
        list[i] += num
    return list


def test_minimum_spanning_tree_directed_graph():
    binary_image = recurvise_H_map((400, 600), invert=True, num_recursions=0)
    graph_adj_matrix, decomposed_im = get_directed_global_adj_matrix(binary_image)
    graph_adj_dict = adj_matrix_to_dict(graph_adj_matrix)
    graph = Graph(graph_adj_dict)
    arc_edges = graph.graph_to_arcs()
    arc_edges_dfs = [
        Arc(tail=0, weight=1, head=1),
        Arc(tail=1, weight=1, head=5),
        Arc(tail=5, weight=1, head=3),
        Arc(tail=3, weight=1, head=0),
        Arc(tail=0, weight=1, head=3),
        Arc(tail=3, weight=1, head=5),
        Arc(tail=5, weight=1, head=1),
        Arc(tail=1, weight=1, head=0),
        Arc(tail=2, weight=1, head=0),
        Arc(tail=4, weight=1, head=5),
    ]

    print(arc_edges)
    mst = min_spanning_arborescence(arc_edges, 0)
    mst_dfs = min_spanning_arborescence(arc_edges_dfs, 0)
    print(mst)
    print(mst_dfs)

def test_mst_dfs_directed_graph():
    binary_image = recurvise_H_map((400, 600), invert=True, num_recursions=0)
    graph_adj_matrix, decomposed_im = get_directed_global_adj_matrix(binary_image)
    graph_adj_dict = adj_matrix_to_dict(graph_adj_matrix)
    graph = Graph(graph_adj_dict)
    start_node = 0
    arc_edges = graph.graph_to_arcs_with_dfs(start_node)
    mst = min_spanning_arborescence(arc_edges, start_node)
    # dfs like tree 
    tree = {2: Arc(tail=2, weight=1, head=0), 3: Arc(tail=3, weight=1, head=0), 5: Arc(tail=5, weight=1, head=3), 4: Arc(tail=4, weight=1, head=5), 1: Arc(tail=1, weight=1, head=5)}
    assert mst == tree

def test_reverse_graph():
    binary_image = recurvise_H_map((400, 600), invert=True, num_recursions=0)
    graph_adj_matrix, decomposed_im = get_directed_global_adj_matrix(binary_image)
    graph_adj_dict = adj_matrix_to_dict(graph_adj_matrix)
    graph = Graph(graph_adj_dict)
    start_node = 0
    arc_edges = graph.graph_to_arcs_with_dfs(start_node)
    mst_arcs = min_spanning_arborescence(arc_edges, start_node).values()
    mst = Graph()
    mst.arcs_to_graph(mst_arcs)
    mst.reverse_graph_edges()
    expected_dict = {2: [], 0: [2, 3], 3: [5], 5: [4, 1], 4: [], 1: []}
    assert mst.adj == expected_dict

def test_post_order_traversal_mst():
    binary_image = recurvise_H_map((400, 600), invert=True, num_recursions=0)
    graph_adj_matrix, decomposed_im = get_directed_global_adj_matrix(binary_image)
    graph_adj_dict = adj_matrix_to_dict(graph_adj_matrix)
    graph = Graph(graph_adj_dict)
    start_node = 0
    arc_edges = graph.graph_to_arcs_with_dfs(start_node)
    mst_arcs = min_spanning_arborescence(arc_edges, start_node).values()
    mst = Graph()
    mst.arcs_to_graph(mst_arcs)
    mst.reverse_graph_edges()
    post_order_traversal = mst.post_order_traversal(start_node)
    expected_post_order_traversal = [2, 4, 1, 5, 3, 0]
    assert post_order_traversal == expected_post_order_traversal

if __name__ == "__main__":
    test_post_order_traversal_mst()
