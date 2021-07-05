import numpy as np
from cpp.bsd import create_global_adj_matrix, find_connectivity, find_slices_adjacency, create_mask
import matplotlib.image as mpimg
from cpp.helpers import *

# tests for connectivy
def test_fully_connected():
    slice = np.ones((10, 1))
    segments = find_connectivity(slice)
    assert segments == [(0, 10)]


def test_one_segment():
    slice = np.zeros((15, 1))
    slice[:10] = 1
    segments = find_connectivity(slice)
    assert segments == [(0, 10)]


def test_two_segments():
    slice = np.zeros((15, 1))
    slice[:5] = 1
    slice[11:] = 1
    segments = find_connectivity(slice)
    assert segments == [(0, 5), (11, 15)]


def test_empty_seg_adj():
    segments_0 = []
    segments_1 = [(0, 1)]
    adj_m = find_slices_adjacency(segments_0, segments_1)
    # assert adj_m == np.array([], dtype=bool)d


# test for adjancency
def test_single_seg_adjacency():
    segments_0 = [(0, 1)]
    segments_1 = [(0, 1)]
    adj_m = find_slices_adjacency(segments_0, segments_1)
    assert adj_m == np.ones((1, 1), dtype=bool)


def test_false_adjancency():
    segments_0 = [(0, 1)]
    segments_1 = [(2, 4)]
    adj_m = find_slices_adjacency(segments_0, segments_1)
    assert adj_m == np.zeros((1, 1), dtype=bool)


def test_two_adj():
    segments_0 = [(0, 2), (7, 10)]
    segments_1 = [(1, 3), (5, 7)]
    target_adj = np.array(([1, 0], [0, 0]), dtype=bool)
    adj_m = find_slices_adjacency(segments_0, segments_1)
    assert np.allclose(adj_m, target_adj)


def test_create_mask():
    image = mpimg.imread("data/test/map.jpg")
    # original image is black and white anyway
    binary_image = image[:, :, 0] > 150
    mask = create_mask(binary_image)


def test_find_connectivity():
    image = mpimg.imread("data/test/map.jpg")
    # original image is black and white anyway
    binary_image = image[:, :, 0] > 150
    expected_graph = np.zeros((10, 10))
    expected_graph[0, 1] = 1
    expected_graph[0, 2] = 1
    expected_graph[1, 5] = 1
    expected_graph[2, 3] = 1
    expected_graph[2, 4] = 1
    expected_graph[3, 5] = 1
    expected_graph[4, 6] = 1
    expected_graph[5, 6] = 1
    expected_graph[6, 7] = 1
    expected_graph[6, 8] = 1
    expected_graph[7, 9] = 1
    expected_graph[8, 9] = 1
    expected_graph = expected_graph + expected_graph.T
    graph = create_global_adj_matrix(binary_image)
    assert np.allclose(graph, expected_graph)

if __name__ == "__main__":
    test_find_connectivity()
