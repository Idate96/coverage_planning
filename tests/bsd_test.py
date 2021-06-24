import numpy as np
from cpp.bsd import find_connectivity, find_slices_adjacency, create_mask
import matplotlib.image as mpimg

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


def create_mask_test():
    image = mpimg.imread("data/test/map.jpg")
    # original image is black and white anyway
    print(image.shape)
    binary_image = image[:, :, 0] > 150
    print(np.min(binary_image))
    mask = create_mask(binary_image)
    print(np.max(mask))


if __name__ == "__main__":
    create_mask_test()
