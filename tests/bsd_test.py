from matplotlib.pyplot import show
import numpy as np
from cpp.bsd import *
from cpp.dfs_tree import *
from cpp.cells import *
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


def test_create_mask_H():
    binary_image = generate_H_map((400, 600), invert=True)
    decomposed_image = create_mask(binary_image)
    cells = Cell.from_image(decomposed_image)
    plot_decomposed_image(decomposed_image, show=False)
    plot_cells(cells, show=False)


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
    graph, _ = create_global_adj_matrix(binary_image)
    assert np.allclose(graph, expected_graph)


def test_get_adj_cell_to_corner():
    """Tests whethere the corners found are the correct ones"""
    image = mpimg.imread("data/test/map.jpg")
    # original image is black and white anyway
    binary_image = image[:, :, 0] > 150
    graph_adj_matrix, _ = create_global_adj_matrix(binary_image)

    # create cells
    decomposed_image = create_mask(binary_image)
    cells = Cell.from_image(decomposed_image)

    # find adjent corners
    adj_cell_6_0 = get_adj_cell_to_corner(cells, cell_id=6, corner_id=0)
    assert adj_cell_6_0 == 5

    adj_cell_6_1 = get_adj_cell_to_corner(cells, cell_id=6, corner_id=1)
    assert adj_cell_6_1 == 4

    adj_cell_6_2 = get_adj_cell_to_corner(cells, cell_id=6, corner_id=2)
    assert adj_cell_6_2 == 8

    adj_cell_6_3 = get_adj_cell_to_corner(cells, cell_id=6, corner_id=3)
    assert adj_cell_6_3 == 7

    adj_cell_3_0 = get_adj_cell_to_corner(cells, cell_id=3, corner_id=0)
    assert adj_cell_3_0 == 2

    adj_cell_3_1 = get_adj_cell_to_corner(cells, cell_id=3, corner_id=1)
    assert adj_cell_3_1 == 2

    adj_cell_3_2 = get_adj_cell_to_corner(cells, cell_id=3, corner_id=2)
    assert adj_cell_3_2 == 5

    adj_cell_3_3 = get_adj_cell_to_corner(cells, cell_id=3, corner_id=3)
    assert adj_cell_3_3 == 5

    # test corners
    adj_cell_0_0 = get_adj_cell_to_corner(cells, cell_id=0, corner_id=0)
    assert adj_cell_0_0 == None

    adj_cell_0_1 = get_adj_cell_to_corner(cells, cell_id=0, corner_id=1)
    assert adj_cell_0_1 == None

    adj_cell_9_2 = get_adj_cell_to_corner(cells, cell_id=9, corner_id=2)
    assert adj_cell_9_2 == None

    adj_cell_9_3 = get_adj_cell_to_corner(cells, cell_id=9, corner_id=3)
    assert adj_cell_9_3 == None


def test_corner_adjencency():
    """Tests whethere the corner adj is correct found are the correct ones"""
    image = mpimg.imread("data/test/map.jpg")
    # original image is black and white anyway
    binary_image = image[:, :, 0] > 150
    graph_adj_matrix = create_global_adj_matrix(binary_image)

    # create cells
    decomposed_image = create_mask(binary_image)
    cells = Cell.from_image(decomposed_image)

    corner_adj = corner_adjencency(cells[0], cells[1])
    assert corner_adj == True


def test_corner_adjencency_H_shape():
    binary_image = generate_H_map((400, 600), invert=True)
    decomposed_image = create_mask(binary_image)
    cells = Cell.from_image(decomposed_image)
    plot_decomposed_image(decomposed_image)
    plot_cells(cells, show=False)

    corner_adj_01 = corner_adjencency(cells[0], cells[1])
    corner_adj_10 = corner_adjencency(cells[1], cells[0])
    corner_adj_02 = corner_adjencency(cells[0], cells[2])
    corner_adj_20 = corner_adjencency(cells[2], cells[0])

    assert corner_adj_01 == True
    assert corner_adj_10 == True
    assert corner_adj_02 == False
    assert corner_adj_20 == True


def test_directed_global_adj_matrix_concave_obstacles():
    image = recurvise_H_map((400, 600), invert=True, num_recursions=0)

    # original image is black and white anyway
    graph_adj_matrix, decomposed_image = create_global_adj_matrix(image)
    directed_global_adj_matrix, decomposed_image = get_directed_global_adj_matrix(image)
    cells = Cell.from_image(decomposed_image)

    assert np.isclose(graph_adj_matrix[0, 1], directed_global_adj_matrix[0, 1])
    assert np.isclose(graph_adj_matrix[1, 0], directed_global_adj_matrix[1, 0])
    assert np.isclose(graph_adj_matrix[1, 2], directed_global_adj_matrix[1, 2])
    assert np.isclose(graph_adj_matrix[2, 1], directed_global_adj_matrix[2, 1])
    assert np.isclose(graph_adj_matrix[2, 3], directed_global_adj_matrix[2, 3])
    assert np.isclose(graph_adj_matrix[3, 2], directed_global_adj_matrix[3, 2])
    assert np.isclose(graph_adj_matrix[3, 4], directed_global_adj_matrix[3, 4])
    assert np.isclose(graph_adj_matrix[4, 3], directed_global_adj_matrix[4, 3])
    assert np.isclose(graph_adj_matrix[4, 5], directed_global_adj_matrix[4, 5])
    assert not np.isclose(graph_adj_matrix[5, 4], directed_global_adj_matrix[5, 4])
    assert not np.isclose(graph_adj_matrix[0, 2], directed_global_adj_matrix[0, 2])
    assert np.isclose(graph_adj_matrix[2, 0], directed_global_adj_matrix[2, 0])

    plot_cells(cells)
    plot_decomposed_image(decomposed_image, show=False)


def test_directed_global_adj_matrix_convex_obstacles():
    image = mpimg.imread("data/test/map.jpg")
    # original image is black and white anyway
    binary_image = image[:, :, 0] > 150
    graph_adj_matrix, _ = create_global_adj_matrix(binary_image)
    directed_graph_adj_matrix, _ = get_directed_global_adj_matrix(binary_image)
    assert np.allclose(graph_adj_matrix, directed_graph_adj_matrix)


def test_plot_bsd_decomposition():
    image = mpimg.imread("data/test/map.jpg")
    # original image is black and white anyway
    binary_image = image[:, :, 0] > 150
    graph_adj_matrix, _ = create_global_adj_matrix(binary_image)

    # create cells
    decomposed_image = create_mask(binary_image)
    cells = Cell.from_image(decomposed_image)
    plot_cells(cells, show=False)
    plt.imshow(decomposed_image)
    plt.savefig("data/test/map_decomposed.pdf", dpi=300)


def test_plot_bsd_decomposition_concave():
    image = recurvise_H_map((400, 600), invert=True, num_recursions=0)
    # original image is black and white anyway
    graph_adj_matrix, _ = create_global_adj_matrix(image)

    # create cells
    decomposed_image = create_mask(image)
    cells = Cell.from_image(decomposed_image)
    plot_cells(cells, show=False)
    plt.imshow(decomposed_image)
    plt.savefig("data/test/map_h_decomposed.pdf", dpi=300)


def test_plot_bsd_reduced_image():
    image = mpimg.imread("data/test/map.jpg")
    # original image is black and white anyway
    binary_image = image[:, :, 0] > 150
    reduced_image = binary_image[:, :240]
    plt.imshow(reduced_image)
    plt.savefig("data/test/reduced_map.pdf", dpi=300)


def test_plot_bsd_reduced_decomp():
    image = mpimg.imread("data/test/map.jpg")
    # original image is black and white anyway
    binary_image = image[:, :, 0] > 150
    reduced_image = binary_image[:, :240]
    decomposed_image = create_mask(reduced_image)
    cells = Cell.from_image(decomposed_image)
    plot_cells(cells, show=False)
    plt.imshow(reduced_image)
    plt.savefig("data/test/reduced_decomposed_map.pdf", dpi=300)


def test_get_corner_to_adj_cell():
    """Tests whethere the corners found are the correct ones"""
    image = mpimg.imread("data/test/map.jpg")
    # original image is black and white anyway
    binary_image = image[:, :, 0] > 150
    graph_adj_matrix, _ = create_global_adj_matrix(binary_image)

    # create cells
    decomposed_image = create_mask(binary_image)
    cells = Cell.from_image(decomposed_image)

    # find adjent corners
    adj_corners_3_1 = get_corners_to_adj_cell(cells, current_cell_id=2, next_cell_id=0)
    assert adj_corners_3_1 == [0, 1]

    adj_corners_1_3 = get_corners_to_adj_cell(cells, current_cell_id=0, next_cell_id=2)
    assert adj_corners_1_3 == [2]

    adj_corners_1_2 = get_corners_to_adj_cell(cells, current_cell_id=0, next_cell_id=1)
    assert adj_corners_1_2 == [3]

    adj_corners_1_4 = get_corners_to_adj_cell(cells, current_cell_id=0, next_cell_id=3)
    assert adj_corners_1_4 == []


def test_create_path():
    image = mpimg.imread("data/test/map.jpg")
    binary_image = image[:, :, 0] > 150
    decomposed = create_mask(binary_image)
    cells = Cell.from_image(decomposed)

    path_00 = create_path(cells[0], 0, coverage_radius=10)
    path_01 = create_path(cells[0], 1, coverage_radius=10)
    path_02 = create_path(cells[0], 2, coverage_radius=10)
    path_03 = create_path(cells[0], 3, coverage_radius=10)
    plot_path(path_00, show=False)
    plot_path(path_01, show=False)
    plot_path(path_02, show=False)
    plot_path(path_03, show=False)


def test_path_length():
    image = mpimg.imread("data/test/map.jpg")
    binary_image = image[:, :, 0] > 150
    decomposed = create_mask(binary_image)
    cells = Cell.from_image(decomposed)

    path_00 = create_path(cells[0], 0, coverage_radius=10)
    path_00_legth = path_length(path_00)
    path_00_exp_length = 459 + 20 + 459 + 17 + 459
    assert path_00_legth == path_00_exp_length


def test_get_path_end_corner():
    image = mpimg.imread("data/test/map.jpg")
    binary_image = image[:, :, 0] > 150
    decomposed = create_mask(binary_image)
    cells = Cell.from_image(decomposed)

    path_00 = create_path(cells[0], 0, coverage_radius=10)
    path_01 = create_path(cells[0], 1, coverage_radius=10)
    path_02 = create_path(cells[0], 2, coverage_radius=10)
    path_03 = create_path(cells[0], 3, coverage_radius=10)
    plot_path(path_00, show=False)
    plot_path(path_01, show=False)
    plot_path(path_02, show=False)
    plot_path(path_03, show=False)

    path_00_end_corner = get_path_end_corner(path_00, 0)
    path_01_end_corner = get_path_end_corner(path_01, 1)
    path_02_end_corner = get_path_end_corner(path_02, 2)
    path_03_end_corner = get_path_end_corner(path_03, 3)

    assert path_00_end_corner == 2
    assert path_01_end_corner == 3
    assert path_02_end_corner == 0
    assert path_03_end_corner == 1


def test_get_path_end_corner_2():
    image = mpimg.imread("data/test/map.jpg")
    binary_image = image[:, :, 0] > 150
    decomposed = create_mask(binary_image)
    cells = Cell.from_image(decomposed)

    path_10 = create_path(cells[1], 0, coverage_radius=10)
    path_11 = create_path(cells[1], 1, coverage_radius=10)
    path_12 = create_path(cells[1], 2, coverage_radius=10)
    path_13 = create_path(cells[1], 3, coverage_radius=10)
    plot_path(path_10, show=False)
    plot_path(path_11, show=False)
    plot_path(path_12, show=False)
    plot_path(path_13, show=False)

    path_10_end_corner = get_path_end_corner(path_10, 0)
    path_11_end_corner = get_path_end_corner(path_11, 1)
    path_12_end_corner = get_path_end_corner(path_12, 2)
    path_13_end_corner = get_path_end_corner(path_13, 3)

    print(
        path_10_end_corner, path_11_end_corner, path_12_end_corner, path_13_end_corner
    )
    assert path_10_end_corner == 3
    assert path_11_end_corner == 2
    assert path_12_end_corner == 1
    assert path_13_end_corner == 0


def test_distance_intra_cells():
    image = mpimg.imread("data/test/map.jpg")
    binary_image = image[:, :, 0] > 150
    decomposed = create_mask(binary_image)
    cells = Cell.from_image(decomposed)
    adj_matrix = create_global_adj_matrix(binary_image)

    dist_0_1 = []
    for corner_id in range(4):
        dist_0_1.append(
            dist_intra_cells(
                cells,
                cells[0],
                cells[1],
                corner_id,
                0,
                coverage_radius=10,
                adj_contraint=True,
            )
        )

    expected_dist_0_1 = [np.infty, 1424.05, np.infty, np.infty]
    assert np.allclose(expected_dist_0_1, dist_0_1)

    dist_1_0 = []
    for corner_id in range(4):
        dist_1_0.append(
            dist_intra_cells(
                cells,
                cells[1],
                cells[0],
                corner_id,
                0,
                coverage_radius=10,
                adj_contraint=True,
            )
        )
    expected_dist_1_0 = [np.infty, np.infty, 842.307, 791.809]
    assert np.allclose(expected_dist_1_0, dist_1_0)


def test_shortest_path():
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
    # visited = [0, 1, 5, 3, 2, 4, 6, 7, 9, 8]

    # create cells
    decomposed_image = create_mask(binary_image)
    cells = Cell.from_image(decomposed_image)

    # indexes of cells
    visited_simple = [1, 0, 2]
    # shortest path
    dp = shortest_path(cells=cells, cell_sequence=visited_simple, coverage_radius=10)


def test_global_path():
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
    print(new_root)

    tree_edges = graph_tree.DFS_tree(new_root)
    longest_tree = Graph()
    longest_tree.add_edges(tree_edges)
    visited = longest_tree.post_order_traversal(new_root)
    # visited = [0, 1, 5, 3, 2, 4, 6, 7, 9, 8]
    # create cells
    decomposed_image = create_mask(binary_image)
    cells = Cell.from_image(decomposed_image)

    # shortest path
    dp = shortest_path(
        cells=cells, cell_sequence=visited, coverage_radius=10, adj_contraint=False
    )
    path = reconstruct_path(dp, cells, visited, coverage_radius=10)
    plot_cells(cells, show=False)
    plot_global_path(path, show=False)
    plt.imshow(image)
    plt.savefig("data/test/test_path.png")


def test_global_path_concave_obstacles():
    binary_image = recurvise_H_map((400, 600), invert=True, num_recursions=0)
    graph_adj_matrix, decomposed_image = get_directed_global_adj_matrix(binary_image)
    graph_adj_dict = adj_matrix_to_dict(graph_adj_matrix)

    graph = Graph(graph_adj_dict)
    mst_arcs = min_spanning_arborescence(
        graph.graph_to_arcs_with_dfs(source=0), sink=0
    ).values()
    mst_graph = Graph()
    mst_graph.arcs_to_graph(mst_arcs)
    mst_graph.reverse_graph_edges()
    visited = list(reversed((mst_graph.post_order_traversal(node=0))))
    print(visited)
    # create cells
    cells = Cell.from_image(decomposed_image)

    # shortest path
    # it does not work with non adjacent cells !!!
    dp = shortest_path(cells=cells, cell_sequence=visited, coverage_radius=10)
    print(dp)
    path = reconstruct_path(dp, cells, visited, coverage_radius=10)
    plot_cells(cells, show=False)
    plot_global_path(path, show=False)
    plt.imshow(binary_image)
    plt.savefig("data/test/test_concave_obs_path.png")


def test_small_global_path():
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
    # visited = [0, 1, 5, 3, 2, 4, 6, 7, 9, 8]

    # create cells
    decomposed_image = create_mask(binary_image)
    cells = Cell.from_image(decomposed_image)

    # indexes of cells
    visited_simple = [1, 0, 2]
    path_0 = create_path(cells[1], 0, coverage_radius=10)

    # shortest path
    dp = shortest_path(cells=cells, cell_sequence=visited_simple, coverage_radius=10)
    path = reconstruct_path(dp, cells, visited_simple, coverage_radius=10)
    plot_cells(cells, show=False)
    plot_global_path(path, show=False)
    plt.imshow(image)
    plt.savefig("logs/images/test_path.png")


def test_plotter():
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
    # visited = [0, 1, 5, 3, 2, 4, 6, 7, 9, 8]

    # create cells
    decomposed_image = create_mask(binary_image)
    cells = Cell.from_image(decomposed_image)

    plt.imshow(image)
    plot_cells(cells, show=False)

    plt.savefig("logs/images/image_decomposed.jpg")


def test_shortest_path_dijstra():
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


def test_global_path_concave_obstacles_dijstra():
    binary_image = recurvise_H_map((400, 600), invert=True, num_recursions=0)
    graph_adj_matrix, decomposed_image = get_directed_global_adj_matrix(binary_image)
    graph_adj_dict = adj_matrix_to_dict(graph_adj_matrix)
    cells = Cell.from_image(decomposed_image)

    graph = Graph(graph_adj_dict)
    start_node = 0
    mst_arcs = min_spanning_arborescence(
        graph.graph_to_arcs_with_dfs(source=start_node), sink=start_node
    ).values()
    mst_graph = Graph()
    mst_graph.arcs_to_graph(mst_arcs)
    mst_graph.reverse_graph_edges()

    graph_adj_matrix_unweighted, decomposed_im = create_global_adj_matrix(binary_image)
    graph_adj_dict_unweighted = adj_matrix_to_dict(graph_adj_matrix_unweighted)
    graph_unweighted = Graph(graph_adj_dict_unweighted)
    graph_unweighted.weights = get_distance_between_cells(graph_unweighted.adj, cells)

    post_order_traversal_sorted = mst_graph.post_order_traversal_sorted(
        start_node, graph_unweighted
    )
    visited = list(reversed(post_order_traversal_sorted))
    # create cells

    # shortest path
    dp = shortest_path(cells=cells, cell_sequence=visited, coverage_radius=10)

    path = reconstruct_path(dp, cells, visited, coverage_radius=10)
    plot_cells(cells, show=False)
    plot_global_path(path, show=False)
    plt.imshow(binary_image)
    plt.savefig("data/test/test_concave_obs_path_dijstra.png")


def test_global_path_concave_obstacles_dijstra_1():
    binary_image = recurvise_H_map((400, 600), invert=True, num_recursions=1)
    graph_adj_matrix, decomposed_image = get_directed_global_adj_matrix(binary_image)
    graph_adj_dict = adj_matrix_to_dict(graph_adj_matrix)
    cells = Cell.from_image(decomposed_image)

    graph = Graph(graph_adj_dict)
    start_node = 0
    mst_arcs = min_spanning_arborescence(
        graph.graph_to_arcs_with_dfs(source=start_node), sink=start_node
    ).values()
    mst_graph = Graph()
    mst_graph.arcs_to_graph(mst_arcs)
    mst_graph.reverse_graph_edges()

    graph_adj_matrix_unweighted, decomposed_im = create_global_adj_matrix(binary_image)
    graph_adj_dict_unweighted = adj_matrix_to_dict(graph_adj_matrix_unweighted)
    graph_unweighted = Graph(graph_adj_dict_unweighted)
    graph_unweighted.weights = get_distance_between_cells(graph_unweighted.adj, cells)

    post_order_traversal_sorted = mst_graph.post_order_traversal_sorted(
        start_node, graph_unweighted
    )
    visited = list(reversed(post_order_traversal_sorted))
    # create cells

    # shortest path
    dp = shortest_path(cells=cells, cell_sequence=visited, coverage_radius=10)
    path = reconstruct_path(dp, cells, visited, coverage_radius=10)
    plot_cells(cells, show=False, fontsize=5)
    plot_global_path(path, show=False, markersize=5)
    plt.imshow(binary_image)
    plt.savefig("data/test/test_concave_obs_path_dijstra_1_.pdf", dpi=300)


def test_global_path_concave_obstacles_dijstra_2():
    binary_image = recurvise_H_map((800, 1200), invert=True, num_recursions=2)
    graph_adj_matrix, decomposed_image = get_directed_global_adj_matrix(binary_image)
    graph_adj_dict = adj_matrix_to_dict(graph_adj_matrix)
    cells = Cell.from_image(decomposed_image)

    graph = Graph(graph_adj_dict)
    start_node = 0
    mst_arcs = min_spanning_arborescence(
        graph.graph_to_arcs_with_dfs(source=start_node), sink=start_node
    ).values()
    mst_graph = Graph()
    mst_graph.arcs_to_graph(mst_arcs)
    mst_graph.reverse_graph_edges()

    graph_adj_matrix_unweighted, decomposed_im = create_global_adj_matrix(binary_image)
    graph_adj_dict_unweighted = adj_matrix_to_dict(graph_adj_matrix_unweighted)
    graph_unweighted = Graph(graph_adj_dict_unweighted)
    graph_unweighted.weights = get_distance_between_cells(graph_unweighted.adj, cells)

    post_order_traversal_sorted = mst_graph.post_order_traversal_sorted(
        start_node, graph_unweighted
    )
    visited = list(reversed(post_order_traversal_sorted))
    # create cells

    # shortest path
    coverage_radius = 2
    dp = shortest_path(
        cells=cells, cell_sequence=visited, coverage_radius=coverage_radius
    )
    path = reconstruct_path(dp, cells, visited, coverage_radius=coverage_radius)
    plot_cells(cells, show=False, fontsize=1)
    plot_global_path(path, show=False, markersize=5)
    plt.imshow(binary_image)
    plt.savefig("data/test/test_concave_obs_path_dijstra_2_.pdf", dpi=300)


if __name__ == "__main__":
    # test_directed_global_adj_matrix_concave_obstacles()
    # test_global_path()test_plot_bsd_reduced_decomp
    test_plot_bsd_reduced_decomp()
