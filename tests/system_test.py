from matplotlib.pyplot import show
import numpy as np
from cpp.bsd import *
from cpp.dfs_tree import *
from cpp.cells import *
import matplotlib.image as mpimg
from cpp.helpers import *


def test_plotter():
    # goes from 0 to 1 the occupancy map
    image = mpimg.imread("data/occupancy.png")
    # original image is black and white anyway
    binary_image = image[:, :, 0] > 0.99
    plt.imshow(binary_image)
    plt.show()
    traversable_outside = False
    graph_adj_matrix, decomposed_image = create_global_adj_matrix(binary_image, traversable_outside)
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
    # decomposed_image = create_mask(binary_image, traversable_outside)
    cells = Cell.from_image(decomposed_image)

    plt.imshow(image)
    plot_cells(cells, show=True)

    plt.savefig("logs/images/image_decomposed.jpg")

def test_global_path():
    image = mpimg.imread("data/occupancy.png")
    # original image is black and white anyway
    binary_image = image[:, :, 0] > 0.99
    graph_adj_matrix, decomposed_image = create_global_adj_matrix(binary_image, traversable_outside=False)
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
    cells = Cell.from_image(decomposed_image)
    coverage_radius = 25
    # shortest path
    dp = shortest_path(
        cells=cells, cell_sequence=visited, coverage_radius=coverage_radius, adj_contraint=False
    )
    path = reconstruct_path(dp, cells, visited, coverage_radius=coverage_radius)
    plot_cells(cells, show=False)
    plot_global_path(path, show=False)
    plt.imshow(image)
    plt.show()
    plt.savefig("logs/occupancy_path.png")

if __name__ == "__main__":
    test_global_path()