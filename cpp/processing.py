import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import bsd
from cells import plot_cells, num_path_turns
import cv2
import dfs_tree
import scipy.optimize
import helpers


def rotate(image: np.ndarray, angle):
    """
    Rotate an array counter-clockwise by a given angle.
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    image = image.astype(np.uint8)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1])
    return result


def post_process_mask(binary_image, mask: np.ndarray):
    """
    Post-process the mask to remove small components.
    """
    unique_mask_id = np.unique(mask)
    cols_per_id = {}
    # initialize cols_per_id dict
    for id in unique_mask_id:
        cols_per_id[id] = 0

    for col in range(mask.shape[1]):
        unique_col_id = np.unique(mask[:, col])
        for mask_id_cl in unique_col_id:
            cols_per_id[mask_id_cl] += 1
    # ids that have appear in less then 2 columns
    ids_to_remove = [id for id in cols_per_id if cols_per_id[id] < 3]

    for j in range(mask.shape[1]):
        col = mask[:, j]
        col_binary = binary_image[:, j]
        for id in ids_to_remove:
            for i in range(col.shape[0]):
                if col[i] == id:
                    col[i] = 0
                    col_binary[i] = 0
            col[col == id] = 0

    return binary_image, mask


def find_num_turns(binary_image, rotation_angle):
    binary_image = rotate(binary_image, rotation_angle)
    binary_image = binary_image == 1

    graph_adj_matrix, decomposed_image = bsd.create_global_adj_matrix(binary_image, traversable_outside=False)
    # print("num of cols per id ", post_process_mask(decomposed_image))
    new_binary_image, mask = post_process_mask(binary_image, decomposed_image)
    _, new_mask = bsd.create_global_adj_matrix(new_binary_image, traversable_outside=False)
    cells = bsd.Cell.from_image(new_mask)
    print([cell.cell_id for cell in cells])
    num_turns = 0
    for cell in cells:
        num_turns += num_path_turns(cell, coverage_radius=20)
    num_turns += len(cells)
    return num_turns


def find_best_map_orientation(binary_image):
    # create closure by passing binary image to find_num_turns
    def find_num_turns_closure(angle):
        return find_num_turns(binary_image, angle)

    angle = scipy.optimize.minimize_scalar(find_num_turns_closure, bounds=(0, 360), method='bounded',
                                           options={'maxiter': 100, 'xatol': 0.5, 'disp': 3})
    return angle


def transform_path(path: np.ndarray, translation, rotation, resolution):
    """
    Scale, translate and rotate the path from image space to map frame.
    """
    path *= resolution  # pixels to meters
    path += translation
    map_frame_path = rotation @ path
    return map_frame_path


if __name__ == '__main__':
    image = mpimg.imread("data/occupancy.png")
    # original image is black and white anyway
    binary_image = image[:, :, 0] > 0.99

    binary_image = rotate(binary_image, 0)
    binary_image = binary_image == 1
    #
    graph_adj_matrix, decomposed_image = bsd.create_global_adj_matrix(binary_image, traversable_outside=False)
    # print("num of cols per id ", post_process_mask(decomposed_image))
    new_binary_image, mask = post_process_mask(binary_image, decomposed_image)
    graph_adj_matrix, new_mask = bsd.create_global_adj_matrix(new_binary_image, traversable_outside=False)
    cells = bsd.Cell.from_image(new_mask)
    print([cell.cell_id for cell in cells])
    num_turns = 0
    for cell in cells:
        num_turns += num_path_turns(cell, coverage_radius=20)
    num_turns += len(cells) * 2
    print(f"Number of turns: {num_turns}")
    plot_cells(cells)
    plt.imshow(binary_image)
    plt.show()


    graph_adj_dict = dfs_tree.adj_matrix_to_dict(graph_adj_matrix)
    graph = dfs_tree.Graph(graph_adj_dict)
    tree_edges = graph.DFS_tree(0)
    graph_tree = dfs_tree.Graph()
    graph_tree.add_edges(tree_edges, directed=False)
    diameter, new_root, furthest_node = graph_tree.find_diameter(0)

    tree_edges = graph_tree.DFS_tree(new_root)
    longest_tree = dfs_tree.Graph()
    longest_tree.add_edges(tree_edges)
    visited = longest_tree.post_order_traversal(new_root)
    # visited = [0, 1, 5, 3, 2, 4, 6, 7, 9, 8]
    # create cells)

    # shortest path
    dp = bsd.shortest_path(
        cells=cells, cell_sequence=visited, coverage_radius=10, adj_contraint=False
    )
    path = bsd.reconstruct_path(dp, cells, visited, coverage_radius=10)
    print(path)
    plot_cells(cells, show=False)
    helpers.plot_global_path(path, show=False)
    plt.imshow(image)
    plt.show()
    # # rotate image
