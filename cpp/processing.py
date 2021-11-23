import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cpp.bsd as bsd
from cpp.cells import plot_cells, num_path_turns
import cv2
import cpp.dfs_tree as dfs_tree
import scipy.optimize
import cpp.helpers as helpers


def rotate_pts(se2_pts, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    se2_pts[:, :2] = rot_mat @ se2_pts[:, :2]
    # rotate the heading
    se2_pts[:, 2] += angle
    return path


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
    # print([cell.cell_id for cell in cells])
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


def path_to_image_frame(path: np.ndarray, rotation_angle,  resolution):
    path_np = path[:, :2]
    yaw_np = path[:, 2]
    image_center = np.array(image.shape[1::-1], dtype=float) / 2
    rot_mat = cv2.getRotationMatrix2D(tuple(image_center), rotation_angle, 1.0)
    # rotate the heading
    path_np = path_np - image_center
    yaw_np += rotation_angle/180*np.pi
    # path_np = np.concatenate((path_np, np.ones((path_np.shape[0], 1))), axis=1)
    pixel_frame_path = rot_mat[:2, :2] @ path_np.T + image_center[:, np.newaxis]
    pixel_frame_path = pixel_frame_path * resolution  # pixels to meters
    return np.hstack((pixel_frame_path.T, yaw_np[:, np.newaxis]))


def image_to_map_frame(pixel_path, m_P_mp):
    """
    Convert a path in pixel frame to map frame.
    """
    path_np = pixel_path[:, :2].T
    yaw_np = pixel_path[:, 2]
    # Rotate 90 degrees clockwise
    R_mp = -np.array([[0, 1], [-1, 0]])
    map_frame_path = R_mp @ path_np
    yaw_np += np.pi/2
    # translate to map frame
    map_frame_path[1, :] = -map_frame_path[1, :]
    map_frame_path = map_frame_path + m_P_mp[:, np.newaxis]
    # flip y axis
    yaw_np = yaw_np % (2 * np.pi)

    path_output = np.hstack((map_frame_path[:2, :].T, yaw_np[:, np.newaxis]))
    return path_output


def path_list_to_np(path_list):
    path_flatten = [item for sublist in path_list for item in sublist]
    return np.array(path_flatten)


def path_np_to_list(path_np, path_list):
    """
    Convert a path in numpy array format to a list of tuples
    """
    path_list_new = []
    idx = 0
    for i in range(len(path_list)):
        cell_path_new = []
        for j in range(len(path_list[i])):
            # convert tuple to np array
            cell_path_new.append((path_np[idx, 0], path_np[idx, 1], path_np[idx, 2]))
            idx += 1
        path_list_new.append(cell_path_new)
    return path_list_new


def reverse_path_order(path_np):
    """
    Reverse the order of the path
    """
    path_np_new = np.zeros_like(path_np)
    for i in range(len(path_np)):
        path_np_new[i, :] = path_np[len(path_np) - i - 1, :]
    return path_np_new


if __name__ == '__main__':
    image = mpimg.imread("data/occupancy_or.png")
    # original image is black and white anyway
    binary_image = image[:, :, 0] > 0.99
    angle = 198
    binary_image = rotate(binary_image, angle)
    binary_image = binary_image == 1
    #
    graph_adj_matrix, decomposed_image = bsd.create_global_adj_matrix(binary_image, traversable_outside=False)
    # print("num of cols per id ", post_process_mask(decomposed_image))
    new_binary_image, mask = post_process_mask(binary_image, decomposed_image)
    graph_adj_matrix, new_mask = bsd.create_global_adj_matrix(new_binary_image, traversable_outside=False)
    cells = bsd.Cell.from_image(new_mask)
    # print([cell.cell_id for cell in cells])
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
       cells=cells, cell_sequence=visited, coverage_radius=40, adj_contraint=False
    )
    path = bsd.reconstruct_path(dp, cells, visited, coverage_radius=40)
    path_np = path_list_to_np(path)

    path_in_image_frame = path_to_image_frame(path_np, rotation_angle=-angle, resolution=0.1)
    path_in_map_frame = image_to_map_frame(path_in_image_frame, np.array([11.17, 54.11]))
    reversed_path_map_frame = reverse_path_order(path_in_map_frame)
    path_map_list = path_np_to_list(path_in_map_frame, path)
    # helpers.save_path(path_map, "data/path")
    np.save("data/path.npy", reversed_path_map_frame)
    # plot_cells(cells, show=False)
    # helpers.plot_global_path(path, show=True)
    # helpers.plot_global_path(path_map_list, show=False)

    # plt.imshow(image)
    # plt.show()
    # # rotate image
