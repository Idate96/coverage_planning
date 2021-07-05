from numpy.lib.function_base import cov
from cpp.cells import Cell
from typing import List, Tuple
import numpy as np
import pickle
import matplotlib.image as mping


def find_connectivity(slice: np.array):
    """Find the connectivity of a boolean slice.
    The slice contains 1 and 0s to indicate wether the point belongs to the workspace

    Args:
        slice (np.array): array of booleans
    Returns:
        segments (Tuple[int, int]): start idx and legth of the segment
    """
    segments = []
    # segments will be identified by their starting idx and legth
    start_segment_idx = 0
    outside_of_workspace = False
    for i in range(len(slice)):
        if slice[i] == 0 and not outside_of_workspace:
            segments.append((start_segment_idx, len(slice[:i])))
            outside_of_workspace = True
        if slice[i] == 1 and outside_of_workspace:
            start_segment_idx = i
            outside_of_workspace = False

    # last segment if present
    if not outside_of_workspace:
        segments.append((start_segment_idx, len(slice)))

    return segments


def find_slices_adjacency(
    slice_low: List[Tuple[int, int]], slice_high: List[Tuple[int, int]]
):
    """Check where the segments of the slices are adjacent to each other

    Args:
        slice_low (List[Tuple[int, int]]): list of connected regions(segments)
        slice_high (List[Tuple[int, int]])

    Returns:
        adj_matrix (np.arrary)
    """
    adj_matrix = np.zeros((len(slice_low), len(slice_high)), dtype=bool)
    for i, segment_low in enumerate(slice_low):
        for j, segment_high in enumerate(slice_high):
            # check if there is an overlapping region
            len_overlapping_region = min(segment_high[1], segment_low[1]) - max(
                segment_high[0], segment_low[0]
            )
            if len_overlapping_region > 0:
                adj_matrix[i, j] = 1
    return adj_matrix


def create_mask(binary_image: np.array):
    """Create a mask that assigns a label, representing the cell id, to each point in the workspace.

    Args:
        binary_images (np.array): array that encodes the workspace

    Returns:
        mask (np.array): array containing the labels for each pt
    """
    mask = np.zeros(binary_image.shape, dtype=int)

    previous_segments = []
    num_previous_segments = 0

    previous_cells = []
    num_cells = 1

    for col_id in range(binary_image.shape[1]):
        slice = binary_image[:, col_id]
        current_segments = find_connectivity(slice)
        num_segments = len(current_segments)

        # at
        if num_previous_segments == 0:

            current_cells = []
            for i in range(num_segments):
                current_cells.append(num_cells)
                num_cells += 1

        else:
            current_cells = [0] * num_segments
            adj_matrix = find_slices_adjacency(previous_segments, current_segments)

            for i in range(num_previous_segments):
                # simply connected 1 to 1
                if np.sum(adj_matrix[i, :]) == 1:
                    # assign to the only j cell connected the same num
                    for j in range(num_segments):
                        if adj_matrix[i, j]:
                            current_cells[j] = previous_cells[i]
                # split event
                elif np.sum(adj_matrix[i, :]) > 1:
                    for j in range(num_segments):
                        if adj_matrix[i, j]:
                            current_cells[j] = num_cells
                            num_cells += 1

            for j in range(num_segments):
                # merge event
                if np.sum(adj_matrix[:, j]) > 1 or np.sum(adj_matrix[:, j]) == 0:
                    current_cells[j] = num_cells
                    num_cells += 1

        for i in range(len(current_cells)):
            mask[
                current_segments[i][0] : current_segments[i][1], col_id
            ] = current_cells[i]

        previous_cells = current_cells
        previous_segments = current_segments
        num_previous_segments = num_segments

    return mask


def create_path(cell: Cell, start_pt, end_pt, coverage_radius: int):
    vertical_dir = "down"
    horizontal_dir = "right"
    # This depends on the starting point
    x_current = min(cell.x_left + coverage_radius, cell.x_right)
    done = False
    path = []
    print(cell.x_left, cell.x_right)

    while not done:
        y_min = min(cell.bottom[x_current] + coverage_radius, cell.top[x_current])
        y_max = max(cell.top[x_current] - coverage_radius, cell.bottom[x_current])

        # add vertical component to the path
        if vertical_dir == "down":
            path.append((x_current, y_max))
            path.append((x_current, y_min))
            vertical_dir = "up"
        else:
            path.append((x_current, y_min))
            path.append((x_current, y_max))
            vertical_dir = "down"

        # check if another segment fit in the cell
        # TODO: this depends on the order too
        # TODO: allow for float
        if x_current + coverage_radius < cell.x_right:
            x_next = min(x_current + 2 * coverage_radius, cell.x_right)
            # here we assume that x lies in the cells
            for x in range(x_current, x_next):
                # check if we are the top or bottom
                if vertical_dir == "down":
                    print("options ", cell.top[x] - coverage_radius, cell.bottom[x])
                    y = max(cell.top[x] - coverage_radius, cell.bottom[x])
                else:
                    print("bottom options ")
                    y = min(cell.bottom[x] + coverage_radius, cell.top[x])
                path.append((x, y))

        else:
            done = True
        x_current = x_next

    # flip the order if the cell had to be covered in the opposite direction
    if horizontal_dir == "left":
        path.reverse()

    return path


def create_global_adj_matrix(binary_image: np.ndarray) -> np.ndarray:
    """Creates a graph representing the global connectivity of the workspace cells

    Args:
        binary_image (np.ndarray): images or 0 and 1, encoding areas belonging
                                   or not to the workspace

    Returns:
        graph: graph of the cells connectivity 
    """
    
    decomposed_image = create_mask(binary_image)
    num_cells = np.max(decomposed_image)
    graph = np.zeros((num_cells + 1, num_cells + 1))
    previous_segments = []

    for col_id in range(decomposed_image.shape[1]):
        slice = binary_image[:, col_id]
        current_segments = find_connectivity(slice)

        adj_matrix = find_slices_adjacency(previous_segments, current_segments)

        for i in range(len(previous_segments)):
            for j in range(len(current_segments)):
                if adj_matrix[i, j]:
                    idx_i = decomposed_image[previous_segments[i][0], col_id - 1]
                    idx_j = decomposed_image[current_segments[j][0], col_id]
                    if idx_i != idx_j:
                        graph[idx_i, idx_j] = 1
                        graph[idx_j, idx_i] = 1

        previous_segments = current_segments
    # excludes the zero cell (obstacles)
    graph = graph[1:, 1:]

    return graph