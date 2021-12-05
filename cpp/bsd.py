from numpy.core.numeric import binary_repr
from numpy.lib.function_base import cov
from cpp.cells import Cell, filter_cells
from typing import List, Tuple
import numpy as np
import pickle
import matplotlib.image as mping
import math


def find_connectivity(slice: np.array, traversable_outside=False):
    """Find the connectivity of a boolean slice.
    The slice contains 1 and 0s to indicate wether the point belongs to the workspace

    Args:
        slice (np.array): array of booleans
        traversable_outside (bool): if False, the region to be segmented is sorrounded by an untraversable region
    Returns:
        segments (Tuple[int, int]): start idx and length of the segment
    """
    segments = []
    # segments will be identified by their starting idx and length
    start_segment_idx = 0
    outside_of_workspace = not traversable_outside
    for i in range(len(slice)):
        # current cell ends
        if slice[i] == 0 and not outside_of_workspace:
            segments.append((start_segment_idx, i))
            outside_of_workspace = True
        # another cell starts
        if slice[i] == 1 and outside_of_workspace:
            start_segment_idx = i
            outside_of_workspace = False

    # last segment if present
    if not outside_of_workspace:
        segments.append((start_segment_idx, len(slice)))

    # make it more robust to imperfect segmentation
    merged_segments = []
    if len(segments) > 1:
        current_start_idx = segments[0][0]
        current_end_idx = segments[0][1]

        for (start_idx, end_idx) in segments[1:]:
            # merge segments that are separated by less than 10 pixels
            if start_idx - current_end_idx < 20:
                current_end_idx = end_idx
                # edge case
                # if end_idx == segments[-1][1]:
                #     merged_segments.append((current_start_idx, current_end_idx))
            else:
                merged_segments.append((current_start_idx, current_end_idx))
                current_start_idx = start_idx
                current_end_idx = end_idx

        merged_segments.append((current_start_idx, current_end_idx))
    else:
        merged_segments = segments

    return merged_segments


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


def create_mask(binary_image: np.array, traversable_outside: bool):
    """Create a mask that assigns a label, representing the cell id, to each point in the workspace.

    Args:
        binary_images (np.array): array that encodes the workspace
        traversable_outside (bool): if False, the region to be segmented is sorrounded by an untraversable region

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
        current_segments = find_connectivity(slice, traversable_outside)
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
            # to make ti robust against artifact in the mask
            # only allow segments biggen than 10
            mask[current_segments[i][0]:current_segments[i][1], col_id] = current_cells[i]

        previous_cells = current_cells
        previous_segments = current_segments
        num_previous_segments = num_segments

    return mask


def create_path(cell: Cell, start_corner, coverage_radius: int):
    if start_corner == 0 or start_corner == 3:
        vertical_dir = "up"
    else:
        vertical_dir = "down"

    if start_corner == 0 or start_corner == 1:
        horizontal_dir = "right"
    else:
        horizontal_dir = "left"

    # This depends on the starting point
    if horizontal_dir == "right":
        x_current = cell.x_left + coverage_radius
        x_end = cell.x_right - coverage_radius
        step = 2*coverage_radius
    else:
        x_current = cell.x_right - coverage_radius
        x_end = cell.x_left + coverage_radius
        step = -2*coverage_radius
    # x_current = min(cell.x_left + coverage_radius, cell.x_right - coverage_radius)
    done = False
    path = []

    while cell.x_right > x_current > cell.x_left:
        y_min = min(cell.bottom[x_current] + coverage_radius, cell.top[x_current])
        y_max = max(cell.top[x_current] - coverage_radius, cell.bottom[x_current])

        if vertical_dir == "down":
            # linearly interpolate between y_min and y_max with spacing equal to the coverage ratio
            # heading will be trasformed from the path frame to the map frame
            for y_current in range(y_max, y_min, -step):
                path.append((x_current, y_current, np.pi/2))
            # path.append((x_current, y_max, np.pi/2))
            path.append((x_current, y_min, np.pi/2))
            vertical_dir = "up"
        else:
            for y_current in range(y_min, y_max, step):
                path.append((x_current, y_current, 3/2*np.pi))
            path.append((x_current, y_max, 3/2 * np.pi))
            vertical_dir = "down"

        x_current += step




    # while not done:
    #     y_min = min(cell.bottom[x_current] + coverage_radius, cell.top[x_current])
    #     y_max = max(cell.top[x_current] - coverage_radius, cell.bottom[x_current])
    #
    #     # add vertical component to the path
    #     # the pose of the robot is in SE2
    #     if vertical_dir == "down":
    #         # linearly interpolate between y_min and y_max with spacing equal to the coverage ratio
    #         # heading will be trasformed from the path frame to the map frame
    #         path.append((x_current, y_max, np.pi/2))
    #         path.append((x_current, y_min, np.pi/2))
    #         vertical_dir = "up"
    #     else:
    #         path.append((x_current, y_min, 3/2 * np.pi))
    #         path.append((x_current, y_max, 3/2 * np.pi))
    #         vertical_dir = "down"
    #     x_current += step
    #
    #     # TODO: this depends on the order too
    #     # TODO: allow for float
    #     # TODO: to allow for different orientation depending on the cells
    #     # the cell representation should be invariant under rotation
    #     x_next = x_current + step
    #     if cell.x_right > x_next > cell.x_left:
    #         # here we assume that x lies in the cells
    #         for x in range(x_current, x_next):
    #             # check if we are the top or bottom
    #             if vertical_dir == "down":
    #                 y = max(cell.top[x] - coverage_radius, cell.bottom[x])
    #             else:
    #                 y = min(cell.bottom[x] + coverage_radius, cell.top[x])
    #             path.append((x, y))
    #         x_current = x_next
    #     else:
    #         done = True

    # flip the order if the cell had to be covered in the opposite direction
    # if horizontal_dir == "left":
    #     path.reverse()
    print("PATH  : ",  path)
    return path


def path_length(path: List[Tuple[int, int]]) -> int:
    """Calculate the length of a path

    Args:
        path (List[Tuple[int, int]]): list of points

    Returns:
        length (int): length of the path
    """
    length = 0
    for i in range(len(path) - 1):
        length += np.sqrt(
            (path[i][0] - path[i + 1][0]) ** 2 + (path[i][1] - path[i + 1][1]) ** 2
        )
    return length


def count_vertical_segmnents_in_path(path: List[Tuple[int, int]]) -> int:
    """Count the number of vertical segments in a path

    Args:
        path (List[Tuple[int, int]]): list of points

    Returns:
        num_segments (int): number of vertical segments
    """
    num_segments = 0
    for i in range(len(path) - 1):
        if path[i][0] == path[i + 1][0] and path[i][1] != path[i + 1][1]:
            num_segments += 1
    return num_segments


def get_path_end_corner(path: List[Tuple[int, int]], corner_start: int) -> int:
    """Return the id of the corner at the end of the path by counting
       the number of vertical segments in the path

    Args:
        path (List[Tuple[int, int]]): list of points
        corner_start (int): id of the starting corner

    Returns:
        corner_end (int): id of the ending corner
    """
    corner_end = None
    num_segments = count_vertical_segmnents_in_path(path)
    if num_segments % 2 == 0:
        if corner_start == 0:
            corner_end = 3
        elif corner_start == 3:
            corner_end = 0
        elif corner_start == 1:
            corner_end = 2
        elif corner_start == 2:
            corner_end = 1
    else:
        if corner_start == 0:
            corner_end = 2
        elif corner_start == 2:
            corner_end = 0
        elif corner_start == 1:
            corner_end = 3
        elif corner_start == 3:
            corner_end = 1

    return corner_end


# def get_path_end_corner_(cell: Cell, path: List[Tuple[int, int]], coverage_radius: int) -> int:
#     """Find the corner at the end of the path"""
#     if path[-1][0] == cell.x_left:
#         # can be either top or bottom
#         if path[-1][1] == cell.top[cell.x_left]:
#             return 0
#         elif path[-1][1] == cell.bottom[cell.x_left]:
#             return 1
#         else:
#             raise ValueError("Unexpected path end")

#     elif path[-1][0] == cell.x_right:
#         # can be either top or bottom
#         if path[-1][1] == cell.bottom[cell.x_right]:
#             return 2
#         elif path[-1][1] == cell.top[cell.x_right]:
#             return 3
#         else:
#             raise ValueError("Unexpected path end")
#     else:
#         raise ValueError("Path does not start at left or right")



def create_global_adj_matrix(binary_image: np.ndarray, traversable_outside: bool) -> np.ndarray:
    """Creates a graph representing the global connectivity of the workspace cells

    Args:
        binary_image (np.ndarray): images or 0 and 1, encoding areas belonging
                                   or not to the workspace

    Returns:
        graph: graph of the cells connectivity
    """

    decomposed_image = create_mask(binary_image, traversable_outside=traversable_outside)
    num_cells = np.max(decomposed_image)
    graph = np.zeros((num_cells + 1, num_cells + 1))
    previous_segments = []

    for col_id in range(1, decomposed_image.shape[1]):
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

    return graph, decomposed_image


def get_directed_global_adj_matrix(binary_image: np.ndarray) -> np.ndarray:
    """Creates a directed graph representing the global connectivity of the workspace cells

    Args:
        binary_image (np.ndarray): images or 0 and 1, encoding areas belonging

    Returns:
        graph: directed graph of the cells connectivity

    """
    decomposed_image = create_mask(binary_image)
    cells = Cell.from_image(decomposed_image)

    directed_adj_matrix = np.zeros((len(cells), len(cells)))
    # cells only connects through corners, as those are the end points of the motion primitives
    # we prune edges from adj_matrix by checking the cells adj to the corners
    for cell in cells:
        for corner in range(4):
            adj_cell_id = get_adj_cell_to_corner(cells, cell.cell_id, corner)
            if adj_cell_id is not None:
                directed_adj_matrix[cell.cell_id, adj_cell_id] = 1

    return directed_adj_matrix, decomposed_image


def corner_adjencency(cell_1: Cell, cell_2: Cell) -> bool:
    """Check if first cell is connected to second cell via a corner

    Args:
        cell_1 (Cell): cell 1
        cell_2 (Cell): cell 2

    Returns:
        adj: True if cell 1 is connected to cell 2 via a corner

    """
    adj = False
    for corner in range(4):
        adj_cell_id = get_adj_cell_to_corner([cell_1, cell_2], cell_1.cell_id, corner)
        if adj_cell_id is not None:
            adj = True
    return adj

def get_adj_cell_to_corner(cells: List[Cell], cell_id: int, corner_id: int) -> int:
    """Returns the cell id adjent to the corner of the current cell

    Args:
        cells (List[Cell])
        cell_id (int)
        corner_id (int): id of the corner of the current cell

    Returns:
        adj_cell_id (int): id of the adjacent cell
    """
    adj_cell_id = None
    # cell ids start from 1 -> tranform to index
    cell = list(filter(lambda c: c.cell_id == cell_id, cells))[0]

    coorner_coord = cell.get_corner_coordinates(corner_id)
    # if corner id is 0 or 1 check on the cells on the left
    if corner_id == 0 or corner_id == 1:
        left_neighbours = filter_cells(cell, cells, side="left")
        for left_cell in left_neighbours:
            adj_pt = (coorner_coord[0] - 1, coorner_coord[1])
            if left_cell.contains(adj_pt):
                if adj_cell_id is None:
                    adj_cell_id = left_cell.cell_id
                else:
                    raise AssertionError("More than one adjacent cell found")

    elif corner_id == 2 or corner_id == 3:
        right_neighbours = filter_cells(cell, cells, side="right")
        for right_cell in right_neighbours:
            adj_pt = (coorner_coord[0] + 1, coorner_coord[1])
            if right_cell.contains(adj_pt):
                if adj_cell_id is None:
                    adj_cell_id = right_cell.cell_id
                else:
                    raise AssertionError("More than one adjacent cell found")

    return adj_cell_id


def get_corners_to_adj_cell(
    cells: List[Cell], current_cell_id: int, next_cell_id: int
) -> List[int]:
    """Returns the cell id adjent to the corner of the current cell


    Args:
        adj_matrix (np.array): adjacency matrix of the workspace
        cells (List[Cell]): list of cells
        current_cell_id (int): id of the current cell
        next_cell_id (int): id of the next cell

    Returns:
        corners (List[int]): list of adjacent corners of the next cell

    """
    corners = []
    # for each corner compute the adj cells ids
    for corner in range(4):
        adj_cell_id = get_adj_cell_to_corner(cells, current_cell_id, corner)
        if adj_cell_id is not None:
            if adj_cell_id == next_cell_id:
                corners.append(corner)
    return corners


def dist_intra_cells(
    cells: List[Cell],
    cell_1: Cell,
    cell_2: Cell,
    corner_start_1: int,
    corner_start_2: int,
    coverage_radius: int,
    adj_contraint: bool = False,
):
    """Compute the distance between two points located in two adjancent cells"""

    path_1 = create_path(cell_1, corner_start_1, coverage_radius=coverage_radius)
    length_path_1 = path_length(path_1)
    corner_end_1 = get_path_end_corner(path_1, corner_start_1)
    adj_cell_id = get_adj_cell_to_corner(cells, cell_1.cell_id, corner_end_1)
    
    # if next cell is not given just return the path length
    # TODO: replace with Dijstra or manhattan distance 
    if cell_2:
        intra_path = get_pts_distance(
            path_1[-1], cell_2.get_corner_coordinates(corner_start_2)
        )

        if adj_contraint:
            if not adj_cell_id == cell_2.cell_id:
                intra_path = np.infty
        else:
            list_cells_ids = [cell.cell_id for cell in cells]
            if adj_cell_id not in list_cells_ids:
                intra_path = np.infty

    return intra_path + length_path_1


def get_pts_distance(pt_1: tuple, pt_2: tuple):
    """Compute the distance between two points"""
    return math.sqrt((pt_1[0] - pt_2[0]) ** 2 + (pt_1[1] - pt_2[1]) ** 2)


def order_cells_by_sequences_of_ids(cells: List[Cell], seq_ids: List[int]):
    """Order the cells by the sequence of ids
    Args:
        cells (List[Cell]): list of cells
        seq_ids (List[int]): list of ids of the cells
    """
    ordered_cells = []
    for seq_id in seq_ids:
        for cell in cells:
            if cell.cell_id == seq_id:
                ordered_cells.append(cell)
    return ordered_cells


def shortest_path(
    cells: List[Cell],
    cell_sequence: List[int],
    coverage_radius: int,
    adj_contraint: bool = False,
):
    """Solve the shortest path problem using dynamic programming.

    Args:
        adj_matrix (np.array): adjacency matrix of the workspace
        cells (List[Cell]): cells in the workspace
        cell_sequence (List[int]): sequence of cells to traverse from the first to the last
        coverage_radius (int): coverage radius of the path

    Returns:
        dp (np.array) : contains the dynamic programming solution.
                        Rows are indexed over cells and columns over corners.
    """
    # first comes the last workspace
    # 2, 1, 0
    sequence = list(reversed(cell_sequence))
    ordered_cells = order_cells_by_sequences_of_ids(cells, sequence)

    num_corners = 4
    n = len(cell_sequence)
    dp = np.ones((len(sequence), num_corners)) * np.infty
    # for i in range(4):
    #     path_0i = create_path(
    #         ordered_cells[0], i, coverage_radius=coverage_radius
    #     )
    #     dp[0, i] = path_length(path_0i)
    next_corner_idx = 0
    for i in range(len(cell_sequence) - 1):
        for j in range(num_corners):
            shortest_path = np.infty
            for k in range(num_corners):
                path_ijk = dist_intra_cells(
                    ordered_cells[i:],
                    cells[sequence[i]],
                    cells[sequence[i + 1]],
                    j,
                    k,
                    coverage_radius,
                    adj_contraint=adj_contraint,
                )
                if i > 0:
                    dist = path_ijk + dp[i - 1, k]
                else:
                    dist = path_ijk
                if dist < shortest_path:
                    shortest_path = dist
                    next_corner_idx = k
            dp[i, j] = shortest_path

        if np.allclose(dp[i, :], np.infty):
            raise AssertionError("No path found")

    if np.shape(dp)[0] > 1:
        # boundary condition
        dp[-1, next_corner_idx] = min(dp[-2, :]) + path_length(create_path(ordered_cells[-1], next_corner_idx, coverage_radius=10))
    else:
        dp[-1, next_corner_idx] = path_length(create_path(ordered_cells[-1], next_corner_idx, coverage_radius=10))
    return dp


def reconstruct_path(
    dp_solution: np.ndarray,
    cells: List[Cell],
    cell_sequence: List[int],
    coverage_radius: int,
):
    """Reconstruct the path from the dynamic programming solution
    Args:
        dp_solution (np.array): dynamic programming solution
        cells (List[Cell]): list of cells
        cell_sequence (List[int]): sequence of cells to traverse from the first to the last
        coverage_radius (int): coverage radius of the path

    Returns:
        path (List[tuple]): list of coordinates of the path
    """
    global_path = []
    cell_sequence = list(reversed(cell_sequence))
    print("recostrucing path in order ", cell_sequence)
    ordered_cells = order_cells_by_sequences_of_ids(cells, cell_sequence)
    n = len(cell_sequence)
    for i in range(n):
        corner_start = np.argmin(dp_solution[i, :])
        path_i = create_path(
            ordered_cells[i], corner_start, coverage_radius=coverage_radius
        )
        global_path.append(path_i)
    return global_path

