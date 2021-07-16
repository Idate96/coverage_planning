from numpy.lib.function_base import cov
from cpp.cells import Cell, filter_cells
from typing import List, Tuple
import numpy as np
import pickle
import matplotlib.image as mping
import math


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
    x_current = min(cell.x_left + coverage_radius, cell.x_right)
    done = False
    path = []

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
                    # print("options ", cell.top[x] - coverage_radius, cell.bottom[x])
                    y = max(cell.top[x] - coverage_radius, cell.bottom[x])
                else:
                    # print("bottom options ")
                    y = min(cell.bottom[x] + coverage_radius, cell.top[x])
                path.append((x, y))

        else:
            done = True
        x_current = x_next

    # flip the order if the cell had to be covered in the opposite direction
    if horizontal_dir == "left":
        path.reverse()

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


# def get_corners_adj_to_cell(
#     adj_matrix: np.array, cells: List[Cell], cell_id: int
# ) -> List[Tuple[int, int]]:
#     """Finds the adjacency of the corners of a cell

#     Args:
#         adj_matrix (np.array): adjacency matrix of the workspace
#         cells (List[Cell]): list of cells
#         cell_id (int): id of the cell

#     Returns:
#         List[Tuple[int, int]]: list of adjacency ids of the corners of the cell in the following
#                                format (cell_id, corner_id)
#     """
#     # neightbouring cells
#     neighbours = []
#     for i in range(len(cells)):
#         if adj_matrix[cell_id, i]:
#             neighbours.append(i)

#     # corners of the cell
#     corners = []
#     left_to_right = False

#     for i in range(len(neighbours)):
#         # left to right or viceversa coverage
#         if cells[cell_id].x_left > cells[neighbours[i]].x_right:
#             left_to_right = False
#         else:
#             left_to_right = True
#     pass


# def get_adj_corners_next_cell(
#     cells: List[Cell], cell_id: int, next_cell_id: int
# ) -> List[int]:
#     """Finds the adjacency of the corners of a cell wrt to an adjacent cell. It is assumed that the cells are adjancent.

#     Args:
#         cells (List[Cell]): list of cells
#         cell_id (int): id of the cell
#         next_cell_id (int): id of the next cell

#     Returns:
#         corners (List[int]): list of adjacent corners of the next cell
#     """
#     # corners of the cell
#     corners = []

#     # left to right or viceversa coverage
#     if cells[cell_id].x_left > cells[next_cell_id].x_right:
#         left_to_right = False
#     else:
#         left_to_right = True

#     if left_to_right:
#         # check if top of the cell is higher than the top of the next cell
#         if (
#             cells[cell_id].top[cells[cell_id].x_right]
#             >= cells[next_cell_id].top[cells[next_cell_id].x_left]
#         ):
#             corners.append(3)
#         # check if bottom of the cell is lower than the bottom of the next cell
#         if (
#             cells[cell_id].bottom[cells[next_cell_id].x_left]
#             <= cells[next_cell_id].bottom[cells[cell_id].x_left]
#         ):
#             corners.append(2)
#     else:
#         # check if top of the cell is higher than the top of the next cell
#         if (
#             cells[cell_id].top[cells[cell_id].x_left]
#             >= cells[next_cell_id].top[cells[next_cell_id].x_right]
#         ):
#             corners.append(0)
#         # check if bottom of the cell is lower than the bottom of the next cell
#         if (
#             cells[cell_id].bottom[cells[cell_id].x_right]
#             <= cells[next_cell_id].bottom[cells[next_cell_id].x_right]
#         ):
#             corners.append(1)

#     return corners


def get_adj_cell_to_corner(
    adj_matrix: np.array, cells: List[Cell], cell_id: int, corner_id: int
) -> int:
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
    cell_id = cell_id - 1

    # neightbouring cells
    neighbours = []
    for i in range(len(adj_matrix[cell_id, :])):
        if adj_matrix[cell_id, i]:
            neighbours.append(cells[i])

    coorner_coord = cells[cell_id].get_corner_coordinates(corner_id)
    # if corner id is 0 or 1 check on the cells on the left
    if corner_id == 0 or corner_id == 1:
        left_neighbours = filter_cells(cells[cell_id], neighbours, side="left")
        for left_cell in left_neighbours:
            adj_pt = (coorner_coord[0] - 1, coorner_coord[1])
            if left_cell.contains(adj_pt):
                if adj_cell_id is None:
                    adj_cell_id = left_cell.cell_id
                else:
                    raise AssertionError("More than one adjacent cell found")

    elif corner_id == 2 or corner_id == 3:
        right_neighbours = filter_cells(cells[cell_id], neighbours, side="right")
        for right_cell in right_neighbours:
            adj_pt = (coorner_coord[0] + 1, coorner_coord[1])
            if right_cell.contains(adj_pt):
                if adj_cell_id is None:
                    adj_cell_id = right_cell.cell_id
                else:
                    raise AssertionError("More than one adjacent cell found")

    return adj_cell_id


def get_corners_to_adj_cell(
    adj_matrix: np.array, cells: List[Cell], current_cell_id: int, next_cell_id: int
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
        adj_cell_id = get_adj_cell_to_corner(adj_matrix, cells, current_cell_id, corner)
        if adj_cell_id is not None:
            if adj_cell_id == next_cell_id:
                corners.append(corner)
    return corners


def dist_intra_cells(
    adj_matrix: np.array,
    cells: List[Cell],
    cell_1: Cell,
    cell_2: Cell,
    corner_start_1: int,
    corner_start_2: int,
    coverage_radius: int,
):
    """Compute the distance between two points located in two adjancent cells"""

    path_1 = create_path(cell_1, corner_start_1, coverage_radius=coverage_radius)
    length_path_1 = path_length(path_1)
    corner_end_1 = get_path_end_corner(path_1, corner_start_1)
    adj_cell_id = get_adj_cell_to_corner(
        adj_matrix, cells, cell_1.cell_id, corner_end_1
    )
    if adj_cell_id == cell_2.cell_id:
        intra_path = get_pts_distance(
            path_1[-1], cell_2.get_corner_coordinates(corner_start_2)
        )
    else:
        intra_path = np.infty

    return intra_path + length_path_1


def get_pts_distance(pt_1: tuple, pt_2: tuple):
    """Compute the distance between two points"""
    return math.sqrt((pt_1[0] - pt_2[0]) ** 2 + (pt_1[1] - pt_2[1]) ** 2)


def shortest_path(
    adj_matrix: np.array,
    cells: List[Cell],
    cell_sequence: List[int],
    coverage_radius: int,
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
    num_corners = 4
    n = len(cell_sequence)
    dp = np.zeros((len(cell_sequence), num_corners))
    for i in range(4):
        path_0i = create_path(
            cells[cell_sequence[0]], i, coverage_radius=coverage_radius
        )
        dp[0, i] = path_length(path_0i)

    for i in range(1, len(cell_sequence)):
        for j in range(num_corners):
            shortest_path = np.infty
            for k in range(num_corners):
                path_ijk = dist_intra_cells(
                    adj_matrix,
                    cells,
                    cells[sequence[i]],
                    cells[sequence[i - 1]],
                    j,
                    k,
                    coverage_radius,
                )
                dist = path_ijk + dp[i - 1, k]
                if dist < shortest_path:
                    shortest_path = dist
            dp[i, j] = shortest_path

        if np.allclose(dp[i, :], np.infty):
            raise AssertionError("No path found")
    return dp


def reconstruct_path(
    dp_solution: np.ndarray,
    cells: List[Cell],
    cell_sequence: List[int],
    coverage_radius: int,
):
    global_path = []
    cell_sequence = list(reversed(cell_sequence))
    n = len(cell_sequence)
    for i in range(n - 1, -1, -1):
        corner_start = np.argmin(dp_solution[i, :])
        print("corner ", corner_start)
        print("cell ", cell_sequence[i])
        path_i = create_path(
            cells[cell_sequence[i]], corner_start, coverage_radius=coverage_radius
        )
        global_path.append(path_i)
    return global_path
