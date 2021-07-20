from collections import defaultdict
from cpp.helpers import distance_pts
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt


class Cell(object):
    """
    Current implementation only support trapezoidal cells
    """

    def __init__(self, x_left: int = 0, x_right: int = 0, cell_id: int = 0) -> None:
        super().__init__()
        # mapping x->y, O(1) contains operation
        self.top = {}
        self.bottom = {}
        # two boundaries are always parallel
        self.x_left = x_left
        self.x_right = x_right
        self.left = []
        self.right = []
        self.cell_id = cell_id

    @classmethod
    def from_image(cls, decomposed_image: np.array):
        image_set = set(decomposed_image.flatten())
        num_cells = len(image_set)
        height, width = np.shape(decomposed_image)
        # init cells with extreme values of x_left and x_right
        cells = {cell_id: Cell(width - 1, 0, cell_id) for cell_id in image_set}
        # cells = [Cell(width - 1, 0, i) for i in range(num_cells)]
        for x in range(width):
            for y in range(height):
                cell = cells[decomposed_image[y, x]]
                if x < cell.x_left:
                    cell.x_left = x
                    cell.left = [y]
                elif x == cell.x_left:
                    cell.left.append(y)
                # print(x == cell.x_right)
                if x > cell.x_right:
                    cell.x_right = x
                    cell.right = [y]
                elif x == cell.x_right:
                    cell.right.append(y)

                if x not in cell.bottom:
                    cell.bottom[x] = y
                # is there a better way to avoid the key error
                elif cell.bottom[x] > y:
                    cell.bottom[x] = y

                if x not in cell.top:
                    cell.top[x] = y
                elif cell.top[x] < y:
                    cell.top[x] = y

        # reindex the cells ids (0 cell are obstables and should be taken out) (?)
        for i in image_set:
            cells[i].cell_id -= 1

        cells_no_obstacles = [cell for cell in cells.values() if cell.cell_id != -1]
        return cells_no_obstacles

    def get_corner_coordinates(self, corner_id: int) -> Tuple[int, int]:
        """Returns the coordinates of the corners of the cell

        Args:
            corner_id (int): identifier for the corners 0 (top left), 1 (bottom
                             left), 2 (bottom right), 3 (top right)

        Returns:
            coord (Tuple[int, int]): coordinates of the corner points
        """
        coord = ()
        if corner_id == 0:
            coord = (self.x_left, self.bottom[self.x_left])
        elif corner_id == 1:
            coord = (self.x_left, self.top[self.x_left])
        elif corner_id == 2:
            coord = (self.x_right, self.top[self.x_right])
        elif corner_id == 3:
            coord = (self.x_right, self.bottom[self.x_right])
        else:
            raise AssertionError("There are only 4 corners in a cell")
        return coord

    def contains(self, point_coords: Tuple[int, int]) -> bool:
        """Returns weather a given point belongs to the cell

        Args:
            point_coords (Tuple[int, int]):

        Returns:
            bool: weather a point is part of the cell
        """
        if point_coords[0] < self.x_left or point_coords[0] > self.x_right:
            return False
        if (
            point_coords[1] > self.top[point_coords[0]]
            or point_coords[1] < self.bottom[point_coords[0]]
        ):
            return False
        return True

    def get_center(self) -> Tuple[int, int]:
        """Returns the coordinates of the center of the cell"""
        return (self.x_left + self.x_right) // 2, (
            (self.top[self.x_left] + self.top[self.x_right]) // 2
            + (self.bottom[self.x_left] + self.bottom[self.x_right]) // 2
        ) // 2


def get_distance_between_cells(
    adj_matrix: Dict[int, List[int]], cells: List[Cell]
) -> Dict[int, Dict[int, int]]:
    """Returns the distance between cells
    Args:
        adj_matrix (np.array): adjacency matrix
        cells (List[Cell]): list of cells
    Returns:
        distance (Dict[int, Dict[int, int]]): distance between cells
    """
    cell_dict = {cell.cell_id: cell for cell in cells}
    distance = {}
    for ci in adj_matrix.keys():
        distance[ci] = {}
        for cj in adj_matrix[ci]:
            distance[ci][cj] = distance_pts(cell_dict[ci].get_center(), cell_dict[cj].get_center())
        # remove entries with distance 0
        distance[ci] = {
            k: v for k, v in distance[ci].items() if not np.isclose(v, 0)
        }
    return distance


def plot_cells(list_cells: List[Cell], show=False):
    """Plot the different identified cells

    Args:
        list_cells (List[Cell]): list of separate cells
    """
    boundary_width = 1
    boundary_color = "red"

    for cell in list_cells:
        # plot side boundaries
        print("Boundaries : [%d, %d]" % (cell.x_left, cell.x_right))
        plt.plot(
            cell.x_left * np.ones(len(cell.left)),
            cell.left,
            markersize=boundary_width,
            markeredgecolor=boundary_color,
        )
        plt.plot(
            cell.x_right * np.ones(len(cell.right)),
            cell.right,
            markersize=boundary_width,
        )

        plt.plot(
            cell.bottom.keys(),
            cell.bottom.values(),
            markersize=boundary_width,
        )
        plt.plot(
            cell.top.keys(), cell.top.values(), markersize=boundary_width, marker="o"
        )
        plt.text(
            cell.get_center()[0],
            cell.get_center()[1],
            str(cell.cell_id),
            ha="center",
            va="center",
        )

    if show:
        plt.show()


def filter_cells(cell: Cell, cells: List[Cell], side: str) -> List[Cell]:
    """Filter the cell based on the side

    Args:
        cell (Cell): cell to be filtered
        neightbouring_cells (List[Cell]): list of cells
        side (str): side to be filtered
    Returns:
        filtered_list (List[Cell]): list of filtered cells
    """
    filtered_list = []
    if side == "left":
        for c in cells:
            if c.x_right < cell.x_left:
                filtered_list.append(c)
    elif side == "right":
        for c in cells:
            if c.x_left > cell.x_right:
                filtered_list.append(c)
    else:
        raise ValueError("side should be either left or right")
    return filtered_list
