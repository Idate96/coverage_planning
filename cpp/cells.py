import numpy as np
from typing import List
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
        num_cells = len(set(decomposed_image.flatten()))
        height, width = np.shape(decomposed_image)
        cells = [Cell(width - 1, 0, i) for i in range(num_cells)]
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
                    cell.bottom

                if x not in cell.top:
                    cell.top[x] = y
                elif cell.top[x] < y:
                    cell.top[x] = y
        return cells


def plot_cells(list_cells: List[Cell]):
    """Plot the different identified cells

    Args:
        list_cells (List[Cell]): list of separate cells
    """
    boundary_width = 2
    boundary_color = "red"

    for cell in list_cells:
        # plot side boundaries
        print(cell.x_right)
        print(cell.right)
        plt.plot(
            cell.x_left * np.ones(len(cell.left)),
            cell.left,
            markersize=boundary_width,
            marker="o",
        )
        plt.plot(
            cell.x_right * np.ones(len(cell.right)),
            cell.right,
            markersize=boundary_width,
            marker="o",
        )

        plt.plot(cell.bottom.keys(), cell.bottom.values(), markersize=boundary_width, marker="o")
        plt.plot(cell.top.keys(), cell.top.values(), markersize=boundary_width, marker="o")

    plt.show()