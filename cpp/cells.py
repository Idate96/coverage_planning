import numpy as np
from typing import List

class Cell(object): 
    def __init__(self, x_left:int=0, x_right:int=0) -> None:
        super().__init__()
        # mapping x->y, O(1) contains operation
        self.top = {}
        self.bottom = {}
        # two boundaries are always parallel 
        self.x_left = x_left
        self.x_right = x_right
        self.left = []
        self.right = []

    @classmethod
    def from_image(decomposed_image: np.array):
        num_cells = np.max(decomposed_image)
        width, height = np.shape(decomposed_image)
        cells =  [Cell(0, width) for i in range(num_cells)]
        for x in range(width):
            for y in range(height):
                cell = cells[decomposed_image[x, y]]
                if x < cell.x_left: 
                    cell.x_left = x
                    cell.left = [y]
                elif x == cell.x_left:
                    cell.left.append(y)
                
                if x > cell.x_right:
                    cell.x_right = x
                    cell.right = [y]
                elif x == cell.right:
                    cell.right.append(y)

                if (x not in cell.bottom):
                    cell.bottom[x] = y
                
                elif cell.bottom[x] > y:
                    cell.bottom[y]


                

