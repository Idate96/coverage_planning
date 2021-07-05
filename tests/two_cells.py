import gurobipy as gp
from gurobipy import GRB
from matplotlib.pyplot import plot
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt 

pts_coordinates = np.zeros(shape=(2, 4))
num_cells = 2
num_corners = 4

def get_coordinates_square_vertices(center: Tuple[int, int], edge_size:int):
    """
    :param center: center of the square
    :param edge_size: edge size of the square
    :return: coordinates of the square vertices
    """
    x_min = center[0] - edge_size
    x_max = center[0] + edge_size
    y_min = center[1] - edge_size
    y_max = center[1] + edge_size
    return np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]])

centers = [(1, 1), (3, 2)]
edge_sizes = [1, 1]

list_vertices = [get_coordinates_square_vertices(center, edge_size) for center, edge_size in zip(centers, edge_sizes)]

def plot_array_of_points(pts_coordinates):
    import matplotlib.pyplot as plt
    plt.plot(pts_coordinates[:, 0], pts_coordinates[:, 1], 'o')

plot_array_of_points(list_vertices[0])
plot_array_of_points(list_vertices[1])
plt.show()

def intercell_dist(i, j, k, l):
    pass

try:
    m = gp.Model("two_cells")

    # Create variables
    x = m.addVar( vtype=GRB.BINARY, name="x")
    y = m.addVar( vtype=GRB.BINARY, name="y")






except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError:
    print('Encountered an attribute error')