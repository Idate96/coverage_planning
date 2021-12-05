import numpy as np


def create_straight_test_path():
    """
    Creates a straight line path for testing purposes.
    """
    path = np.array([[0,0,0], [1,0,0], [2,0,0], [3,0,0], [4,0,0], [5,0,0], [6,0,0], [7,0,0], [8,0,0], [9,0,0]])
    return path


def create_straight_path(initial_position, final_position, num_points):
    """
    Computer the coordinates of a straight line between the initial position and the final position
    Args:
        initial_position: inial position in the form [x,y,z]
        final_position: final position in the form [x,y,z]
        num_points: number of points to be generated
    Returns:
        path: a numpy array with the coordinates of the path
    """
    path = np.zeros((num_points, 3))
    path[0, :] = initial_position
    path[-1, :] = final_position
    path[1:-1, :] = np.linspace(initial_position, final_position, num_points - 2, endpoint=False)
    return path[1:, :]


def set_heading_to_zero(path_filename):
    path = np.load(path_filename)
    path[:, 2] = 0
    np.save(path_filename[:-4] + '_zero_heading.npy', path)


if __name__ == "__main__":
    filepath = "data/path.npy"
    set_heading_to_zero(filepath)
    path = create_straight_path([10,10,0], [60,10,0], 4)
    print(path)
    np.save("tests/data/straight_path.npy", path)