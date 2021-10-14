from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from typing import List


def plot_decomposed_image(image, show=True):
    max_v, min_v = np.max(image), np.min(image)
    scaled_img = 255 / (max_v - min_v) * (image - min_v)
    plt.imshow(scaled_img)
    if show:
        plt.show()


def plot_path(path: List[Tuple[int, int]], show=False):
    # plot start pt
    plt.plot(path[0][0], path[0][1], "o")
    # ploe end pt
    plt.plot(path[-1][0], path[-1][1], "x")

    # plot whole path
    x, y = zip(*path)
    plt.gca().invert_yaxis()
    plt.plot(x, y)

    if show:
        plt.show()


def plot_global_path(path: List[List[Tuple[int, int]]], show=True):
    # store segments across cells
    intersection_segments = []

    # plot local cell paths
    for cell_path in path:
        plot_path(cell_path, show=False)

    for i in range(len(path) - 1):
        intersection_segments.append([path[i][-1], path[i + 1][0]])

    for segment in intersection_segments:
        x, y = zip(*segment)
        plt.gca().invert_yaxis()
        plt.plot(x, y, "--")
    if show:
        plt.show()

def write_path_to_csv(path: List[List[Tuple[int, int, int]]], filename: str):
    with open(filename, "w") as f:
        for i in range(len(path)):
            f.write(str(path[i][0]) + "," + str(path[i][1]) + "," + str(path[i][2]) + "\n")


def transform_coordinates(path: List[Tuple[int, int, int]], grid_size: Tuple[float, float], resolution: Tuple[int, int]):
    """Map the path x and y coordinates from image space to the physical space
    Args:
        path: nested list of (x, y, yaw) cooridnates 
        grid_size: size of the grid in meters
        resolution: number of pixels in the x and y dimensions
    Returns:
        transformed_path: nested list of (x, y, yaw) cooridnates 
    """
    transformed_path = []
    for i in range(len(path)):
        x = path[i][0] * grid_size[0] / resolution[0]
        y = path[i][1] * grid_size[1] / resolution[1]
        yaw = path[i][2]
        transformed_path.append((x, y, yaw))
    return transformed_path


def generate_H_map(image_size: Tuple[int, int], invert=False) -> np.ndarray:
    """Generate a occupancy map with an obstacle that has the shape of the letter H

    Args:
        image_size (Tuple[int, int]): size in pixels of the image

    Returns:
        H_map (np.ndarray): occupancy map
    """
    # flip order of image_size
    image_size = (image_size[1], image_size[0])
    H_map = np.zeros(image_size)

    v_size = image_size[0] // 4
    v_start_horizontal_bar = image_size[0] // 2 - image_size[0] // 9
    v_end_horizontal_bar = image_size[0] // 2 + image_size[0] // 9
    h_size = image_size[1] // 5
    # add vertical rectangles
    H_map[v_size : 3 * v_size, h_size : 2 * h_size] = 1
    H_map[v_size : 3 * v_size, 3 * h_size : 4 * h_size] = 1
    # add horizontal rectangles
    H_map[v_start_horizontal_bar:v_end_horizontal_bar, 2 * h_size : 3 * h_size] = 1

    # rotate 90 degrees
    H_map = np.rot90(H_map)

    # invert H_map
    if invert:
        H_map = 1 - H_map

    return H_map


def recurvise_H_map(
    image_size: Tuple[int, int], invert=False, num_recursions=np.infty
) -> np.ndarray:
    """Generate a recursive occupancy map with an obstacle that has the shape of the letter H.
    in each of the holes of the letter H, other letter Hs are embedded until the thickness of the letter reaches 1.

    Args:
        image_size (Tuple[int, int]): size in pixels of the image

    Returns:
        H_map (np.ndarray): occupancy map
    """
    # flip order of image_size
    image_size = (image_size[1], image_size[0])
    H_map = np.zeros(image_size)

    v_size = image_size[0] // 4
    v_start_horizontal_bar = (
        image_size[0] // 2 - image_size[0] // 10 + image_size[0] // 20
    )
    v_end_horizontal_bar = (
        image_size[0] // 2 + image_size[0] // 10 - image_size[0] // 20
    )
    h_size = image_size[1] // 5
    # add vertical rectangles
    H_map[v_size : 3 * v_size, h_size : 2 * h_size] = 1
    H_map[v_size : 3 * v_size, 3 * h_size : 4 * h_size] = 1
    # add horizontal rectangles
    H_map[v_start_horizontal_bar:v_end_horizontal_bar, 2 * h_size : 3 * h_size] = 1

    red_image_size = (3 * v_size - v_end_horizontal_bar, h_size)

    # invert H_map
    if invert:
        H_map = 1 - H_map

    # termination condition
    if image_size[1] < 5 or image_size[0] < 20 or num_recursions == 0:
        return np.rot90(H_map)
    else:
        H_map[
            v_end_horizontal_bar : 3 * v_size, 2 * h_size : 3 * h_size
        ] = recurvise_H_map(
            red_image_size, invert=invert, num_recursions=num_recursions - 1
        )

        H_map[v_size:v_start_horizontal_bar, 2 * h_size : 3 * h_size] = recurvise_H_map(
            red_image_size, invert=invert, num_recursions=num_recursions - 1
        )

    return np.rot90(H_map)


if __name__ == "__main__":
    H_map = recurvise_H_map((400, 600), invert=True, num_recursions=1)
    plt.imshow(H_map)
    plt.show()
