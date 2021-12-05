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


def plot_path(path: List[Tuple[int, int]], show=False, markersize=10):
    # plot start pt
    plt.plot(path[0][0], path[0][1], "o", markersize=markersize)
    # ploe end pt
    plt.plot(path[-1][0], path[-1][1], "x", markersize=markersize)

    # plot whole path
    x, y, yaw = zip(*path)
    plt.gca().invert_yaxis()
    plt.plot(x, y)

    if show:
        plt.show()


def distance_pts(pt1: Tuple[int, int], pt2: Tuple[int, int]) -> float:
    """Calculate the distance between two points
    """
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def plot_global_path(path: List[List[Tuple[int, int]]], show=True, markersize=10):
    # store segments across cells
    intersection_segments = []

    # plot local cell paths
    for cell_path in path:
        plot_path(cell_path, show=False, markersize=markersize)

    for i in range(len(path) - 1):
        intersection_segments.append([path[i][-1], path[i + 1][0]])

    for segment in intersection_segments:
        x, y, yaw = zip(*segment)
        plt.gca().invert_yaxis()
        plt.plot(x, y, "--", markersize=markersize)
    if show:
        plt.show()


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


def save_path(path: List[tuple], filename: str):
    """Save the path in a csv file
    Args:
        path (List[tuple]): list of coordinates of the path
        filename (str): name of the file
    """
    # convert to numpy array
    path_flattened = []
    for cell_path in path:
        for coordinates in cell_path:
            # convert tuple to np array
            path_flattened.append(coordinates)
    path_np = np.array(path_flattened)
    # save to csv
    np.save(filename, path_np)


if __name__ == "__main__":
    H_map = recurvise_H_map((400, 600), invert=True, num_recursions=1)
    plt.imshow(H_map)
    plt.show()
