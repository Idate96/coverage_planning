from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from typing import List 

def plot_decomposed_image(image):
    max_v, min_v = np.max(image), np.min(image)
    scaled_img =  255/(max_v - min_v) * (image - min_v)
    plt.imshow(scaled_img)
    plt.show()


def plot_path(path: List[Tuple[int, int]], show=False):
    # plot start pt 
    plt.plot(path[0][0], path[0][1], 'o')
    # ploe end pt 
    plt.plot(path[-1][0], path[-1][1], 'x')

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
        intersection_segments.append([path[i][-1], path[i+1][0]])
    
    for segment in intersection_segments:
        x, y = zip(*segment)
        plt.gca().invert_yaxis()
        plt.plot(x, y, '--')
    if show:
        plt.show()

