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
    plt.plot(x, y)
    
    if show:
        plt.show()