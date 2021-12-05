import matplotlib.pyplot as plt
import numpy as np
from cpp.helpers import * 

def test_h_recursive_map():
    image = recurvise_H_map((400, 600), invert=True, num_recursions=0)
    plt.imshow(image)
    plt.imsave('data/test/recursive_H_map_0.png', image)
    
    image = recurvise_H_map((400, 600), invert=True, num_recursions=1)
    plt.imshow(image)
    plt.imsave('data/test/recursive_H_map_1.png', image)
    
    image = recurvise_H_map((800, 1200), invert=True, num_recursions=2)
    plt.imshow(image)
    plt.imsave('data/test/recursive_H_map_2.png', image)



if __name__ == '__main__':
    test_h_recursive_map()