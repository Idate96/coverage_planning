import numpy as np
from cpp.helpers import * 

def test_h_recursive_map():
    image = recurvise_H_map((400, 600), invert=True, num_recursions=0)
    assert image.shape == (400, 600)
    assert np.isclose(image.min(), 0)
    assert np.isclose(image.max(), 1)
    
    image = recurvise_H_map((400, 600), invert=True, num_recursions=1)
    assert image.shape == (400, 600)
    assert np.isclose(image.min(), 0)
    assert np.isclose(image.max(), 1)
    
    image = recurvise_H_map((800, 1200), invert=True, num_recursions=2)
    assert image.shape == (800, 1200)
    assert np.isclose(image.min(), 0)
    assert np.isclose(image.max(), 1)



if __name__ == '__main__':
    test_h_recursive_map()
