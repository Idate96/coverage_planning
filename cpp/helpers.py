import matplotlib.pyplot as plt
import numpy as np

def plot_decomposed_image(image):
    max_v, min_v = np.max(image), np.min(image)
    scaled_img =  255/(max_v - min_v) * (image - min_v)
    plt.imshow(scaled_img)
    plt.show()

