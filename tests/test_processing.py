from cpp.processing import *
import numpy as np


def test_processing_transform():
    # path made up by the corners of the image
    path = np.array([[0, 0, np.pi/2], [60.6, 0, np.pi/2], [0, 86.2, 3/2 * np.pi], [60.6, 86.2, 3/2 * np.pi]])
    # expected output
    expected = np.array([[11.7, 54.11], [11.7, -6.4], [-74.8, 54.11], [-74.8, -6.4]])
    map_path = image_to_map_frame(path, m_P_mp=np.array([11.7, 54.1]))
    expected_yaw = np.array([np.pi, np.pi, 0, 0])
    assert np.allclose(map_path[:, :2], expected, atol=1)


if __name__ == "__main__":
    test_processing_transform()