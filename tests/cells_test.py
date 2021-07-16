from numpy.core.fromnumeric import sort
from cpp.cells import *
from cpp.bsd import *
import matplotlib.image as mpimg
from cpp.helpers import *


def test_unique_cell():
    """Testes the trivial case with a single cell"""
    simple_cell = np.zeros((10, 10), dtype=np.int)
    cell = Cell.from_image(simple_cell)[0]

    target_cell = Cell(0, 10)
    target_cell.left = list(range(10))
    target_cell.right = list(range(10))
    target_cell.x_left = 0
    target_cell.x_right = 9
    for i in range(10):
        target_cell.top[i] = 9
        target_cell.bottom[i] = 0

    assert sorted(target_cell.left) == sorted(cell.left)
    assert sorted(target_cell.right) == sorted(cell.right)
    assert target_cell.x_left == cell.x_left
    assert target_cell.x_right == cell.x_right
    shared_items_top = {
        k: target_cell.top[k]
        for k in target_cell.top
        if k in cell.top and target_cell.top[k] == cell.top[k]
    }
    shared_items_bottom = {
        k: target_cell.top[k]
        for k in target_cell.top
        if k in cell.top and target_cell.top[k] == cell.top[k]
    }
    assert len(shared_items_bottom) == len(cell.bottom)
    assert len(shared_items_top) == len(cell.bottom)


def test_simple_cell():
    simple_cell = np.zeros((10, 10), dtype=np.int)
    simple_cell[:, :5] = 1
    cells = Cell.from_image(simple_cell)

    target_cell = Cell(0, 10)
    target_cell.left = list(range(10))
    target_cell.right = list(range(10))
    target_cell.x_left = 0
    target_cell.x_right = 4
    for i in range(5):
        target_cell.top[i] = 9
        target_cell.bottom[i] = 0

    cell = cells[1]

    assert cell.cell_id == 1
    assert sorted(target_cell.left) == sorted(cell.left)
    assert sorted(target_cell.right) == sorted(cell.right)
    assert target_cell.x_left == cell.x_left
    assert target_cell.x_right == cell.x_right
    shared_items_top = {
        k: target_cell.top[k]
        for k in target_cell.top
        if k in cell.top and target_cell.top[k] == cell.top[k]
    }
    shared_items_bottom = {
        k: target_cell.bottom[k]
        for k in target_cell.bottom
        if k in cell.bottom and target_cell.bottom[k] == cell.bottom[k]
    }
    assert len(shared_items_bottom) == len(cell.bottom)
    assert len(shared_items_top) == len(cell.top)

    target_cell = Cell(0, 10)
    target_cell.left = list(range(10))
    target_cell.right = list(range(10))
    target_cell.x_left = 5
    target_cell.x_right = 9
    for i in range(5, 10):
        target_cell.top[i] = 9
        target_cell.bottom[i] = 0

    cell = cells[0]

    assert cell.cell_id == 0
    assert sorted(target_cell.left) == sorted(cell.left)
    assert sorted(target_cell.right) == sorted(cell.right)
    assert target_cell.x_left == cell.x_left
    assert target_cell.x_right == cell.x_right
    shared_items_top = {
        k: target_cell.top[k]
        for k in target_cell.top
        if k in cell.top and target_cell.top[k] == cell.top[k]
    }
    shared_items_bottom = {
        k: target_cell.bottom[k]
        for k in target_cell.bottom
        if k in cell.bottom and target_cell.bottom[k] == cell.bottom[k]
    }
    assert len(shared_items_bottom) == len(cell.bottom)
    assert len(shared_items_top) == len(cell.top)


def test_simple_cell_2():
    simple_cell = np.zeros((10, 10), dtype=np.int)
    simple_cell[:5, :] = 1
    cells = Cell.from_image(simple_cell)

    target_cell = Cell(0, 10)
    target_cell.left = list(range(5))
    target_cell.right = list(range(5))
    target_cell.x_left = 0
    target_cell.x_right = 9
    for i in range(10):
        target_cell.top[i] = 4
        target_cell.bottom[i] = 0

    cell = cells[1]

    assert cell.cell_id == 1
    assert sorted(target_cell.left) == sorted(cell.left)
    assert sorted(target_cell.right) == sorted(cell.right)
    assert target_cell.x_left == cell.x_left
    assert target_cell.x_right == cell.x_right
    shared_items_top = {
        k: target_cell.top[k]
        for k in target_cell.top
        if k in cell.top and target_cell.top[k] == cell.top[k]
    }
    shared_items_bottom = {
        k: target_cell.bottom[k]
        for k in target_cell.bottom
        if k in cell.bottom and target_cell.bottom[k] == cell.bottom[k]
    }
    assert len(shared_items_bottom) == len(cell.bottom)
    assert len(shared_items_top) == len(cell.top)


def test_cell_boundaries():
    expected_boundaries = [
        (0, 0),
        (0, 47),
        (48, 200),
        (48, 108),
        (109, 200),
        (109, 473),
        (201, 473),
        (474, 516),
        (517, 601),
        (517, 601),
        (602, 639)
    ]
    image = mpimg.imread("data/test/map.jpg")
    # original image is black and white anyway
    binary_image = image[:, :, 0] > 127
    decomposed = create_mask(binary_image)
    cells = Cell.from_image(decomposed)
    for i in range(1, len(cells)):
        assert expected_boundaries[i] == (cells[i].x_left, cells[i].x_right)


def test_plotting():
    simple_cell = np.zeros((10, 10), dtype=np.int)
    simple_cell[:, :5] = 1
    print(simple_cell)
    cells = Cell.from_image(simple_cell)
    # plot_cells(cells)


def test_plotting_map():
    image = mpimg.imread("data/test/map.jpg")
    # original image is black and white anyway
    binary_image = image[:, :, 0] > 127
    decomposed = create_mask(binary_image)
    cells = Cell.from_image(decomposed)
    # plot_cells(cells)


def test_simple_plotting_map():
    image = mpimg.imread("data/test/map.jpg")
    # original image is black and white anyway
    binary_image = image[:, :, 0] > 127
    decomposed = create_mask(binary_image)
    # plot_decomposed_image(decomposed)


def test_bsd_path():
    image = mpimg.imread("data/test/map.jpg")
    print(image.shape)
    binary_image = image[:, :, 0] > 127
    decomposed = create_mask(binary_image)
    cells = Cell.from_image(decomposed)
    print(len(cells))
    for i in range(1, len(cells)):
        path = create_path(cells[i], 0, 0, coverage_radius=10)
        plot_path(path, show=True)
    plt.show()


def test_filter_cells():
    image = mpimg.imread("data/test/map.jpg")
    binary_image = image[:, :, 0] > 127
    decomposed = create_mask(binary_image)
    cells = Cell.from_image(decomposed)
    binary_image = image[:, :, 0] > 127
    decomposed = create_mask(binary_image)
    cells = Cell.from_image(decomposed)

    # filter cells
    left_cells = filter_cells(cells[5], cells=cells, side="left")
    assert [cell.cell_id for cell in left_cells] == [1, 2, 3, 4]
    right_cells = filter_cells(cells[5], cells=cells, side="right")
    assert [cell.cell_id for cell in right_cells] == [7, 8, 9, 10]



if __name__ == "__main__":
    test_bsd_path()
