segments = [(0, 1), (3, 4)]
current_cells = [0, 1]

for cell, slice in zip(current_cells, segments):
    print(cell, slice)