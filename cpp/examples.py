from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from cpp.bsd import (
    create_global_adj_matrix,
    get_directed_global_adj_matrix,
    reconstruct_path,
    shortest_path,
)
from cpp.cells import Cell, plot_cells
from cpp.dfs_tree import Graph, adj_matrix_to_dict
from cpp.helpers import (
    generate_H_map,
    plot_decomposed_image,
    plot_global_path,
    recurvise_H_map,
)


def _load_binary_image(path: Path) -> np.ndarray:
    image = mpimg.imread(path)
    if image.ndim == 3:
        return image[:, :, 0] > 150
    return image > 150


def _visit_order(adj_matrix: np.ndarray) -> list[int]:
    graph = Graph(adj_matrix_to_dict(adj_matrix))
    if not graph.adj:
        return []

    tree_edges = graph.DFS_tree(0)
    graph_tree = Graph()
    graph_tree.add_edges(tree_edges, directed=False)
    _, new_root, _ = graph_tree.find_diameter(0)

    tree_edges = graph_tree.DFS_tree(new_root)
    longest_tree = Graph()
    longest_tree.add_edges(tree_edges)
    return longest_tree.post_order_traversal(new_root)


def _write_path_csv(path: list[list[tuple]], filename: Path) -> None:
    with filename.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell_index", "x", "y", "yaw"])
        for cell_index, cell_path in enumerate(path):
            for point in cell_path:
                yaw = point[2] if len(point) > 2 else ""
                writer.writerow([cell_index, point[0], point[1], yaw])


def _save_decomposition(decomposed_image: np.ndarray, cells: list[Cell], filename: Path) -> None:
    plt.figure()
    plot_decomposed_image(decomposed_image, show=False)
    plot_cells(cells, show=False)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def _save_path(path: list[list[tuple]], filename: Path) -> None:
    plt.figure()
    plot_global_path(path, show=False)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


CaseFactory = Callable[[Path], np.ndarray]


def _run_case(
    name: str,
    binary_image: np.ndarray,
    output_dir: Path,
    coverage_radius: int,
    directed: bool,
) -> dict[str, str | int]:
    graph_kind = "directed" if directed else "undirected"
    row: dict[str, str | int] = {
        "case": name,
        "graph": graph_kind,
        "status": "failed",
        "cells": 0,
        "visited_cells": 0,
        "waypoints": 0,
        "error": "",
    }

    try:
        if directed:
            adj_matrix, decomposed_image = get_directed_global_adj_matrix(binary_image)
        else:
            adj_matrix, decomposed_image = create_global_adj_matrix(binary_image)

        cells = Cell.from_image(decomposed_image)
        visit_order = _visit_order(adj_matrix)
        row.update({"cells": len(cells), "visited_cells": len(visit_order)})

        prefix = f"{name}_{graph_kind}"
        _save_decomposition(decomposed_image, cells, output_dir / f"{prefix}_decomposed.png")

        dp_solution = shortest_path(
            adj_matrix,
            cells,
            visit_order,
            coverage_radius=coverage_radius,
        )
        global_path = reconstruct_path(
            dp_solution,
            cells,
            visit_order,
            coverage_radius=coverage_radius,
        )
        _save_path(global_path, output_dir / f"{prefix}_path.png")
        _write_path_csv(global_path, output_dir / f"{prefix}_path.csv")

        status = "ok"
        error = ""
        if len(visit_order) != len(cells):
            status = "partial"
            error = "visited subset of cells"

        row.update(
            {
                "status": status,
                "waypoints": sum(len(cell_path) for cell_path in global_path),
                "error": error,
            }
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"

    return row


def _case_factories() -> dict[str, CaseFactory]:
    return {
        "map_jpg": lambda map_image: _load_binary_image(map_image),
        "h_map": lambda _: generate_H_map((400, 600), invert=True),
        "recursive_h_0": lambda _: recurvise_H_map((400, 600), invert=True, num_recursions=0),
        "recursive_h_1": lambda _: recurvise_H_map((400, 600), invert=True, num_recursions=1),
    }


def _selected_cases(map_image: Path, selected_case: str) -> Iterable[tuple[str, np.ndarray]]:
    factories = _case_factories()
    selected = factories.keys() if selected_case == "all" else [selected_case]
    for name in selected:
        yield name, factories[name](map_image)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run bundled coverage planning examples.")
    parser.add_argument(
        "--map-image",
        type=Path,
        default=repo_root / "data" / "test" / "map.jpg",
        help="Path to the map image used by the map_jpg example.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/coverage_planning_runs"),
        help="Directory for generated images, paths, and summary CSV.",
    )
    parser.add_argument(
        "--coverage-radius",
        type=int,
        default=10,
        help="Coverage radius in pixels.",
    )
    parser.add_argument(
        "--case",
        choices=["all", *_case_factories().keys()],
        default="all",
        help="Run one bundled case, or all cases.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code if any selected case is not fully covered.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, binary_image in _selected_cases(args.map_image, args.case):
        for directed in (False, True):
            rows.append(
                _run_case(
                    name=name,
                    binary_image=binary_image,
                    output_dir=args.output_dir,
                    coverage_radius=args.coverage_radius,
                    directed=directed,
                )
            )

    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        fieldnames = ["case", "graph", "status", "cells", "visited_cells", "waypoints", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            "{case} ({graph}): {status}, cells={cells}, visited={visited_cells}, "
            "waypoints={waypoints} {error}".format(**row)
        )
    print(f"Wrote {summary_path}")
    if args.strict and any(row["status"] != "ok" for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
