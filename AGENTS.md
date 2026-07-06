# Agent Notes

This is a small legacy Python research repo for boustrophedon-style cellular
decomposition and coverage path planning. Keep changes simple and behavior
preserving unless the user explicitly asks for algorithm work.

## Environment

- Preferred setup: `uv sync --locked --extra dev`
- Preferred test command: `uv run pytest -q`
- Example smoke run: `uv run coverage-planning-examples --case map_jpg --strict --output-dir /tmp/coverage_planning_runs`
- The package import name is still `cpp` for compatibility with the existing
  tests and scripts.

## Optional Dependencies

- Gurobi examples are optional. Core tests and examples must not require
  `gurobipy`.
- Install optional optimizer dependencies with
  `uv sync --locked --extra dev --extra opt` only when working on optimizer
  examples.

## Generated Artifacts

- Write generated images, CSVs, and ad hoc run outputs under `/tmp` or `outputs/`.
- Do not commit generated planner output unless the user specifically asks for
  updated fixtures.
- `data/test/map.jpg` is the canonical tracked image fixture. Treat other
  generated planner output as disposable run artifacts.

## Code Style

- Prefer small, direct functions over new framework layers.
- Keep matplotlib usage headless-friendly for tests and CI.
- Preserve the current `cpp.*` public module layout.
