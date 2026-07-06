# Coverage Planning

Legacy Python experiments for cellular decomposition and coverage path planning.
The package import name remains `cpp` for compatibility with the original code.

## Setup

The reproducible path is `uv`:

```bash
uv sync --locked --extra dev
```

Plain `pip` also works:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

## Verify

```bash
uv run pytest -q
```

Gurobi-based optimizer examples are optional and skipped unless `gurobipy` is
installed:

```bash
uv sync --locked --extra dev --extra opt
```

## Run Examples

Generate decompositions, paths, and a summary CSV for the bundled map fixture:

```bash
uv run coverage-planning-examples --case map_jpg --strict --output-dir /tmp/coverage_planning_runs
```

To inspect all bundled geometries, omit `--case map_jpg`. The H-shaped examples
currently expose legacy planner limitations and are reported as partial or
failed in `summary.csv`. The runner writes PNG and CSV artifacts to the output
directory instead of modifying tracked fixture images.
