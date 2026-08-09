# Repository Guidelines

## Project Structure & Module Organization

`critiq/` contains the installable Python package. Core evaluation logic lives at the package root, while `critiq/structured/` implements schemas, routing, execution, telemetry, caching, and rubric evolution. Command-line training and annotation utilities are under `critiq/scripts/`. Put research runners and reusable experiment helpers in `experiments/evolving_structured_rubrics/`; keep example configurations in its `configs/` directory. Tests mirror the structured package in `tests/structured/`. Documentation and diagrams belong in `docs/`, `figures/`, and `assets/`; datasets live in `data/`. Treat `output/` and local configuration directories as generated or machine-specific unless a compact result is intentionally reviewed and committed.

## Build, Test, and Development Commands

- `conda activate critiq` activates the required project environment. Run every test, demo, and experiment in this environment.
- `python -m pip install -e .` installs CritiQ in editable mode (Python 3.10+).
- `python -m unittest discover -s tests -p "test_*.py"` runs the full test suite without requiring pytest.
- `python -m unittest tests.structured.test_executor` runs one focused test module.
- `python -m experiments.evolving_structured_rubrics.run_shared_output_pool --help` shows the main experiment interface; start from `configs/shared_output_pool.example.json`.
- `python demo.py` runs the basic demonstration; model-backed workflows may require configured endpoints and credentials.

## Coding Style & Naming Conventions

Use four-space indentation and standard PEP 8 naming: `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants. Preserve the existing preference for type hints, `dataclass` value objects, enums, module docstrings, and explicit validation errors. Keep imports grouped as standard library, third-party, then local. No formatter or linter is configured in `pyproject.toml`, so keep changes consistent with nearby code and avoid unrelated formatting churn.

## Testing Guidelines

Tests use Python's `unittest` framework and follow `test_<behavior>.py`, `Test...`, and `test_...` naming. Add deterministic unit tests beside the closest structured component. Use fake or offline backends instead of live model calls. Cover successful behavior and validation/error paths; there is currently no enforced coverage percentage. Run the full suite before opening a pull request.

## Commit & Pull Request Guidelines

Recent history uses concise Conventional Commit-style subjects such as `feat: add multi-epoch split evolution`, `feat(structured): ...`, and `docs: ...`. Use an imperative subject, add a scope when useful, and keep each commit focused. Pull requests should explain the motivation, summarize code and experiment changes, list verification commands, and link relevant issues or plans. Include plots or screenshots when outputs or diagrams change, but do not commit `.env`, credentials, local endpoints, large datasets, caches, traces, predictions, or raw logs.
