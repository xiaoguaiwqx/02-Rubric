# Repository Guidelines

## Project Structure & Module Organization

`critiq/` contains the installable Python package, with Agent and parsing primitives at the package root and Rubric schemas, validation, backend specifications, and telemetry in `critiq/structured/`. The current method, CLI, and reusable research helpers live in `experiments/evolving_structured_rubrics/`; example configurations are in its `configs/` directory. Tests are in `tests/structured/`. Documentation and diagrams belong in `docs/`, `figures/`, and `assets/`; datasets live in `data/`. Treat `output/` and local configuration directories as generated or machine-specific unless a compact result is intentionally reviewed and committed. The complete historical experiment tree is preserved on `codex/subtree-local-reflection`.

## Build, Test, and Development Commands

- `conda activate critiq` activates the required project environment. Run every test, demo, and experiment in this environment.
- `python -m pip install -e ".[data]"` installs CritiQ and parquet readers in editable mode (Python 3.10+).
- `python -m unittest discover -s tests -p "test_*.py"` runs the full test suite without requiring pytest.
- `python -m unittest tests.structured.test_current_core tests.structured.test_subtree_local_reflection` runs current-method offline tests.
- `python -m experiments.evolving_structured_rubrics.current_experiment --help` shows the current experiment stages; start from `configs/current_generated_roots.example.json`.

## Coding Style & Naming Conventions

Use four-space indentation and standard PEP 8 naming: `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants. Preserve the existing preference for type hints, `dataclass` value objects, enums, module docstrings, and explicit validation errors. Keep imports grouped as standard library, third-party, then local. No formatter or linter is configured in `pyproject.toml`, so keep changes consistent with nearby code and avoid unrelated formatting churn.

## Research-code scope and implementation restraint

This repository is primarily a research codebase. The main objective is to test algorithmic ideas and hypotheses, not to build production services. For experiment requests, first identify the smallest change that isolates the intended scientific variable and reuse the existing runner, configuration, cache, and reporting paths. Do not turn a research experiment into a production platform by adding deployment frameworks, elaborate preflight systems, identity/fingerprint binding, migration layers, compatibility wrappers, or broad defensive validation unless the user explicitly asks for them or a concrete existing bug shows they are required for the experiment to run correctly.

Keep the algorithmic path in focus: do not change thresholds, acceptance rules, scheduling, prompts, parsers, data splits, or evaluation semantics when the request is only a model, endpoint, or parameter substitution. Preserve necessary checks that catch real configuration errors or invalidate the scientific comparison, but prefer a clear, local validation and a short experiment note over speculative infrastructure. When a proposed safeguard expands the scope beyond the stated research question, explain the tradeoff and ask before implementing it.

## Testing Guidelines

Tests use Python's `unittest` framework and follow `test_<behavior>.py`, `Test...`, and `test_...` naming. Add deterministic unit tests beside the closest structured component. Use fake or offline backends instead of live model calls. Cover successful behavior and validation/error paths; there is currently no enforced coverage percentage. Run the full suite before opening a pull request.

## Research document and artifact hygiene

Use `docs/experiments/README.md` as the experiment index. Update an existing experiment protocol/result section instead of creating timestamped PLAN/TRACKER/REVIEW copies. Keep pre-run frozen protocols distinct from observed results; use persisted run artifacts as the authoritative execution status. Temporary agent drafts and review rounds belong in `.local/`. Preserve negative results and necessary reproduction metadata. Do not delete a historical runner or artifact before checking downstream references. Never mutate another experiment module's global settings to select a protocol; pass an explicit protocol/configuration instead. Historical compatibility constraints must not be silently relaxed during refactoring.

## Commit & Pull Request Guidelines

Recent history uses concise Conventional Commit-style subjects such as `feat: add multi-epoch split evolution`, `feat(structured): ...`, and `docs: ...`. Use an imperative subject, add a scope when useful, and keep each commit focused. Pull requests should explain the motivation, summarize code and experiment changes, list verification commands, and link relevant issues or plans. Include plots or screenshots when outputs or diagrams change, but do not commit `.env`, credentials, local endpoints, large datasets, caches, traces, predictions, or raw logs.
