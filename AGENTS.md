# Repository Guidelines

If anything about this repository is surprising, brittle, or easy to get wrong, add a short, specific note to this file when you discover it. Capture recurring quirks, unexpected workflows, hidden dependencies, stale references, or repeatable errors so future agents do not waste time rediscovering them.

## Project Structure & Module Organization
Core code lives in `scripts/`, with one CLI per pipeline stage: response collection, answer-token extraction, balanced ID sampling, activation extraction, classifier training, and intervention helpers. Put new research utilities in `scripts/` unless they justify a reusable package.

`data/` holds local datasets, sampled JSONL outputs, and review artifacts. Keep large generated files out of commits unless they are small, reproducible examples such as `data/examples/`. `docs/` contains research notes and deep-research writeups; the root `README.md` is the canonical pipeline overview.

## Build, Test, and Development Commands
Use `uv` for Python environment management.

- `uv sync` installs the project from `pyproject.toml`.
- `uv add <package>` adds a dependency; keep `requirements.txt` aligned in the same change because the repo still documents runtime deps there.
- `uv run python scripts/collect_responses.py --help` checks a script entrypoint and available flags.
- `uv run python scripts/classifier.py --model_path /path/to/model ...` runs the training stage.
- `uv run ruff check scripts` and `uv run ruff format scripts` lint and format Python before review.
- `uv run ty check` performs a lightweight type pass if you introduce nontrivial new logic.

## Coding Style & Naming Conventions
Follow existing Python script style: 4-space indentation, module-level helper functions, and explicit `argparse` flags for CLIs. Prefer `snake_case` for functions, variables, file names, and JSON keys. Keep scripts composable: inputs and outputs should be passed by path flags, not hard-coded.

## Testing Guidelines
There is no formal `tests/` suite yet. Treat validation like a lab notebook: run `--help`, then exercise changes on a small sample slice such as `--max_samples 5`, and record any important assumptions in the PR. If you add tests, place them under `tests/` and name files `test_<feature>.py`.

## Commit & Pull Request Guidelines
Recent history uses Conventional Commit-style subjects such as `feat(scripts): add transformers backend` and `docs(setup): add compatibility notes`. Follow that pattern with clear scopes: `feat`, `fix`, `docs`, `build`, or `refactor`.

PRs should explain the pipeline stage affected, the exact commands run for verification, any dataset or model prerequisites, and sample output paths when behavior changes. Include screenshots only when documentation or rendered artifacts change.
