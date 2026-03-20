# Repository Guidelines

Specific technical reference docs live in `docs/`. This file is for behavioral nudges only.

## Build, Test, and Development Commands

Use `uv` for Python environment management.

- `uv sync` installs the project from `pyproject.toml`.
- `uv add <package>` adds a dependency; regenerate `requirements.txt` with `uv export --no-hashes --frozen --no-emit-project > requirements.txt` in the same change.
- `ruff check scripts` and `ruff format scripts` lint and format Python before review.
- `ty check` type-checks `scripts/` (configured in `[tool.ty]` in pyproject.toml; resolves third-party imports from `../.venv`).
- `prek run` runs all pre-commit hooks (ruff check, ruff format, ty) on staged files. `prek install` wires hooks into `.git/hooks/`.
- `ruff`, `ty`, and `prek` are global tools on PATH (installed via `uv tool`). No venv activation needed to run them.

## Coding Style & Naming Conventions

Follow PEP 8 naming conventions, enforced by ruff's `pep8-naming` (N) rules:
- **Functions and variables**: `snake_case` (e.g., `load_data`, `train_ids`)
- **Classes**: `PascalCase` (e.g., `HNeuronScaler`, `TokenExtractor`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `TARGET_IDX`, `ALPHAS`)
- **ML convention exception**: uppercase `X`, `X_train`, `X_test`, `C`, `C_values` are allowed for scikit-learn feature matrices and regularization parameters (configured in `[tool.ruff.lint.pep8-naming]` ignore-names).

## Testing Guidelines

Tests live in `tests/` and run via `uv run pytest`. Core evaluation helpers (`normalize_answer`, `extract_mc_answer`) have unit tests in `tests/test_utils.py`. When modifying scoring or extraction logic, add or update tests before committing.

## Commit & Pull Request Guidelines

Recent history uses Conventional Commit-style subjects.

## Quantitative Reporting Standards

Every quantitative claim in presentation materials must include uncertainty estimates (bootstrap 95% CIs or binomial proportion CIs). Treat `docs/ci_manifest.json` as the source-of-truth registry. Before finishing any change that touches quantitative reporting surfaces, run `uv run python scripts/audit_ci_coverage.py`.

## Long-Running Jobs

Use `systemd-inhibit` for any GPU job longer than ~20 minutes. Recipes and recovery steps in `docs/tooling-assessment.md`.
- If a script supports `--wandb`, prefer enabling it for long, comparison-heavy, or audit-heavy runs: treat W&B as the visual cockpit for curves, tables, and triage, while local `summary.json` and `*.provenance.json` remain the source of truth.

## GPU Monitoring

Use `nvitop -1` for a one-shot GPU status check before/during long jobs. Details in `docs/tooling-assessment.md`.

## Codebase Prompt Export

`./code2prompt.sh` exports all source code as a single LLM-ready prompt. Pass `--output-file=prompt.txt` or `-c` for clipboard.
