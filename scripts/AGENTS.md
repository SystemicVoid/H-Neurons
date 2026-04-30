# Scripts Guide

Use this file for script and evaluation-helper work. For long-run orchestration, read `scripts/infra/AGENTS.md`. For `scripts.lib.pipeline` guard changes, read `scripts/lib/AGENTS.md`.

## Defaults

- Use `uv run python ...` for repo-local execution.
- Add Python dependencies with `uv add <package>` from the repo root; update `requirements.txt` with `uv export --no-hashes --frozen --no-emit-project > requirements.txt`.
- Run focused tests for touched behavior, then `ruff check scripts`, `ruff format scripts`, and `ty check` before review or commit.
- Prefer existing helpers in `scripts/utils.py` and `scripts/lib/` over local reimplementations.

## Naming

Follow PEP 8 naming conventions enforced by ruff's `pep8-naming` rules:

- Functions and variables: `snake_case`.
- Classes: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- ML convention exception: uppercase `X`, `X_train`, `X_test`, `C`, and `C_values` are allowed for scikit-learn feature matrices and regularization parameters.

## Evaluation and Claim-Bearing Scripts

- New claimable jailbreak runs should use `--run_profile canonical` or canonical-equivalent explicit decode values: `do_sample=True`, `temperature=0.7`, `top_k=20`, `top_p=0.8`, `max_new_tokens=5000`.
- Noncanonical decode overrides belong to exploratory or throughput experiments and must not be mixed into claim-bearing comparisons.
- Greedy jailbreak decode is valid for deterministic gold-label extraction only, not for jailbreak refusal evaluation.
- Judge/evaluation paths that use OpenAI should use batch mode unless the script is explicitly a small synchronous canary.
- Scripts that generate claim-bearing outputs must follow `docs/quantitative-reporting-standards.md` and pass `uv run python scripts/audit_ci_coverage.py`.
- Resumability skips for intervention outputs must validate both alpha JSONL
  completeness and `intervention_run_config.json` compatibility.

## Output Safety

- Use `start_run_provenance()` or the existing pipeline wrappers for scripts that produce claim-relevant run outputs.
- Never delete, bypass, or overwrite `*.provenance.json`.
- Before staging or restructuring output paths, run `uv run python -m scripts.lib.pipeline active-run-status`.
- For run-directory layout and provenance policy, read `data/AGENTS.md`.
