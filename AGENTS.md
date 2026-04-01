# Repository Guidelines

**Prioritize steps by information gained per unit time. After every result, ask what was learned, what branch of the search tree was ruled out, and whether the failure was conceptual, methodological, or merely implementation-level.**

- Use the literature when building experiments based on related work, start with `papers/INDEX.md`.

## Build, Test, and Development Commands

- `uv add <package>` adds a dependency; regenerate `requirements.txt` with `uv export --no-hashes --frozen --no-emit-project > requirements.txt` in the same change.
- `ruff check scripts` and `ruff format scripts` lint and format Python before review.
- `ty check` type-checks `scripts/` (configured in `[tool.ty]` in pyproject.toml; resolves third-party imports from `../.venv`).
- `ruff`, `ty`, and `prek` are global tools on PATH (installed via `uv tool`).
Tests live in `tests/` and run via `uv run pytest`. Core evaluation helpers (`normalize_answer`, `extract_mc_answer`) have unit tests in `tests/test_utils.py`.
- Follow existing conventional commits pattern.

## Coding Style & Naming Conventions

Follow PEP 8 naming conventions, enforced by ruff's `pep8-naming` (N) rules:
- **Functions and variables**: `snake_case` (e.g., `load_data`, `train_ids`)
- **Classes**: `PascalCase` (e.g., `HNeuronScaler`, `TokenExtractor`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `TARGET_IDX`, `ALPHAS`)
- **ML convention exception**: uppercase `X`, `X_train`, `X_test`, `C`, `C_values` are allowed for scikit-learn feature matrices and regularization parameters (configured in `[tool.ruff.lint.pep8-naming]` ignore-names).

## Quantitative Reporting Standards

Every quantitative claim in presentation materials must include uncertainty estimates (bootstrap 95% CIs or binomial proportion CIs). Treat `docs/ci_manifest.json` as the source-of-truth registry. Before finishing any change that touches quantitative reporting surfaces, run `uv run python scripts/audit_ci_coverage.py`.

## Site Deployment

To redeploy the project site at its canonical URL:

```bash
scripts/infra/publish.sh site --slug aware-fresco-4a2q --client amp
```

## Run Directory Conventions

Keep the existing semantic layout `data/<model>/intervention/<benchmark>/experiment/`. The provenance sidecars already carry the "when" and "how."

When a re-run would overwrite an existing `experiment/` directory that contains committed or analysed data, archive it first:

```
data/<model>/intervention/<benchmark>/experiment_YYYY-MM-DD_<reason>/
```

For genuinely new experiments (new benchmark, new method), create a new semantic directory rather than just timestamped one. Prefer names that describe what varies, not just when it ran.

## GPU Job Discipline

<important if="running GPU jobs or pipeline scripts">
Never start a GPU job without pre-staging its downstream chain in a tmux window.

Chaining rules:
- Write the chain as a robust bash script, then run that script in one tmux window, open for user to monitor. Integrate HITL review gates, with easy continue ux post-review.
- Use `set -euo pipefail` instead of `&&` — it halts on any non-zero exit and catches pipe failures.
- Use `systemd-inhibit` for any GPU job longer than ~20 minutes.

After the chain completes, append an entry to `notes/runs_to_analyse.md` :

```markdown
## <ISO timestamp> | <run_dir relative path>
What: <one-line: benchmark + method + alpha grid>
Key files: results.json, *.provenance.json, activations/responses.jsonl
Status: awaiting analysis
```

After that entry is analysed, remove it.

Rules:
- Every script already writes a `*.provenance.json` sidecar via `start_run_provenance` / `finish_run_provenance` — never bypass or delete these. They are the machine-readable chain of custody.
- Check `nvitop -1` before launching to confirm GPU is free.
</important>
