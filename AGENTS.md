# Repository Guidelines

**Great research taste is mostly the ability to identify the bottleneck assumption and attack it with the cheapest decisive test. Always do the components in order from most informative per unit time to least informative per unit time. Before proposing a new experiment, name the specific claim it would upgrade or falsify. If you cannot, it is probably garnish.** 


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

## Long-Running Jobs

Use `systemd-inhibit` for any GPU job longer than ~20 minutes. Recipes and recovery steps in `docs/tooling-assessment.md`.
- If a script supports `--wandb`, prefer enabling it for long, comparison-heavy, or audit-heavy runs: treat W&B as the visual cockpit for curves, tables, and triage, while local `summary.json` and `*.provenance.json` remain the source of truth.

## GPU Monitoring

Use `nvitop -1` for a one-shot GPU status check before/during long jobs. Details in `docs/tooling-assessment.md`.

## Codebase Prompt Export

`./code2prompt.sh` exports all source code as a single LLM-ready prompt. Pass `--output-file=prompt.txt` or `-c` for clipboard.

## Run Directory Conventions

Keep the existing semantic layout `data/<model>/intervention/<benchmark>/experiment/`. Do not restructure into timestamped run directories — the provenance sidecars already carry the "when" and "how."

When a re-run would overwrite an existing `experiment/` directory that contains committed or analysed data, archive it first:

```
data/<model>/intervention/<benchmark>/experiment_YYYY-MM-DD_<reason>/
```

For genuinely new experiments (new benchmark, new method), create a new semantic directory rather than a timestamped one. Prefer names that describe what varies, not when it ran.

## GPU Job Discipline

<important if="running GPU jobs or pipeline scripts">
Never start a GPU job without pre-staging its downstream chain in a tmux window.
The chain for intervention experiments is:

```bash
# Chain: inference → evaluate → export → log to backlog
set -euo pipefail
PYTHONUNBUFFERED=1 uv run python scripts/run_intervention.py \
  <args> 2>&1 | tee logs/<benchmark>_intervention.log
PYTHONUNBUFFERED=1 uv run python scripts/evaluate_intervention.py \
  --input_dir <run_dir> 2>&1 | tee logs/<benchmark>_evaluate.log
uv run python scripts/export_site_data.py 2>&1 | tee logs/export.log
```

Chaining rules:
- Write the full chain as a single bash script (or `bash -c` heredoc), then run that script in one tmux pane. Never use `tmux send-keys` to inject commands one at a time — it races or stalls.
- Use `set -euo pipefail` instead of `&&` — it halts on any non-zero exit and catches pipe failures.
- The pattern in `scripts/infra/gh200_sequential_pipeline.sh` is the reference implementation.

After the chain completes, append an entry to `notes/runs_to_analyse.md` (create file and directory if they do not exist):

```markdown
## <ISO timestamp> | <run_dir relative path>
What: <one-line: benchmark + method + alpha grid>
Key files: results.json, *.provenance.json, activations/responses.jsonl
Status: awaiting analysis
```

Change status to `done: <brief finding>` after analysis.

Rules:
- Every script already writes a `*.provenance.json` sidecar via `start_run_provenance` / `finish_run_provenance` — never bypass or delete these. They are the machine-readable chain of custody.
- Local `summary.json` and `*.provenance.json` are the source of truth. W&B is the visual cockpit, not the record.
- Tell the user know before queuing a new GPU job if there are more than 3 entries in `notes/runs_to_analyse.md` still `awaiting analysis`, especially when these analysis outcomes could undermine the value of the experiment you are about to run, be sensible.
- Check `nvitop -1` before launching to confirm GPU is free.
</important>
