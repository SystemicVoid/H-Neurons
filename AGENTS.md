# Repository Guidelines

If anything about this repository is surprising, brittle, or easy to get wrong, add a short, specific note to this file. Keep it concise — archive resolved or narrow-scope notes to `docs/` instead.

## Project Structure & Module Organization
Core pipeline code lives in `scripts/` (flat — sibling imports like `from intervene_model import …` require it). Cloud/remote orchestration scripts live in `scripts/infra/`. Put new research utilities in `scripts/` unless they justify a reusable package.

`data/` is organized by model: `data/gemma3_4b/` (primary, local runs) and `data/mistral24b/` (secondary, cloud GPU run). Shared datasets live at `data/benchmarks/` and `data/TriviaQA/`. Original paper reference outputs are in `data/original_paper_examples/`. Commit compact experiment artifacts (JSONL, JSON) to git; keep heavy activations and investigation dumps gitignored (`data/*/activations/`, `data/*/investigation_*/`).

`site/index.html` is a hand-maintained presentation deck: narrative copy, chart arrays, and intervention numbers are hardcoded. When results change, update both the prose and the embedded JS arrays together or the site drifts.

`data/gemma3_4b/intervention/faitheval_standard/results.json` only stores raw compliance totals. If you need parse-failure counts, parseable-subset rates, or format-sensitive site exports, derive them from the committed `alpha_*.jsonl` rows instead of assuming `results.json` is sufficient.

## Key Workflow Notes

- `scripts/utils.py` contains shared lightweight helpers (`normalize_answer`, `extract_mc_answer`). Import from there — never duplicate these functions into individual scripts.
- `scripts/collect_responses.py` imports `torch`/`transformers`/`openai` at module level — don't import it from lightweight scripts for utility functions.
- `scripts/evaluate_intervention.py` loads the OpenAI key via `python-dotenv` from the repo-root `.env`.
- For zero-cost runs without an OpenAI key: `--strategy synthetic-output` on `extract_answer_tokens.py` paired with `--locations output` on `extract_activations.py`.
- `extract_activations.py` needs the same `apply_chat_template()` tensor-vs-`BatchEncoding` guard as `collect_responses.py`.
- For BioASQ OOD probing, use the official BioASQ Task B JSON (question `body` + `type` + `exact_answer`) rather than HF mirrors like `kroshan/BioASQ`, which flatten answer/context into CSV text and do not match the original task schema.

## Known Issues
- **FaithEval standard-prompt mis-scoring**: The MC letter extractor in `evaluate_intervention.py` fails on ~150 items at `alpha=3.0`; a strict answer-text remap recovers most. Treat raw standard-prompt compliance drops as evaluator artifacts until text-based rescoring is wired in.
- **`extract_answer_tokens.py` JSON brittleness**: The judge model occasionally returns malformed JSON on short quoted answers (observed on `tc_115` in Mistral canary). Needs retry/fallback hardening.

## Paper-Faithful Replication Notes
- `scripts/classifier.py` sweeps `C` on held-out probe metrics but does **not** implement the paper's full selection rule (which also scores TriviaQA behavior after suppression). Treat as a detector-selection baseline, not the final paper-equivalent criterion.
- `scripts/run_intervention.py` defaults to `--prompt_style anti_compliance`; use `--prompt_style standard` for paper-faithful replication (matches official Salesforce/FaithEval framing).

## Build, Test, and Development Commands
Use `uv` for Python environment management.

- `uv sync` installs the project from `pyproject.toml`.
- `uv add <package>` adds a dependency; regenerate `requirements.txt` with `uv export --no-hashes --frozen --no-emit-project > requirements.txt` in the same change.
- `ruff check scripts` and `ruff format scripts` lint and format Python before review.
- `ty check` type-checks `scripts/` (configured in `[tool.ty]` in pyproject.toml; resolves third-party imports from `../.venv`).
- `prek run` runs all pre-commit hooks (ruff check, ruff format, ty) on staged files. `prek install` wires hooks into `.git/hooks/`.
- `ruff`, `ty`, and `prek` are global tools on PATH (installed via `uv tool`). No venv activation needed to run them.

## Quantitative Reporting Standards
Every quantitative claim in presentation materials must include uncertainty estimates. Use bootstrap 95% CIs where sample sizes allow (n > 30). For classifier metrics, report ± from stratified bootstrap over test samples. For intervention compliance rates, report ± from binomial proportion CIs. If uncertainty cannot yet be computed, flag the number explicitly as "no CI".

## Long-Running Jobs & System Suspend
On Pop!_OS / COSMIC DE, system auto-suspend will kill tmux jobs mid-run. Always hold a `systemd-inhibit` lock for jobs longer than ~20 minutes.

**At launch:**
```bash
systemd-inhibit --what=sleep:idle --who="<job-name>" --why="<description>" \
    uv run python scripts/run_intervention.py ...
```

**For already-running jobs** — watcher in a tmux window named `inhibit`:
```bash
systemd-inhibit --what=sleep:idle --who="<job-name>" --why="<description>" \
    bash -c 'while tmux list-windows | grep -q "<job-window-name>"; do sleep 30; done'
```

**Post-suspend recovery:** if a job is running slowly after wake, `kill -9` the python PID (`pgrep -a python | grep <script>`). The script's resume logic picks up from the last written line.

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
