# Repository Guidelines

If anything about this repository is surprising, brittle, or easy to get wrong, add a short, specific note to this file. Keep it concise — archive resolved or narrow-scope notes to `docs/` instead.

## Project Structure & Module Organization
Core pipeline code lives in `scripts/` (flat — sibling imports like `from intervene_model import …` require it). Cloud/remote orchestration scripts live in `scripts/infra/`. Put new research utilities in `scripts/` unless they justify a reusable package.

`data/` is organized by model: `data/gemma3_4b/` (primary, local runs) and `data/mistral24b/` (secondary, cloud GPU run). Shared datasets live at `data/benchmarks/` and `data/TriviaQA/`. Original paper reference outputs are in `data/original_paper_examples/`. Commit compact experiment artifacts (JSONL, JSON) to git; keep heavy activations and investigation dumps gitignored (`data/**/activations/`, `data/**/investigation_*/`).

Within `data/gemma3_4b/`, artifacts are grouped by pipeline stage:
- `pipeline/` — TriviaQA probe-training outputs (consistency_samples, answer_tokens, train/test qids, classifier summaries, pipeline_report)
- `probing/bioasq13b_factoid/` — BioASQ OOD probe transfer (samples, answer_tokens, classifier_summary, logs/)
- `intervention/<benchmark>/experiment/` — H-neuron intervention results (alpha_*.jsonl, results.json)
- `intervention/<benchmark>/control/` — negative control results (seed_* dirs, comparison_summary)
- `swing_characterization/` — swing-sample analysis

`site/index.html` is a hand-maintained presentation deck: narrative copy, chart arrays, and intervention numbers are hardcoded. When results change, update both the prose and the embedded JS arrays together or the site drifts.

`data/gemma3_4b/intervention/faitheval_standard/experiment/results.json` only stores raw compliance totals. If you need parse-failure counts, parseable-subset rates, or format-sensitive site exports, derive them from the committed `alpha_*.jsonl` rows instead of assuming `results.json` is sufficient.

## Key Workflow Notes

- `scripts/utils.py` contains shared lightweight helpers (`normalize_answer`, `extract_mc_answer`). Import from there — never duplicate these functions into individual scripts.
- `scripts/collect_responses.py` imports `torch`/`transformers`/`openai` at module level — don't import it from lightweight scripts for utility functions.
- `scripts/evaluate_intervention.py` loads the OpenAI key via `python-dotenv` from the repo-root `.env`.
- For zero-cost runs without an OpenAI key: `--strategy synthetic-output` on `extract_answer_tokens.py` paired with `--locations output` on `extract_activations.py`.
- `extract_activations.py` needs the same `apply_chat_template()` tensor-vs-`BatchEncoding` guard as `collect_responses.py`.
- For BioASQ OOD probing, use the official BioASQ Task B JSON (question `body` + `type` + `exact_answer`) rather than HF mirrors like `kroshan/BioASQ`, which flatten answer/context into CSV text and do not match the original task schema.
- `data/gemma3_4b/pipeline/test_qids_disjoint.json` contains 782 sampled IDs, but the current disjoint classifier evaluation covers 780 because two IDs are missing activation files. Use the CI-bearing summary JSON as the reporting source of truth.
- `scripts/infra/lambda-bootstrap.sh` now supports Tailscale-first access. Pass `TAILSCALE_AUTH_KEY` to auto-enroll/tag the instance; SSH is only locked to the Tailscale address after enrollment succeeds, so a missing/bad key will not cut off the public bootstrap session.

## SAE Investigation (Gemma Scope 2)
- Full plan and status tracker: `docs/sae_investigation_plan.md`
- SAE scripts: `extract_sae_activations.py`, `classifier_sae.py`, `intervene_sae.py`, `spike_sae_feasibility.py`
- **Critical hook point:** Gemma Scope 2 MLP SAEs are trained on `post_feedforward_layernorm` output, NOT raw `down_proj` output. Gemma 3 has a post-feedforward RMSNorm between MLP output and residual addition. SAE extraction/intervention must hook at `post_feedforward_layernorm`, not `down_proj`.
- SAE release: `gemma-scope-2-4b-it-mlp-all`, SAE IDs: `layer_{N}_width_{16k|262k}_l0_{small|big}`
- SAE dimensions: d_in=2560 (hidden_size), d_sae=16384 (16k width)
- SAE feature data goes to `data/gemma3_4b/pipeline/activations_sae/` (gitignored)

## Known Issues
- **FaithEval standard-prompt mis-scoring**: The MC letter extractor in `evaluate_intervention.py` fails on ~150 items at `alpha=3.0`; a strict answer-text remap recovers most. Treat raw standard-prompt compliance drops as evaluator artifacts until text-based rescoring is wired in.
- **`extract_answer_tokens.py` JSON brittleness**: The judge model occasionally returns malformed JSON on short quoted answers (observed on `tc_115` in Mistral canary). Needs retry/fallback hardening.

## Paper-Faithful Replication Notes
- `scripts/classifier.py` sweeps `C` on held-out probe metrics but does **not** implement the paper's full selection rule (which also scores TriviaQA behavior after suppression). Treat as a detector-selection baseline, not the final paper-equivalent criterion.
- `scripts/run_intervention.py` defaults to `--prompt_style anti_compliance`; use `--prompt_style standard` for paper-faithful replication (matches official Salesforce/FaithEval framing).
- Jailbreak eval uses JailbreakBench (Chao et al., NeurIPS 2024) by default (`--jailbreak_source jailbreakbench`). Use `--jailbreak_source forbidden` + `--jailbreak_path` for the legacy 390-question forbidden set. `--benchmark jailbreak_benign` runs the JBB benign split for over-refusal testing.

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

### Quantitative Claim Workflow
- Treat `docs/ci_manifest.json` as the source-of-truth registry for paper-facing claims, their evidence artifacts, and their allowed statuses (`required`, `blocked_data`, `not_applicable`).
- Prefer structured rendering over manual copy. Site pages should read metrics from `site/data/*.json`; markdown tables should be generated from structured artifacts where practical instead of retyped by hand.
- If prose must still include a literal quantitative claim, add an adjacent provenance comment of the form `<!-- from: claim_id -->` using a claim id from `docs/ci_manifest.json`.
- Before finishing any change that touches quantitative reporting surfaces, run `uv run python scripts/audit_ci_coverage.py`. The `prek` config also enforces this at commit time.
- When promoting a new result into site/report/grant surfaces, update `docs/ci_manifest.json` first, then wire the surface to the artifact or add the provenance comment in the same change.

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
