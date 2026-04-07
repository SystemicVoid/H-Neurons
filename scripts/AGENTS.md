# Scripts — Pipeline & Evaluation Guide

## Pipeline guard library (`scripts/lib/pipeline.py`)

Shared, tested guards for GPU pipeline orchestration. Every function traces to a specific incident that burned GPU hours. **Use these instead of reimplementing guards in bash.**

### Usage from bash pipelines

```bash
PIPELINE="uv run python -m scripts.lib.pipeline"

# Check if all alpha files are complete (file existence AND line count)
if ${PIPELINE} check-stage \
    --output-dir "${EXPERIMENT_DIR}" \
    --manifest "${MANIFEST}" \
    --alphas 0.0 1.0 2.0 4.0 8.0; then
    echo "Stage complete; skipping"
else
    # run generation...
fi

# Print manifest sizes (multiple at once)
${PIPELINE} manifest-count "${PILOT_IDS}" "${FULL_IDS}"

# GPU health snapshot (non-fatal if nvitop missing)
${PIPELINE} gpu-preflight

# Log a completed run to the analysis queue
${PIPELINE} log-run \
    --run-dir "data/gemma3_4b/intervention/jailbreak_d7/pilot100_canonical/causal/experiment" \
    --description "D7 pilot causal, alphas 0-8"
```

### Why this exists

| Function | Incident it prevents |
|---|---|
| `check_stage_complete` | 2026-04-07: bash guard only checked file existence; power-off left alpha_8.0 at 68/100 lines, guard skipped re-generation |
| `manifest_count` | Hardcoded sample counts that drift from actual manifests |
| `gpu_preflight` | Launching on a busy GPU |
| `log_run` | Forgotten `runs_to_analyse.md` entries causing skipped or duplicate analysis |

Tests: `tests/test_pipeline.py` (22 cases). Run `uv run pytest tests/test_pipeline.py`.

### Adding new guards

Only add functions that trace to a real incident or are duplicated across 5+ pipeline scripts. Each new function needs a docstring with the incident date and a corresponding test.

---

## Conventions still enforced by documentation (not yet structural)

### Jailbreak generation: never override decode defaults

`run_jailbreak()` defaults to `do_sample=True, temperature=0.7, max_new_tokens=256`.
Always pass `--max_new_tokens 5000` from pipeline scripts (256 truncates disclaimers and creates alpha-dependent false positives — see week3-log). Never pass `--jailbreak_do_sample`, `--jailbreak_temperature`, `--jailbreak_top_k`, or `--jailbreak_top_p` — the function defaults are the canonical jailbreak evaluation surface. Greedy decode (`do_sample=False`) is only valid for deterministic gold-label extraction, never for jailbreak refusal evaluation.

### JSONL writes: reopen file per record

Open the output file **inside** the sample loop, not outside it. Holding one fd across the loop silently loses all writes if the path is unlinked mid-run (e.g. by a concurrent `git commit`). Incident: 2025-03-23, 500 samples destroyed.

### Never touch output directories during a GPU run

Before any `git add/rm`, `mv`, `rm`, or restructuring, check for active runs:
```bash
ps aux | grep run_intervention   # or check tmux panes
```
A running job has `"status": "running"` in its `*.provenance.json`. Wait for it to finish.

### OpenAI evaluation: batch mode only, never fast

Always use `--api-mode batch`. Fast mode is all-or-nothing in memory — a crash or quota hit loses everything. Batch is crash-safe (`.eval_batch_state.json`) and 50% cheaper. Both `evaluate_intervention.py` and `evaluate_csv2.py` default to batch — omitting the flag is fine. Pipeline scripts run `scripts/infra/check_openai_batch_limits_via_codex.sh` before submitting. Incident: 2026-03-27, 190 judge calls lost.

---

## Writing new pipeline scripts

Bash stays thin — process lifecycle only (systemd-inhibit, tmux, signals, `tee`). Decision logic (guards, state, validation) goes through `scripts.lib.pipeline`.

Minimal template:
```bash
#!/usr/bin/env bash
set -euo pipefail

# Re-launch under systemd-inhibit for runs > 20min
if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle --why="<pipeline name>" \
        -- bash "$0" "$@"
fi

cd /home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons

PIPELINE="uv run python -m scripts.lib.pipeline"
MANIFEST="data/manifests/<manifest>.json"
OUTPUT_DIR="data/<model>/intervention/<benchmark>/experiment"
LOG="logs/<name>_$(date +%Y%m%d_%H%M%S).log"

${PIPELINE} gpu-preflight 2>&1 | tee -a "${LOG}" || true

if ${PIPELINE} check-stage --output-dir "${OUTPUT_DIR}" --manifest "${MANIFEST}" --alphas <alphas>; then
    echo "Generation complete; skipping" | tee -a "${LOG}"
else
    uv run python scripts/run_intervention.py \
        --benchmark <benchmark> \
        --output_dir "${OUTPUT_DIR}" \
        --sample_manifest "${MANIFEST}" \
        --alphas <alphas> \
        --max_new_tokens 5000 \
        2>&1 | tee -a "${LOG}"
fi

# Judge (always batch mode)
uv run python scripts/evaluate_intervention.py \
    --benchmark <benchmark> \
    --input_dir "${OUTPUT_DIR}" \
    --alphas <alphas> \
    --api-mode batch 2>&1 | tee -a "${LOG}"

# Log to analysis queue
${PIPELINE} log-run --run-dir "${OUTPUT_DIR}" --description "<benchmark + method + alphas>"
```
