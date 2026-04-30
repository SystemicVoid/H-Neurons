# Infra and Long-Run Guide

Use this file when designing or launching long GPU jobs, pipeline wrappers, tmux/systemd-inhibit orchestration, remote-run scripts, or expensive evaluation jobs.

## GPU Run Constitution

- Never run long GPU jobs ad hoc.
- Always launch via a dedicated bash script in a tmux window.
- Use `set -euo pipefail`.
- Use `systemd-inhibit` for runs longer than about 20 minutes.
- Check `nvitop -1` before launch.

Long runs must be:

- idempotent
- incrementally persisted
- resumable
- failure-visible

Forbidden:

- keeping hours of results only in memory
- collect-everything-write-at-end designs
- manual multi-step shell driving for critical runs
- bypassing provenance sidecars

Required patterns:

- write outputs incrementally (`jsonl`, shards, per-batch/per-split artifacts)
- flush or close files regularly
- for long JSONL append loops, use an existing safe writer or reopen/flush per record so crashes or external unlinks cannot silently lose buffered output
- checkpoint expensive stages
- on restart, skip completed work via existence and integrity guards
- fail fast with clear logs and non-zero exits

If killing the process loses substantial work, the pipeline is misdesigned.

## Pipeline Guards

Use `scripts.lib.pipeline` instead of reimplementing guard logic in bash:

```bash
PIPELINE="uv run python -m scripts.lib.pipeline"
${PIPELINE} gpu-preflight
${PIPELINE} gpu-hardware-guard --min-memory-gib 75 --name-pattern 'H100|A100'
${PIPELINE} active-run-status
${PIPELINE} check-stage --output-dir "${OUTPUT_DIR}" --manifest "${MANIFEST}" --alphas "${ALPHAS[@]}"
${PIPELINE} check-live-output-track-state --output-target "${OUTPUT_DIR}"
${PIPELINE} log-run --run-dir "${OUTPUT_DIR}" --description "<benchmark + method + alphas>"
```

Before any staging or restructuring near output paths, run:

```bash
uv run python -m scripts.lib.pipeline active-run-status
```

For guard-library implementation rules, read `scripts/lib/AGENTS.md`.

## Pipeline Wrapper Shape

Bash wrappers should stay thin: process lifecycle, tmux/systemd-inhibit friendliness, signal behavior, logging, and calls into Python helpers.

Minimal wrapper pattern:

```bash
#!/usr/bin/env bash
set -euo pipefail

if [ -z "${INHIBIT_WRAPPED:-}" ] && command -v systemd-inhibit &>/dev/null; then
    exec env INHIBIT_WRAPPED=1 systemd-inhibit \
        --what=sleep:idle --why="<pipeline name>" \
        -- bash "$0" "$@"
fi

cd "${PROJECT_DIR:-/workspace/02-h-neurons}"

PIPELINE="uv run python -m scripts.lib.pipeline"
LOG="logs/<name>_$(date +%Y%m%d_%H%M%S).log"

${PIPELINE} gpu-preflight 2>&1 | tee -a "${LOG}" || true
```

## Evaluation Jobs

- OpenAI evaluation should use `--api-mode batch` except for explicit small canaries.
- Batch mode is crash-safe through state files and cheaper than fast mode.
- Pipeline scripts should run `scripts/infra/check_openai_batch_limits_via_codex.sh` before submitting large batches.

## Jailbreak Pipelines

- New claimable jailbreak generation should use `--run_profile canonical` or canonical-equivalent explicit decode values.
- Canonical decode is `do_sample=True`, `temperature=0.7`, `top_k=20`, `top_p=0.8`, `max_new_tokens=5000`.
- Noncanonical decode settings are exploratory and must not be mixed into claim-bearing comparisons.

## Completion

After successful completion, append claim-relevant runs to `notes/runs_to_analyse.md` using the format in `data/AGENTS.md`.

Remote Lambda/GH200 specifics live in `scripts/infra/cloud/runbooks/lambda.md`.

## Cloud Adapter

For RunPod/Lambda cloud orchestration, use `scripts/infra/cloudctl.py` and the
runbooks under `scripts/infra/cloud/`. The `vendor/zombuul` submodule is
reference-only; do not install or auto-load its plugin, skills, or prompts.
