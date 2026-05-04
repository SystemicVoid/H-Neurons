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
${PIPELINE} check-intervention-contract --output-dir "${OUTPUT_DIR}" --classifier-path "${CLASSIFIER_PATH}" --alphas "${ALPHAS[@]}"
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

PROJECT_DIR="${PROJECT_DIR:-/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons}"
# shellcheck source=scripts/lib/inhibit_suspend.sh
source "${PROJECT_DIR}/scripts/lib/inhibit_suspend.sh"
inhibit_suspend "<pipeline name>" "$@"

cd "${PROJECT_DIR}"

PIPELINE="uv run python -m scripts.lib.pipeline"
LOG="logs/<name>_$(date +%Y%m%d_%H%M%S).log"

${PIPELINE} gpu-preflight 2>&1 | tee -a "${LOG}" || true
```

`inhibit_suspend` (in `scripts/lib/inhibit_suspend.sh`) re-execs the wrapper
under `systemd-inhibit` with the full
`--what=sleep:idle:shutdown:handle-suspend-key:handle-power-key:handle-lid-switch`
class set, then disables COSMIC DE auto-suspend for the run by writing a
far-future sentinel into cosmic-idle's config (with restore on EXIT/INT/TERM).
Incident 2026-04-30: a 9h local GPU run was suspended despite a block-mode
`sleep:idle` inhibit because COSMIC's `cosmic-idle` daemon shells out to
`systemctl suspend` on its own timer through a path the narrow class did not
cover. Honours `DRY_RUN=1` (no-op).

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

## Remote Runtime and Volume Rules

Remote wrappers must make the runtime boring. Do not hand-assemble long
`uv`/tmux commands in a paid SSH session when the wrapper can encode the choice.

- Always set `PROJECT_DIR` explicitly on remote runs, usually
  `/workspace/02-h-neurons`. Local workstation defaults are not valid remote
  launch configuration.
- Wrapper code should build one runtime command array and reuse it for guards and
  workload commands. If the pod uses the baked image, prefer
  `UV_PROJECT_ENVIRONMENT=/opt/h-neurons/.venv` with `uv run --no-sync`. If the
  pod must recover from `requirements.txt`, use an explicit requirements mode:
  `uv run --no-project --with-requirements requirements.txt --python 3.11 --index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match`.
  Do not silently mix runtime modes inside one launch.
- A requirements-backed runtime is an acceptable emergency path only after it
  imports the exact stack with the same command family that will launch the run:
  Python version, `torch.__version__`, `transformers.__version__`, CUDA
  availability, GPU name, BF16 support, and memory guard. Log that proof before
  model work.
- Do not run `uv sync --frozen` in this repo on a pod unless a committed
  `uv.lock` exists and the command also pins the intended Python/CUDA index.
  Without a lockfile, `uv sync --frozen` is not a repair operation; it can select
  a newer Python and rewrite `.venv`.
- Never trust a retained network-volume `.venv` under
  `/workspace/02-h-neurons/.venv` just because it is large. Treat it as invalid
  until its Python target exists and the exact workload imports `torch`,
  `transformers`, and CUDA successfully. Prefer container-local or baked
  environments for paid runs.
- For RunPod network-volume syncs, use `rsync` with
  `--no-owner --no-group --no-perms`; plain archive mode may fail on `chown`.
  For full mirrors, use `--delete --backup --backup-dir=<outside-repo> --partial`
  and exclude machine-local state such as `.venv`, `.env`, caches, logs, and
  wandb.
- Do not launch from a repo tree while `rsync --delete` is still replacing that
  same tree. The only acceptable parallel launch is after critical code,
  manifest, classifier, and dependency hashes match, and outputs go outside the
  repo being deleted/replaced.
- Before detached tmux launch, run the wrapper in remote dry-run mode with the
  same `PROJECT_DIR`, runtime mode, output root, and stage selection. Missing
  sourced helpers such as `scripts/lib/inhibit_suspend.sh` should fail here, not
  after tmux detaches.
- For complex remote tmux scripts, prefer a checked-in wrapper, heredoc, or
  base64 payload. Nested inline shell quoting with arrays is not acceptable for
  claim-bearing launches.

**RunPod launch (template-based):** RunPod profiles use the private template
`h-neurons-runpod-private` (`v88hqzvuxk`), backed by
`ghcr.io/systemicvoid/h-neurons-runpod` with the project venv pre-baked at
`/opt/h-neurons/.venv`. Render the default launch with
`uv run python scripts/infra/cloudctl.py render-launch --profile mistral24b-runpod --stage cp1 --attempt 1`.
`render-launch` verifies the template image against the profile
`expected_image_digest` before printing the paid command; direct-image fallback
requires `--allow-image-fallback --confirmed-image-pin`, and unpinned RunPod
profiles must be explicitly marked non-production with
`allow_unpinned_image = true`. Wrappers use `uv run --no-sync` and
`UV_PROJECT_ENVIRONMENT`; no `uv sync` or tool installation happens on the pod.
Rebuild and push the image when
`requirements.txt`, `pyproject.toml`, or `scripts/infra/docker/*` changes, then
update the profile digest pins; see `scripts/infra/cloud/runpod_bake_image.sh`.

**RunPod burst shards:** For urgent one-off Gemma/SIMID tail shards, the private
template is not the default after the 2026-05-01 cold-pull incident. Use the
Burst Shard Launch Pattern in `scripts/infra/cloud/runbooks/runpod.md`: official
RunPod PyTorch template, timed SSH canary, system-site venv that reuses the
image's CUDA Torch, HF-staged repo/model/wheelhouse, and no project `uv sync`.
