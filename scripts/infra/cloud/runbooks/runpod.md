# RunPod Runbook

RunPod is the first supported provider because it blocks the Mistral 24B CP2 path.

## Before Paid Launch

```bash
git status --short --branch
uv run python scripts/infra/cloudctl.py preflight --profile mistral24b-runpod
uv run python scripts/infra/cloudctl.py render-launch --profile mistral24b-runpod
```

For a second CP1 attempt or any CP2+ work, create/prewarm a Secure Cloud network volume first and render with `--network-volume-id`.

## After Pod Creation

```bash
uv run python scripts/infra/cloudctl.py ssh-info --pod-id "$POD_ID"
ssh -o StrictHostKeyChecking=accept-new -p "$SSH_PORT" root@"$SSH_HOST" true
```

On the pod, assert H100/BF16/>=75 GiB before running `scripts/infra/mistral24b_replication.sh`. Use `uv sync --no-dev`; use `DEVICE_MAP=cuda:0` unless the CP1 validator decision deliberately changes.

## After Run

Sync only enumerated artifacts: smoke JSON, provenance sidecar, environment capture, and wrapper log. Before staging outputs, run:

```bash
uv run python -m scripts.lib.pipeline active-run-status
```

After deleting compute, verify cleanup:

```bash
uv run python scripts/infra/cloudctl.py cleanup-check
```
