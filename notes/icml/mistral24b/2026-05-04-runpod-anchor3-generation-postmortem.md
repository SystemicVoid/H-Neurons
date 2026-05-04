# RunPod Anchor 3 Generation Postmortem

- **Date:** 2026-05-04
- **Scope:** Mistral 24B anchor 3 JBB harmful500 generation launch on RunPod H100, direct SSH `root@38.80.152.148 -p 30111`.
- **Canonical status:** keep scientific progress state in [2026-04-28-5.5-pro-l1-mitigation-strategy.md](2026-04-28-5.5-pro-l1-mitigation-strategy.md). This note is operational only.
- **Outcome as of 2026-05-04T16:45:54Z:** generation was running in tmux session `mistral24b-anchor3-jbb500-h100`; repo head on the pod was clean at `913713c`; H100 memory was about 46.2 GiB; `/workspace/mistral24b_anchor3_full500/experiment/alpha_0.0.jsonl` had 10 rows. No API judging was launched.

## Trace and Log Locations

- Full Codex trace: `/home/hugo/.codex/sessions/2026/05/04/rollout-2026-05-04T13-36-25-019df2fd-6329-7370-8498-356a9e0f3872.jsonl`.
- User message index: `/home/hugo/.codex/history.jsonl`, session `019df2fd-6329-7370-8498-356a9e0f3872`, messages around the generation request and sync interruption.
- Local wrapper dry-run logs: `logs/mistral24b_anchor3_jailbreak_first_pass_20260504T123902Z.log`, `logs/mistral24b_anchor3_jailbreak_first_pass_20260504T123917Z.log`, `logs/mistral24b_anchor3_jailbreak_first_pass_20260504T155115Z.log`, `logs/mistral24b_anchor3_jailbreak_first_pass_20260504T160803Z.log`.
- Remote successful launch log: `/workspace/02-h-neurons/logs/mistral24b_anchor3_jbb500_generation_20260504T164219Z.log`.
- Remote output root: `/workspace/mistral24b_anchor3_full500`.
- Remote pre-sync backup directory: `/workspace/repo-sync-backups/02-h-neurons-20260504T162017Z`.

## Timeline

| Time UTC | Event |
|---|---|
| 16:16 | User approved only the GPU generation run. H100 was idle, `tmux` and `uv` existed, repo was `/workspace/02-h-neurons`. |
| 16:17 | Remote repo was stale: branch `main`, head `3a091fb3`, dirty remote-only data/notes, missing scaffold/test files. A narrow `rsync -av --relative` failed with FUSE/network-volume `chown` permission errors. |
| 16:18-16:20 | User requested full network-volume sync. Local clean head was `913713c`; remote head was `3a091fb3`. Dry-run estimated 20.3 GiB transfer and 15 stale remote deletes. Real mirror used `--delete --backup`, excluded `.venv`, `.env`, caches, logs, and wandb, and used `--no-owner --no-group --no-perms`. |
| 16:23-16:27 | Launch was held while `rsync --delete` was active because the default output root was inside the repo. A safer parallel path was identified: verify critical hashes and write outputs outside `/workspace/02-h-neurons`. |
| 16:30-16:34 | First tmux attempts exited before logging. Root causes were missing `PROJECT_DIR=/workspace/02-h-neurons` and missing `scripts/lib/inhibit_suspend.sh` while sync was still incomplete. The helper was synced directly and remote dry-run then passed. |
| 16:34-16:35 | Approved wrapper launch stayed alive but stalled before GPU inside `uv run --no-sync python -m scripts.lib.pipeline active-run-status`. The preserved `.venv` pointed to `/usr/bin/python3`, which did not exist in the container. |
| 16:36 | Attempted to repoint `.venv/bin/python` to uv-managed CPython 3.11. That exposed that the large `.venv` still lacked `torch`, `transformers`, and `pip`; it was not a trustworthy runtime. |
| 16:37 | `uv sync --frozen --inexact --no-install-project` was the wrong repair path. The repo had no `uv.lock`; uv selected CPython 3.14 from `requires-python >=3.11`, removed the old `.venv`, and created a tiny 3.14 env before failing. |
| 16:38-16:39 | A clean `/tmp/02h-venv` `uv sync --frozen --python 3.11` also failed because there is no `uv.lock`. The setup path switched to `uv run --no-project --with-requirements requirements.txt --python 3.11`. |
| 16:39-16:40 | Requirements resolution failed on default PyPI because `requirements.txt` pins `torch==2.9.1+cu130`. Adding `--index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match` fixed the runtime: Python 3.11.15, `torch 2.9.1+cu130`, `transformers 4.57.6`, CUDA available. |
| 16:41-16:42 | Two remote tmux construction attempts failed before launch due nested shell quoting and bash arrays. Base64 payload launch avoided local interpolation. |
| 16:42 | Final guarded generation started in tmux. Guards passed: active-run registry 0 live/stale/malformed locks, one sample lock validated, H100/BF16 guard passed, and no `MAX_SAMPLES` or API stages were present. |
| 16:43-16:45 | Model loaded, classifier loaded, hooks installed, 500 manifest rows loaded, canonical jailbreak decode logged, and alpha `0.0` generation began. |

## What To Not Repeat

| Issue | Cost | Root cause | Rule |
|---|---|---|---|
| Treating the network-volume repo as current. | Required a 20.3 GiB sync on paid time and made launch timing ambiguous. | The attached volume persisted an older checkout and remote-only dirty outputs. | Before launch, prove `git rev-parse --short HEAD`, `git status --short`, and hashes for the wrapper, helper libraries, manifest, classifier, `requirements.txt`, and `pyproject.toml`. |
| Direct `rsync -av` to the RunPod volume. | Failed with code 23 and partial file copies. | The network volume did not permit ownership changes. | For network-volume mirrors, use `--no-owner --no-group --no-perms`; for full mirrors use `--delete --backup --backup-dir=<outside-repo> --partial`. |
| Launching from a tree under active `rsync --delete`. | Risked deleted/replaced code, manifests, or outputs. | Code, data, `.git`, and output paths lived under the same tree being mirrored. | Wait for sync before launching from in-repo outputs. If urgent, use an out-of-repo `OUTPUT_ROOT` and only after critical file hashes match. |
| Relying on wrapper defaults on a remote host. | Tmux exited immediately before logging. | `PROJECT_DIR` defaulted to the local workstation path. | Remote launch commands must set `PROJECT_DIR=/workspace/02-h-neurons` and should run a remote `DRY_RUN=1 TMUX_WRAPPED=1 INHIBIT_WRAPPED=1` before detached tmux. |
| Hashing only the top-level wrapper. | Wrapper failed before log creation. | `scripts/lib/inhibit_suspend.sh` was a required sourced dependency and was missing remotely. | Include sourced helpers and wrapper support files in the critical-hash checklist. |
| Trusting `/workspace/02-h-neurons/.venv`. | Stalled before generation, then lost time debugging package state. | The preserved env was on the network volume, symlinked to missing `/usr/bin/python3`, lacked `torch`/`transformers`, and had no `pip`. | Do not use an in-repo network-volume `.venv` as the runtime for paid runs. Prefer the baked `/opt/h-neurons/.venv`, a container-local `/opt/venvs/...`, or an explicitly verified `uv run --with-requirements` runtime. |
| Running `uv sync --frozen` in this repo. | Rewrote `.venv` to a tiny CPython 3.14 env and still failed. | This repo currently has `requirements.txt` but no `uv.lock`; `requires-python >=3.11` lets uv choose newer Python than the CUDA stack wants. | Without `uv.lock`, do not use `uv sync --frozen`. For requirements-based setup, pin `--python 3.11`. If a future lockfile is added, also pin the Python version for CUDA jobs. |
| Resolving `torch==2.9.1+cu130` against default PyPI. | Produced an unsatisfiable resolver error. | CUDA-local-version PyTorch wheels are on the PyTorch wheel index, not default PyPI. | Requirements-based remote setup must include `--index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match`. |
| Building complex remote tmux commands with nested quoting. | Two failed launch constructions before any work started. | Local shell expanded variables and bash arrays before the remote tmux script ran. | Use a checked-in remote wrapper or send a base64/heredoc payload. Avoid nested inline arrays through multiple shell layers. |

## Known Good Launch Shape

This is the exact runtime shape that passed guards and entered generation:

```bash
uv run --no-project \
  --with-requirements requirements.txt \
  --python 3.11 \
  --index https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match \
  python -m scripts.lib.pipeline active-run-status
```

Then the same `UV_RUN` prefix was used for:

- `python -m scripts.lib.pipeline validate-sample-locks data/manifests/jbb_d7_full_harmful500_seed42_mistral24b.lock.json`
- `python -m scripts.lib.pipeline gpu-hardware-guard --min-memory-gib 75 --name-pattern 'H100|A100'`
- `python scripts/run_intervention.py --model_key mistral_small_24b_instruct_2501 --model_path mistralai/Mistral-Small-24B-Instruct-2501 --classifier_path models/mistral24b_classifier_canonical.pkl --device_map cuda:0 --benchmark jailbreak --sample_manifest data/manifests/jbb_d7_full_harmful500_seed42_mistral24b.lock.json --alphas 0.0 1.0 3.0 --run_profile canonical --seed 42 --output_dir /workspace/mistral24b_anchor3_full500/experiment`

The successful tmux log showed:

- `Live locks: 0`, `Stale or remote locks: 0`, `Malformed locks: 0`.
- `Validated 1 sample manifest locks.`
- H100 guard: `NVIDIA H100 80GB HBM3`, `85017493504` bytes, BF16 supported.
- `MAX_SAMPLES unset; canonical serial generation; no API judging stages launched.`
- `Filtered to 500 samples via manifest ...jbb_d7_full_harmful500_seed42_mistral24b.lock.json`.
- Decode controls: `profile=canonical`, `batch_size=1`, `do_sample=True`, `temperature=0.7`, `top_k=20`, `top_p=0.8`, `max_new_tokens=5000`.

## Future Pre-Run Checklist

Run these before spending H100 minutes:

```bash
# Local
git status --short --branch
uv run python -m scripts.lib.pipeline active-run-status
sha256sum \
  scripts/infra/mistral24b_anchor3_jailbreak_first_pass.sh \
  scripts/lib/inhibit_suspend.sh \
  scripts/run_intervention.py \
  scripts/lib/pipeline.py \
  scripts/materialize_jailbreak_truncation_view.py \
  data/manifests/jbb_d7_full_harmful500_seed42_mistral24b.lock.json \
  models/mistral24b_classifier_canonical.pkl \
  requirements.txt pyproject.toml

# Remote
cd /workspace/02-h-neurons
git rev-parse --short HEAD
git status --short
sha256sum <same-files-as-above>
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

If a sync is needed:

```bash
rsync -rltDz --delete --backup --backup-dir=/workspace/repo-sync-backups/02-h-neurons-<UTC> \
  --partial --no-owner --no-group --no-perms \
  --exclude='/.venv/' --exclude='/.env' --exclude='/.pytest_cache/' \
  --exclude='/.ruff_cache/' --exclude='/logs/' --exclude='/wandb/' \
  --exclude='__pycache__/' \
  -e 'ssh -i ~/.ssh/id_ed25519 -p <port>' \
  ./ root@<host>:/workspace/02-h-neurons/
```

Before detached launch, prove the runtime imports with the exact command family:

```bash
cd /workspace/02-h-neurons
uv run --no-project \
  --with-requirements requirements.txt \
  --python 3.11 \
  --index https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match \
  python -c 'import sys, torch, transformers; print(sys.version.split()[0]); print(torch.__version__); print(transformers.__version__); print(torch.cuda.is_available())'
```

Do not judge or sync back claim outputs until the run finishes and `active-run-status` is clean.
