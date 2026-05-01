# RunPod SIMID Prospective Tail-Shard Launch Postmortem

- **Date:** 2026-04-30
- **Scope:** SIMID prospective-effect tail-shard launch on RunPod RTX 5090, gemma3-4b, datacenter EUR-IS-1, pod `pxqg9hp295qwhz`.
- **Canonical status:** ops-only event log. Run goals and split decisions live in the plan and `notes/research-log.md`.

## TL;DR

- Per-IP/CDN ingress shaping on this pod throttled rsync/scp from local home to ~16-30 KB/s; HuggingFace Hub from this same pod hit ~18 MB/s. Stage bundles via a private HF dataset, not direct rsync.
- **2026-05-01 addendum:** the private baked RunPod template solved dependency setup in principle, but failed the urgent-start requirement. Pod `wwimpi3wzix62q` in EUR-IS-1 launched at 13:53:23 UTC and was killed by the user at 16:41:09 UTC after ~2h48m of pulling/installing `ghcr.io/systemicvoid/h-neurons-runpod@sha256:15c9114d...`. RunPod metadata never exposed SSH; it stayed at `ssh.error="pod not ready"` and `uptimeSeconds=0`, so no remote SIMID shard started and no remote output was produced. The console reported slow layer progress such as 3/8 layers complete and later `e2f70f97f95f` 31% / `a6edaf499c88` 82%. The pinned amd64 manifest is ~5.0 GiB compressed, dominated by layers `e2f70f97f95f` (~3.10 GB) and `a6edaf499c88` (~1.51 GB). For bursty tail-shard launches, a large private CUDA image is not a reliable "launch now" path unless the worker image cache is already warm.
- **2026-05-01 deeper infra decision:** the right burst pattern is not "bake more into the private image." Use an official/common RunPod PyTorch image first, then hydrate the repo, model cache, and optionally a filtered non-CUDA wheelhouse over Hugging Face. For RTX 5090/Blackwell, current RunPod official template `runpod-torch-v280` points at `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (OCI index `sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`, amd64 manifest `sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851`). It is larger than the private image (~10.6 GB compressed amd64), so the advantage is expected worker-cache locality, not raw image size. Treat it as a canary: if SSH is not available within 15-20 minutes, kill and switch paths. After SSH, create a `--system-site-packages` venv that reuses the image's CUDA Torch; do not run project `uv sync` or install `torch`, `triton`, or `nvidia-*` wheels on the pod.
- **`UV_HTTP_TIMEOUT=900` is mandatory before `uv sync` on RunPod.** Default 30 s is far too short for `nvidia-cuda-runtime`/`nvidia-cusparse`/`nvidia-cusparselt-cu13` wheels (~150 MiB each) on RunPod's slow PyPI/Fastly route. We burned ~13 min on this exact failure even though yesterday's mistral24b CP1 postmortem warned the project venv would not skip its CUDA stack.
- 5090 stock was empty in US datacenters at launch time. EUR-IS-1 (NL) had stock. Run `runpodctl gpu list -o json` before `render-launch`.
- Bundle pinning needs a tag; piping `main` into `git bundle create` captures whatever HEAD is at bundle time, which can drift from your intended commit during plan-mode wait.
- Default RunPod `runpod-torch-v240` template ships without `tmux`, `rsync`, or `pigz`. Bootstrap them; never background `apt-get` over ssh.
- `/workspace` on RunPod is MooseFS over LAN; rsync `--inplace` and avoid many-small-files patterns.

## What we expected vs. what happened

Expected: bundle from local main, scp to pod, run wrapper. Actual: bundle drifted off the intended commit during plan mode; ingress to pod was CDN-throttled into single-digit KB/s; default template missing core tools; `/workspace` is a network filesystem; HF Hub turned out to be the only fast path into the pod.

2026-05-01 retry expectation: use the baked private template to skip `uv sync`, pull the HF-staged bundle, and run the seed3-5 direction-tail shard. Actual: the pod never reached runtime because the custom image cold-pull/install consumed the entire acceleration window. Local progressed into `random_direction_seed2` while the remote pod was still pre-SSH, so the remote split lost its wallclock value and the pod was killed before any remote work started.

## 1. Snags and resolutions

| Snag | Root cause | Resolution |
|---|---|---|
| Smoke-clone HEAD was `c49183c`, not the `d185f12` the local run used. | `git bundle create ... main` captures the current main HEAD; main advanced (mistral data commits) during plan-mode review. | Pin with a lightweight tag: `git tag simid-tail-pin d185f12 && git bundle create /tmp/h-neurons-simid-tail.bundle simid-tail-pin && git tag -d simid-tail-pin`. Smoke-clone HEAD then matched. |
| `runpodctl pod create` returned out-of-stock in US-MD-1, US-MO-1, US-WA-1. | No 5090 inventory in US regions at launch time. | Created in EUR-IS-1 (NL). Run `runpodctl gpu list -o json` upfront and prefer EU regions for 5090 outside US business hours. |
| `cloudctl ssh-info` returned "could not derive public SSH" on first attempt. | Pod was not yet schedulable; metadata briefly returned `ssh.error="pod not ready"`. | Polled with 20s sleeps; resolved on second call (157.157.221.29:54688). Do not delete the pod just for this metadata-readiness gap; use the runbook SSH-info retry pattern. |
| `scp` of 162 MB bundle disconnected at ~4.7 MB with Broken Pipe. | Idle/timeout on the route; no keepalives. | Switched to rsync with ssh keepalives. Still slow (see CDN shaping below). |
| `apt-get install -y rsync tmux pigz` hung indefinitely. | A prior backgrounded `apt-get` (from a dead ssh) still held PID 268 with the dpkg lock. | `pkill -9 apt-get && rm -f /var/lib/apt/lists/lock /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock`, then re-run apt. Never background `apt-get` over ssh; run synchronously or under `nohup`/tmux. |
| Default `runpod-torch-v240` template lacked `tmux`, `rsync`, `pigz`. | Template is minimal. | `apt-get update && apt-get install -y --no-install-recommends rsync tmux pigz`. `--no-install-recommends` keeps it fast. |
| Direct ingress to pod was throttled to single-digit KB/s. | RunPod ingress is per-source/CDN-route shaped, not per-pod aggregate. | Stage bundle through HuggingFace Hub (private dataset). End-to-end ~45 s vs ~90 min via direct rsync. Recipe below. |
| `/workspace` exhibited slow rename and small-file behavior. | `/workspace` is `mfs#eur-is-1.runpod.net:9421` — MooseFS over LAN. | Use rsync `--inplace`; avoid tight-loop small-file writes; treat `/workspace` as a network filesystem. |
| `uv sync --no-dev` failed at ~13 min with `Failed to download distribution due to network timeout. Try increasing UV_HTTP_TIMEOUT (current value: 30s).` on `nvidia-cuda-runtime==13.0.96` (162 MiB). | RunPod's PyPI/Fastly route is slow (single-digit-to-low-hundreds KB/s); default 30 s timeout is exceeded by every nvidia-* wheel. The pod ships with system torch `2.4.1+cu124` but the project venv resolves a newer torch (`2.9.1+cu130` or `2.11+`) and a fresh `cuda-toolkit` chain regardless. Yesterday's mistral24b CP1 postmortem flagged that base-image CUDA does not skip project setup, but the timeout was not pinned then. | `export UV_HTTP_TIMEOUT=900` (15 min/file). The `uv-cache` is preserved across retries, so resuming after the timeout finishes the partials. Recipe added to `scripts/infra/cloud/runbooks/runpod.md` Pod Setup and `scripts/infra/CLAUDE.md` Cloud Adapter. |
| Baked private template spent ~2h48m pulling/installing and never exposed SSH. | RunPod templates are launch recipes, not VM snapshots. A cold worker still has to pull and extract uncached private GHCR layers before SSH exists; the pinned image is ~5.0 GiB compressed with two very large layers. | Do not use this large private image for urgent one-off shards. Use a smaller/common-base image plus HF bundle/runtime setup, a deliberately warm retained pod, or a provider/path with image prewarming or snapshot semantics. Kill the attempt once image-pull delay consumes the expected speedup margin. |
| Official PyTorch template is still a large image. | `runpod-torch-v280`/`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` is ~10.6 GB compressed amd64; its expected win is official/common layer cache locality, not small transfer size. | Use it as a timed startup canary, not a guarantee. After SSH, avoid project `uv sync`; build a system-site venv on top of the image's CUDA Torch and hydrate only repo/model/non-CUDA wheels through HF. |

## 2. CDN ingress shaping (measured)

All measured on pod `pxqg9hp295qwhz`, EUR-IS-1, on 2026-04-30:

| Path | Throughput |
|---|---|
| local home → pod `/workspace` via single-stream rsync | ~16-30 KB/s |
| local home → pod `/workspace` via 8 parallel scp streams | ~134 KB/s aggregate |
| pod → `codeload.github.com` (Fastly) | ~20 KB/s |
| pod → `objects.githubusercontent.com` (GH releases) | ~309 KB/s |
| pod → `huggingface.co` (CloudFront, authed) | ~18 MB/s |

The shaping is per-source/CDN-route, not per-pod aggregate. Multiplexing scp streams gave only marginal gain. HF Hub via authed CloudFront was the only fast path observed.

Implication for model weights: gemma-3-4b safetensors (~8 GB) downloads in ~7-8 min from HF on this pod. The HF cache path is fine on EUR-IS-1.

## 3. Recipes / boilerplate

### Bundle pinning to an explicit commit

```bash
TARGET=d185f12
git tag simid-tail-pin "$TARGET"
git bundle create /tmp/h-neurons-simid-tail.bundle simid-tail-pin
git tag -d simid-tail-pin

# Smoke-verify the bundle HEAD matches before launch
rm -rf /tmp/h-neurons-bundle-smoke
git clone -b main /tmp/h-neurons-simid-tail.bundle /tmp/h-neurons-bundle-smoke 2>/dev/null \
  || git clone /tmp/h-neurons-simid-tail.bundle /tmp/h-neurons-bundle-smoke
git -C /tmp/h-neurons-bundle-smoke rev-parse HEAD  # must equal $TARGET
```

### GPU stock probe before render-launch

```bash
runpodctl gpu list -o json | jq '.[] | select(.id|test("5090"))'
runpodctl datacenter list -o json
# Outside US business hours, prefer EUR-IS-1 / EU regions for 5090.
```

### Pod bootstrap (default `runpod-torch-v240` template)

```bash
ssh -p "$SSH_PORT" root@"$SSH_HOST" '
set -euo pipefail
apt-get update
apt-get install -y --no-install-recommends rsync tmux pigz
command -v uv || python3 -m pip install uv
'
# Never background apt-get over ssh. If a prior apt is wedged:
#   pkill -9 apt-get
#   rm -f /var/lib/apt/lists/lock /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock
```

### HF-staged bundle transfer (the fast path)

Local upload:

```bash
uv run --with huggingface_hub python - <<'PY'
import os
from huggingface_hub import HfApi, create_repo

token = os.environ["HF_TOKEN"]
user = os.environ["HF_USER"]            # e.g. "<user>"
repo = f"{user}/h-neurons-simid-tail-bundle"
bundle = "/tmp/h-neurons-simid-tail.bundle"

create_repo(repo, token=token, repo_type="dataset", private=True, exist_ok=True)
HfApi().upload_file(
    path_or_fileobj=bundle,
    path_in_repo="h-neurons-simid-tail.bundle",
    repo_id=repo,
    repo_type="dataset",
    token=token,
)
print("uploaded")
PY
```

Pod download:

```bash
ssh -p "$SSH_PORT" root@"$SSH_HOST" '
set -euo pipefail
mkdir -p /workspace
curl -fL -H "Authorization: Bearer '"$HF_TOKEN"'" \
  -o /workspace/h-neurons-simid-tail.bundle \
  "https://huggingface.co/datasets/'"$HF_USER"'/h-neurons-simid-tail-bundle/resolve/main/h-neurons-simid-tail.bundle"
git -C /workspace clone -b main /workspace/h-neurons-simid-tail.bundle /workspace/02-h-neurons
'
```

A fine-grained HF token with `write` scope on the user's own namespace is sufficient. Tokens listed in `reference_huggingface_token.md`.

### HF dataset cleanup post-run

```bash
uv run --with huggingface_hub python - <<'PY'
import os
from huggingface_hub import HfApi
HfApi().delete_repo(
    repo_id=f"{os.environ['HF_USER']}/h-neurons-simid-tail-bundle",
    repo_type="dataset",
    token=os.environ["HF_TOKEN"],
)
print("deleted")
PY
```

### MooseFS-friendly rsync (only if HF path is unavailable)

```bash
rsync -av --inplace \
  -e "ssh -p $SSH_PORT -o ServerAliveInterval=30 -o ServerAliveCountMax=4" \
  /tmp/h-neurons-simid-tail.bundle \
  root@"$SSH_HOST":/workspace/
```

`--inplace` avoids the slow rename across MooseFS. Keepalives prevent the Broken-Pipe-at-4.7-MB pattern.

### SSH-info retry pattern (do not delete the pod)

```bash
for i in 1 2 3 4 5 6; do
  if uv run python scripts/infra/cloudctl.py ssh-info --pod-id "$POD_ID" \
      | tee /tmp/ssh-info.json \
      | jq -e '.host and .port' >/dev/null; then
    break
  fi
  sleep 20
done
```

See `scripts/infra/cloud/runbooks/runpod.md` for why retry, not recreate.

## 4. Observed environment facts

- `/workspace` mount: `mfs#eur-is-1.runpod.net:9421` (MooseFS over LAN).
- HF model download throughput on EUR-IS-1: ~18 MB/s on this pod (CloudFront, authed).
- Default template `runpod-torch-v240` is missing: `tmux`, `rsync`, `pigz`.
- `apt-get` lock paths to clear after a wedged background apt: `/var/lib/apt/lists/lock`, `/var/lib/dpkg/lock-frontend`, `/var/lib/dpkg/lock`.
- Baked template retry pod `wwimpi3wzix62q` launched 2026-05-01 13:53:23 UTC and was killed at 16:41:09 UTC before SSH/runtime; no remote outputs were created.
- Pinned baked image manifest: ~5.0 GiB compressed. Largest compressed layers: `e2f70f97f95f` ~3.10 GB and `a6edaf499c88` ~1.51 GB.
- Official RunPod Blackwell PyTorch template observed 2026-05-01: `runpod-torch-v280` -> `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`; OCI index digest `sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`; amd64 manifest digest `sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851`; compressed amd64 layer sum ~10.6 GB.

## 5. Do not repeat

- Do not assume `git bundle create ... main` captures the commit you reviewed; pin with a tag.
- Do not pick a US datacenter for 5090 without first checking `runpodctl gpu list -o json`.
- Do not assume a RunPod template makes a large private image start quickly. The template is a launch recipe; a cold worker still pulls and extracts every uncached image layer before SSH exists. For urgent sharding, prefer one of:
  - a small official/base template plus HF-staged repo bundle and narrowly-scoped runtime install;
  - a deliberately warm retained pod or retained worker/cache strategy when spend is acceptable;
  - a smaller custom image that leaves model/cache/repo artifacts to HF download and keeps CUDA layers aligned with a commonly cached RunPod base;
  - another provider/path with explicit image prewarming or VM snapshot semantics.
- Do not treat "official PyTorch image" as proof of instant boot. Treat it as a 15-20 minute canary. The expected benefit is cache locality; if the target worker is cold, the remote shard can still lose its speedup margin.
- Do not let a burst pod resolve the repo's full CUDA stack. Use the image's CUDA Torch, a `--system-site-packages` venv, and a filtered non-CUDA dependency/wheelhouse path.
- Do not keep waiting on a cold image pull after it has consumed the expected acceleration margin. At that point the remote shard is no longer a speedup; cancel and fall back to local or a different setup path.
- Do not background `apt-get` over ssh.
- Do not retry direct rsync/scp into the pod after seeing single-digit KB/s — switch to HF staging.
- Do not delete and recreate the pod when `cloudctl ssh-info` returns "pod not ready"; poll instead.
- Do not leave the private HF bundle dataset around after the run; delete it.
- **Do not run `uv sync` on a RunPod pod without `export UV_HTTP_TIMEOUT=900` first.** The 30 s default fails on every nvidia-* wheel here. Yesterday's CP1 postmortem warned the project venv won't skip CUDA setup; today's lesson is the timeout, not just the principle.
