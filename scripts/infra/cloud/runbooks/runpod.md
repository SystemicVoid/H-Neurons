# RunPod Runbook

RunPod is the first supported provider because it blocks the Mistral 24B CP2 path. Paid actions stay rendered until an operator manually runs the printed command.

## Template-Based Launch

RunPod profiles launch through the private template `h-neurons-runpod-private`
(`v88hqzvuxk`), backed by a GHCR image with the project venv baked at
`/opt/h-neurons/.venv`. The normal project launch path is a single local render
command; inspect its paid `runpodctl pod create ... --template-id v88hqzvuxk`
output, then run that output deliberately:

```bash
uv run python scripts/infra/cloudctl.py render-launch --profile mistral24b-runpod --stage cp1 --attempt 1
```

`render-launch` calls `runpodctl template get v88hqzvuxk -o json` and verifies
the template image digest against `runpod.expected_image_digest` in the profile
before printing the paid command. RunPod profiles without
`expected_image_digest` fail unless explicitly marked
`allow_unpinned_image = true` for non-production use. A profile that swaps
`template_id` for a direct `image` also fails by default. A stale template is
never a fallback case; update the template or the profile digest before
rendering a launch. Only bypass the direct-image guard with both
`--allow-image-fallback` and `--confirmed-image-pin` after manually verifying a
digest-pinned fallback image is acceptable for the launch.

Do not run remote dependency setup on RunPod pods. The template already ships
`uv`, `tmux`, `rsync`, `pigz`, `openssh-server`, CUDA tooling, and the baked
project venv.

## Template Maintenance

Rebuild the template image whenever `requirements.txt`, `pyproject.toml`, or
`scripts/infra/docker/*` changes.

One-time GHCR login:

```bash
echo "$GH_PAT_TOKEN" | docker login ghcr.io -u SystemicVoid --password-stdin
```

Bake and push:

```bash
REGISTRY=ghcr.io/systemicvoid bash scripts/infra/cloud/runpod_bake_image.sh
```

The GHCR package is private. Keep a RunPod registry credential named
`ghcr-systemicvoid` with a GitHub PAT that has `read:packages`, then update the
private RunPod template `h-neurons-runpod-private` after each bake:

```bash
set -a
source .env
set +a

: "${GH_PAT_TOKEN:?missing GH_PAT_TOKEN}"

runpodctl registry list -o json

# Only create this if `ghcr-systemicvoid` is missing from the registry list.
runpodctl registry create \
  --name ghcr-systemicvoid \
  --username SystemicVoid \
  --password "$GH_PAT_TOKEN"

DIGEST="$(cat /tmp/h-neurons-runpod-digest)"
IMAGE="ghcr.io/systemicvoid/h-neurons-runpod@${DIGEST}"
runpodctl template update v88hqzvuxk --image "$IMAGE"
# Update expected_image_digest in both RunPod profiles to ${DIGEST}, then verify:
uv run python scripts/infra/cloudctl.py verify-image-pin --profile mistral24b-runpod
uv run python scripts/infra/cloudctl.py verify-image-pin --profile gemma3-4b-runpod
```

If the registry credential already exists, do not recreate it; reuse the existing
auth record. `runpodctl template update` works because the template already has
`containerRegistryAuthId`; if recreating the template from scratch, use the
RunPod console or REST API to attach the registry credential. Current template:
`v88hqzvuxk`. After each bake, update `expected_image_digest` in both RunPod
profiles to the new `/tmp/h-neurons-runpod-digest` value before running the
verification commands above.

Local image checks:

```bash
docker buildx build --load --platform=linux/amd64 \
  -f scripts/infra/docker/Dockerfile \
  -t h-neurons-runpod:test .
docker run --rm h-neurons-runpod:test \
  bash -lc 'python -V && python -c "import torch; print(torch.__version__)" && command -v ptxas'
docker run --rm -v "$PWD:/workspace/02-h-neurons" -w /workspace/02-h-neurons \
  h-neurons-runpod:test \
  uv run --no-sync python -m scripts.lib.pipeline gpu-preflight
docker run --rm -v "$PWD:/workspace/02-h-neurons" -w /workspace/02-h-neurons \
  -e PYTHONPATH=scripts \
  h-neurons-runpod:test \
  uv run --no-sync python -c 'import run_simid'
```

If local Docker has NVIDIA runtime support, also run:

```bash
docker run --rm --gpus all h-neurons-runpod:test python /opt/smoke/triton.py
```

## CP2 Volume and Launch

Before paid work:

```bash
git status --short --branch
uv run python scripts/infra/cloudctl.py status
uv run python scripts/infra/cloudctl.py preflight --profile mistral24b-runpod --stage cp2 --attempt 1
git bundle create /tmp/h-neurons-cp2.bundle main
git clone -b main /tmp/h-neurons-cp2.bundle /tmp/h-neurons-bundle-smoke
```

Choose one live A100 SXM Secure Cloud datacenter before creating storage. Default to `US-MD-1` only if that datacenter has A100 SXM stock; otherwise choose exactly one live A100 SXM datacenter and reuse it for both commands below. H100 SXM remains an explicit speed/availability fallback, not the default.

For an existing network volume, do not choose from the profile datacenter list. First read the volume's actual datacenter, then use that exact datacenter for the launch and confirm live stock for the selected GPU:

```bash
NETWORK_VOLUME_ID=<volume-id>
runpodctl network-volume list -o json
runpodctl datacenter list -o json
```

If the retained volume is in a datacenter without live A100 SXM stock, stop and make an explicit operator decision: either use H100 in that same datacenter for this run, or transfer/sync artifacts to a new volume in a datacenter with live A100 stock. Do not launch a pod in one datacenter against a volume in another.

Render the 200 GB network volume create command, inspect it, then manually run the exact output:

```bash
DATA_CENTER_ID=US-MD-1
uv run python scripts/infra/cloudctl.py render-volume-create \
  --profile mistral24b-runpod \
  --name hneurons-mistral24b-cp2 \
  --data-center-id "$DATA_CENTER_ID"
```

After creation, render the pod launch with the new volume and the same datacenter:

```bash
NETWORK_VOLUME_ID=<volume-id>
uv run python scripts/infra/cloudctl.py render-launch \
  --profile mistral24b-runpod \
  --stage cp2 \
  --attempt 1 \
  --network-volume-id "$NETWORK_VOLUME_ID" \
  --data-center-id "$DATA_CENTER_ID" \
  --confirmed-gpu-stock
```

If A100 is unavailable in the volume datacenter and the operator chooses the H100 fallback, render that fallback explicitly:

```bash
uv run python scripts/infra/cloudctl.py render-launch \
  --profile mistral24b-runpod \
  --stage cp4 \
  --attempt 1 \
  --network-volume-id "$NETWORK_VOLUME_ID" \
  --data-center-id "$DATA_CENTER_ID" \
  --gpu-id "NVIDIA H100 80GB HBM3" \
  --allow-gpu-fallback \
  --confirmed-gpu-stock
```

RunPod network volumes for Pods are Secure Cloud-only, replace the default `/workspace` volume, and must be attached during deployment. Under 1 TB they are billed as network-volume storage, so keep the volume only for the short CP4/CP5 reuse window.

## Pod Setup (image-based)

Derive direct SSH from pod metadata and prove access:

```bash
POD_ID=<pod-id>
uv run python scripts/infra/cloudctl.py ssh-info --pod-id "$POD_ID"
ssh -o StrictHostKeyChecking=accept-new -p "$SSH_PORT" root@"$SSH_HOST" true
```

Freshly launched pods can briefly return pod metadata with `ssh.error="pod not ready"`
even when `runpodctl ssh info <pod-id>` already has the exposed TCP endpoint.
`cloudctl ssh-info` falls back to that command; if both paths fail, wait and retry
before deleting or recreating the pod.

Stage the git bundle through the Hugging Face route from Footguns, then on the pod:

```bash
cd /workspace
git clone -b main /workspace/h-neurons-cp2.bundle /workspace/02-h-neurons
export HF_HOME=/workspace/hf
export PROJECT_DIR=/workspace/02-h-neurons
cd /workspace/02-h-neurons
STAGES=splits,activations,classifier \
  TMUX_WRAPPED=1 INHIBIT_WRAPPED=1 \
  bash scripts/infra/mistral24b_replication.sh
```

The wrapper asserts A100/H100-class CUDA, BF16 support, and at least 75 GiB before GPU stages. It also validates the token-span summary and CP1 model-load smoke before activation/classifier work.
The image ships `uv`, `tmux`, `rsync`, `pigz`, `openssh-server`, and the baked
project venv. The wrappers use `uv run --no-sync` with
`UV_PROJECT_ENVIRONMENT=/opt/h-neurons/.venv`, so no PyPI traffic should happen
on the pod.

## Sync Back and Cleanup

Sync back only canonical CP2/CP3 outputs: split JSONs, `activations_llm_canonical/`, `models/mistral24b_classifier_canonical.pkl`, classifier dev/test metrics and provenance, environment capture, and the wrapper log. Before staging outputs locally, run:

```bash
uv run python -m scripts.lib.pipeline active-run-status
```

Use the image-provided `rsync` and keep the allowlist narrow:

```bash
rsync -av --prune-empty-dirs -e "ssh -p $SSH_PORT" \
  --include='data/' \
  --include='data/mistral24b/' \
  --include='data/mistral24b/pipeline/' \
  --include='data/mistral24b/pipeline/*_qids_llm.json' \
  --include='data/mistral24b/pipeline/*_qids_llm.json.provenance.*.json' \
  --include='data/mistral24b/pipeline/activations_llm_canonical/***' \
  --include='data/mistral24b/pipeline/classifier_canonical_*_metrics.json' \
  --include='data/mistral24b/pipeline/classifier_canonical_*_metrics.json.provenance.*.json' \
  --include='data/mistral24b/pipeline/environment_<RUN_TS>.txt' \
  --include='models/' \
  --include='models/mistral24b_classifier_canonical.pkl' \
  --include='logs/' \
  --include='logs/mistral24b_replication_<RUN_TS>.log' \
  --exclude='*' \
  root@"$SSH_HOST":/workspace/02-h-neurons/ ./
```

If a damaged or mismatched pod lacks `rsync` but is still alive, stream only the
allowlisted paths from the network volume:

```bash
ssh -p "$SSH_PORT" root@"$SSH_HOST" 'cd /workspace/02-h-neurons && tar -cf - \
  data/mistral24b/pipeline/activations_llm_canonical \
  data/mistral24b/pipeline/*_qids_llm.json \
  data/mistral24b/pipeline/*_qids_llm.json.provenance.*.json \
  data/mistral24b/pipeline/classifier_canonical_*_metrics.json \
  data/mistral24b/pipeline/classifier_canonical_*_metrics.json.provenance.*.json \
  data/mistral24b/pipeline/environment_<RUN_TS>.txt \
  models/mistral24b_classifier_canonical.pkl \
  logs/mistral24b_replication_<RUN_TS>.log' | tar -xpf -
```

Delete the pod, then verify cleanup:

```bash
uv run python scripts/infra/cloudctl.py cleanup-check --allow-network-volumes
```

With `--allow-network-volumes`, nonzero storage-only spend is acceptable only when
there are no live pods and retained network volumes are the only listed resources.
Keep the network volume only if CP4/CP5 reuse is immediate. If reuse is not
immediate, sync critical artifacts locally and delete the volume too.

References: [network volumes](https://docs.runpod.io/storage/network-volumes), [Pods pricing](https://docs.runpod.io/pods/pricing), [SSH over exposed TCP](https://docs.runpod.io/pods/configuration/use-ssh), [runpodctl pod](https://docs.runpod.io/runpodctl/reference/runpodctl-pod), and [runpodctl network-volume](https://docs.runpod.io/runpodctl/reference/runpodctl-network-volume).

## Footguns

- Stage git bundles through a private Hugging Face dataset rather than direct
  `rsync`/`scp`; direct ingress was throttled to single-digit KB/s, while HF Hub
  hit about 18 MB/s on the same pod.
- Keep using `/workspace` for HF cache, model weights, cloned repo, and outputs.
  The baked venv lives in the container layer at `/opt/h-neurons/.venv`.
- See `notes/icml/gemma3_4b/2026-04-30-runpod-simid-tail-postmortem.md` for the
  HF-staged bundle recipe and EUR-IS-1 ingress observations. Its default-template
  bootstrap commands are historical and superseded by this template runbook.
