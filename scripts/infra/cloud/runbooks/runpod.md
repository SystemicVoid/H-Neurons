# RunPod Runbook

RunPod is the first supported provider because it blocks the Mistral 24B CP2 path. Paid actions stay rendered until an operator manually runs the printed command.

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

## Pod Setup

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

Sync only the bundle and required environment, then on the pod:

```bash
cd /workspace
git clone -b main /workspace/h-neurons-cp2.bundle /workspace/02-h-neurons
cd /workspace/02-h-neurons
export HF_HOME=/workspace/hf
export UV_CACHE_DIR=/workspace/uv-cache
command -v uv || python3 -m pip install uv
command -v tmux || (apt-get update && apt-get install -y tmux)
command -v rsync || (apt-get update && apt-get install -y rsync)
uv sync --no-dev
git ls-files --error-unmatch uv.lock >/dev/null 2>&1 || rm -f uv.lock
STAGES=splits,activations,classifier \
  TMUX_WRAPPED=1 \
  INHIBIT_WRAPPED=1 \
  PROJECT_DIR=/workspace/02-h-neurons \
  bash scripts/infra/mistral24b_replication.sh
```

The wrapper asserts A100/H100-class CUDA, BF16 support, and at least 75 GiB before GPU stages. It also validates the token-span summary and CP1 model-load smoke before activation/classifier work.
The official RunPod PyTorch template may not include `uv` or `tmux`; install only
those small tools if absent. It may also omit `rsync`; install it before long
artifact syncs, or use tar-over-SSH from `/workspace/02-h-neurons` as the fallback.
If `uv sync --no-dev` creates an untracked `uv.lock` from this repo's unlocked
project metadata, remove that generated file before the wrapper so provenance does
not record a dirty checkout. The wrapper itself uses `uv run --no-sync` after the
provisioning sync and removes only untracked generated `uv.lock` files before
capturing environment provenance. A committed `uv.lock` must never be deleted.

## Sync Back and Cleanup

Sync back only canonical CP2/CP3 outputs: split JSONs, `activations_llm_canonical/`, `models/mistral24b_classifier_canonical.pkl`, classifier dev/test metrics and provenance, environment capture, and the wrapper log. Before staging outputs locally, run:

```bash
uv run python -m scripts.lib.pipeline active-run-status
```

Prefer `rsync` when it is installed on both ends, but keep the allowlist narrow:

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

If remote `rsync` is unavailable and the pod is still alive, stream only the allowlisted
paths from the network volume:

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
