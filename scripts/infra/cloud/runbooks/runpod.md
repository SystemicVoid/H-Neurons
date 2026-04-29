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

Choose one live H100 SXM Secure Cloud datacenter before creating storage. Default to `US-CA-2` only if that datacenter has H100 SXM stock; otherwise choose exactly one live H100 SXM datacenter and reuse it for both commands below.

Render the 200 GB network volume create command, inspect it, then manually run the exact output:

```bash
DATA_CENTER_ID=US-CA-2
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
  --data-center-id "$DATA_CENTER_ID"
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
uv sync --no-dev
git ls-files --error-unmatch uv.lock >/dev/null 2>&1 || rm -f uv.lock
STAGES=splits,activations,classifier \
  TMUX_WRAPPED=1 \
  INHIBIT_WRAPPED=1 \
  PROJECT_DIR=/workspace/02-h-neurons \
  bash scripts/infra/mistral24b_replication.sh
```

The wrapper asserts H100/A100-class CUDA, BF16 support, and at least 75 GiB before GPU stages. It also validates the token-span summary and CP1 model-load smoke before activation/classifier work.
The official RunPod PyTorch template may not include `uv` or `tmux`; install only
those small tools if absent. If `uv sync --no-dev` creates an untracked `uv.lock`
from this repo's unlocked project metadata, remove that generated file before the
wrapper so provenance does not record a dirty checkout.

## Sync Back and Cleanup

Sync back only canonical CP2/CP3 outputs: split JSONs, `activations_llm_canonical/`, `models/mistral24b_classifier_canonical.pkl`, classifier dev/test metrics and provenance, environment capture, and the wrapper log. Before staging outputs locally, run:

```bash
uv run python -m scripts.lib.pipeline active-run-status
```

Delete the pod, then verify cleanup:

```bash
uv run python scripts/infra/cloudctl.py cleanup-check --allow-network-volumes
```

Keep the network volume only if CP4/CP5 reuse is immediate. If reuse is not immediate, sync critical artifacts locally and delete the volume too.

References: [network volumes](https://docs.runpod.io/storage/network-volumes), [Pods pricing](https://docs.runpod.io/pods/pricing), [SSH over exposed TCP](https://docs.runpod.io/pods/configuration/use-ssh), [runpodctl pod](https://docs.runpod.io/runpodctl/reference/runpodctl-pod), and [runpodctl network-volume](https://docs.runpod.io/runpodctl/reference/runpodctl-network-volume).
