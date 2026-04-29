# CP1 RunPod Smoke Postmortem and Future Guardrails

- **Date:** 2026-04-29
- **Scope:** CP1 `model_smoke` for `mistralai/Mistral-Small-24B-Instruct-2501` on RunPod H100 SXM.
- **Canonical status:** keep progress state in [2026-04-28-5.5-pro-l1-mitigation-strategy.md](2026-04-28-5.5-pro-l1-mitigation-strategy.md). This note is an operational postmortem and runbook only.
- **Outcome:** the model load and one-token generation succeeded in BF16 with no quantization, but CP1 remains partial because the exact handoff validator expected a non-null `runtime.hf_device_map` even though the wrapper used explicit `DEVICE_MAP=cuda:0`.
- **Adapter follow-up:** Zombuul is vendored read-only at `vendor/zombuul`; local fixes for the issues below belong in [scripts/infra/cloud/](../../../scripts/infra/cloud/), not in an installed plugin.

## 1. What Cost Time, Money, or Restarts

| Snag | Cost | Root cause | Future rule |
|---|---|---|---|
| Relay SSH address failed: `ssh "$POD_ID@ssh.runpod.io"` returned `Permission denied (publickey)`. | Pod waited while orchestration was debugged on paid time. | RunPod v2 exposed the usable direct SSH endpoint as `root@<ip> -p <port>` via pod metadata; the relay form in the older infra note was stale for this run. | After pod creation, discover SSH from `runpodctl pod get "$POD_ID"` and prove `ssh ... true` before any `scp`, setup, or dependency install. Treat `<pod-id>@ssh.runpod.io` as untrusted until tested. |
| Git bundle clone failed because the bundle had no default HEAD. | Required another orchestration iteration and pod restart. | `git bundle create /tmp/h-neurons-cp1.bundle main` produced a usable `main` ref but `git clone /workspace/h-neurons-cp1.bundle ...` had no default branch to infer. | Clone bundles with an explicit branch: `git clone -b main /workspace/h-neurons-cp1.bundle /workspace/02-h-neurons`. Better: test the exact bundle locally with `git clone -b main /tmp/h-neurons-cp1.bundle /tmp/bundle-smoke`. |
| Remote `uv sync --no-dev` failed on the pod before the metadata fix. | Paid H100 minutes were spent discovering a resolver issue. | `pyproject.toml` claimed `requires-python >=3.10`, while the resolved dependency set included `scikit-learn>=1.8.0`, which requires Python >=3.11. The pod image used Python 3.11, but the project metadata still promised 3.10 compatibility. | Before launching paid compute, run `uv sync --no-dev --dry-run` from the exact committed state and fix Python lower bounds locally. Do not let the paid pod be the first clean dependency-resolution environment. |
| The strict acceptance jq failed after the successful model load. | The run had to be treated as partial, blocking CP2 and requiring a memo update instead of closing CP1. | The validator required `.runtime.hf_device_map != null`. Transformers did not populate `hf_device_map` for the explicit single-GPU load, even though the artifact recorded `requested_device_map=="cuda:0"`, `loaded_primary_device=="cuda:0"`, H100 CUDA allocation, BF16 dtype, no quantization, and successful generation. | Validators are part of the workload. Before paid launch, validate the exact acceptance predicate against the intended runtime mode. If using `DEVICE_MAP=cuda:0`, either accept explicit-device evidence, record an effective map in `model_load_smoke.py`, or intentionally run with `DEVICE_MAP=auto`. |
| `logs/` is ignored by git. | Evidence could have been missed during staging. | The wrapper log lived under an ignored directory, so normal `git status` did not list the file. | For committed run evidence under `logs/`, use `git status --ignored <path>` and `git add -f <exact log path>`. Prefer mirroring critical machine-readable evidence under `data/mistral24b/preflight/` when possible. |
| Remote `uv sync` generated an untracked `uv.lock`. | Minor review noise during artifact sync decisions. | The remote clone from the bundle was not a normal working checkout with all local untracked state. `uv sync` could create a lockfile in the pod workspace. | Sync only enumerated evidence files back from the pod. Do not tar arbitrary remote workspace roots. Do not import remote `uv.lock` unless lockfile policy is deliberately changed. |
| Pre-commit hooks stashed unrelated dirty work during commits. | Added review overhead and risk of confusing user edits with CP1 changes. | The local worktree already had unrelated modified/untracked files. Hooks stashed and restored them around each commit. | Start every handoff with `git status --short --branch`, write down unrelated dirt, and stage only explicit paths. Expect hook-managed stash/restore and never assume a clean tree after commit without checking again. |
| RunPod stock and CLI output were treated as stable enough in the plan. | Launch orchestration needed retries and dynamic parsing. | H100 SXM stock was low, and `runpodctl` response shapes/SSH details were not stable enough for a single hard-coded command path. | Keep a region fallback list, but make the launch script idempotent: parse pod ID defensively, install a cleanup trap immediately, and re-check pods/volumes/spend after every failure. |
| Local CUDA was available but insufficient. | Risk of accidentally running the 24B load on a 16 GB RTX 5060 Ti. | `torch.cuda.is_available()` was true locally, but the local device was not the target H100-class environment. | A green local CUDA check is not permission to load the model. For CP1, assert H100 name, >=75 GiB memory, and BF16 support on the pod before invoking `scripts/infra/mistral24b_replication.sh`. |
| Storage strategy was too binary. | Avoiding persistent storage was defensible for one bounded CP1 attempt, but repeated cold-start/setup failures made ephemeral-only execution more expensive than planned. | The plan optimized against forgotten storage billing, not against paid H100 time lost to repeated setup, model/cache downloads, and restart friction. | Use ephemeral storage for a first one-shot CP1 smoke. If CP1 needs a second paid attempt, or for any CP2+ work, create a Secure Cloud network volume in the chosen H100 region and prewarm caches before more H100 runs. |
| Ready-made CUDA image did not remove project setup cost. | The pod booted quickly, but `uv sync --no-dev` still created a fresh project environment and pulled the repo's resolved ML stack, including transitive CUDA/PyTorch packages. | `runpod-torch-v240` supplied a CUDA-capable base image; the repo dependency graph still resolves its own project venv rather than reusing the image's preinstalled Python packages. | Treat base images as OS/CUDA bootstraps, not full environment caches. For repeated work, either store `.venv`/`UV_CACHE_DIR` on a network volume or bake a custom image/template after the dependency set is proven. |

## 2. Compounded Lessons

1. **Paid GPU time starts after local proof, not during discovery.** Dependency resolution, bundle cloning, validator semantics, and SSH command shape all need local or zero-cost proof before the pod exists.
2. **The launch artifact is the committed repo, not the dirty worktree.** If the pod gets a git bundle from `main`, every required code or metadata change must be committed first.
3. **A validator failure is not automatically a model failure.** Preserve the artifact, identify the exact predicate that failed, and block only the next stage that depends on that predicate.
4. **RunPod cleanup is part of the experiment, not an afterthought.** A paid run is not complete until pods, network volumes, and current hourly spend are checked after cleanup.
5. **Use explicit, enumerated artifact sync.** Pull back the smoke JSON, provenance sidecar, environment capture, and exact wrapper log. Do not sync broad directories from an ephemeral pod.
6. **Persistent storage is a threshold decision.** A network volume is waste for a single clean smoke, cheap insurance after the first paid restart, and the default for CP2+ activation/classifier work.

## 3. Storage and Image Policy

As of the 2026-04-29 RunPod docs, network volumes persist independently of compute, are backed by NVMe storage, cost `$0.07/GB/month` for the first 1 TB and `$0.05/GB/month` beyond 1 TB, are available for Secure Cloud pods, replace the default pod volume at `/workspace`, and must be attached during pod deployment rather than after creation. The same docs warn that a volume constrains deployments to its datacenter unless multiple datacenter-specific volumes are created and manually synchronized. Sources: [RunPod network volumes](https://docs.runpod.io/storage/network-volumes) and [RunPod pod pricing](https://docs.runpod.io/pods/pricing).

Use this decision rule:

| Situation | Storage choice | Rationale |
|---|---|---|
| First bounded CP1 smoke, no prior paid failure in the same session | Ephemeral `/workspace`; no network volume | Avoids leaving billable resources behind for a one-shot load/generation check. |
| CP1 retry after a paid setup/cold-start failure | Create a 200 GB Secure Cloud network volume in the best H100 region, usually the current stock winner | Roughly `$14/month` at current pricing is cheaper than repeated H100 cold-start/debug minutes if more than one retry is plausible. |
| CP2+ splits/activations/classifier work | Network volume by default | HF weights, `UV_CACHE_DIR`, repo checkout, `.venv`, datasets, artifacts, and resumable run state matter more than region flexibility. |
| Multi-region fallback remains critical | Prefer ephemeral or create one volume per target datacenter and sync explicitly | A single volume improves reuse but reduces failover because pods must deploy in the volume's datacenter. |
| Dependency stack is stable and will be reused | Bake a custom image/template or persist `.venv` on the volume | The CUDA base image does not by itself prevent `uv sync` from building a fresh project environment. |

Prewarm a network volume with:

- `/workspace/hf` for `HF_HOME` and Mistral weights.
- `/workspace/uv-cache` for `UV_CACHE_DIR`.
- `/workspace/02-h-neurons` as a committed-state checkout or bundle clone.
- `/workspace/02-h-neurons/.venv` only after the dependency set is locally proven.
- Run artifacts under `data/mistral24b/`, with critical evidence still synced back locally before deleting compute.

Do not create or retain a network volume from a side conversation while another paid run is active. Creation is an external billable mutation and region choice affects future pod placement.

## 4. Future CP1 Workflow

Run these checks before creating any paid pod:

```bash
git status --short --branch
uv sync --no-dev --dry-run
uv run python -m scripts.lib.pipeline active-run-status
uv run pytest tests/test_model_load_smoke.py tests/test_run_negative_control.py -q
ruff check scripts/model_load_smoke.py tests/test_model_load_smoke.py tests/test_run_negative_control.py
shellcheck scripts/infra/mistral24b_replication.sh
ty check scripts/model_load_smoke.py tests/test_model_load_smoke.py tests/test_run_negative_control.py
git diff --check
git bundle create /tmp/h-neurons-cp1.bundle main
rm -rf /tmp/h-neurons-bundle-smoke
git clone -b main /tmp/h-neurons-cp1.bundle /tmp/h-neurons-bundle-smoke
```

Before the run, make one explicit decision about the CP1 acceptance gate:

- If the run uses `DEVICE_MAP=cuda:0`, the acceptance predicate must allow `runtime.hf_device_map == null` when `runtime.requested_device_map=="cuda:0"`, `runtime.loaded_primary_device=="cuda:0"`, CUDA allocation is nonzero, BF16 dtype loaded, no quantization is present, and generation succeeds.
- If the acceptance predicate must require `runtime.hf_device_map != null`, then run with `DEVICE_MAP=auto` or update `scripts/model_load_smoke.py` to record an effective explicit-device map.

Before creating a pod, make one explicit decision about storage:

- First one-shot CP1 attempt: ephemeral volume is acceptable.
- Second CP1 attempt or CP2+: use a network volume or custom image/template path.
- If using a network volume, create it in the chosen Secure Cloud datacenter before pod deployment and pass it at creation time; do not expect to attach it later.

After pod creation, do this before remote setup:

```bash
runpodctl pod list
runpodctl network-volume list
runpodctl user | jq '{currentSpendPerHr}'

# Source of truth for SSH is pod metadata, not the relay alias.
runpodctl pod get "$POD_ID"
# Extract the direct SSH host/port, then prove it:
ssh -o StrictHostKeyChecking=accept-new -p "$SSH_PORT" root@"$SSH_HOST" true
```

On the pod, assert the hardware before the wrapper:

```bash
uv run python - <<'PY'
import torch
assert torch.cuda.is_available()
name = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_memory
assert "H100" in name, name
assert mem >= 75 * 2**30, mem
assert torch.cuda.is_bf16_supported()
print(name, mem)
PY
```

Run only the smoke gate:

```bash
timeout --foreground 90m env TMUX_WRAPPED=1 INHIBIT_WRAPPED=1 STAGES=model_smoke \
  bash scripts/infra/mistral24b_replication.sh
```

Before deletion, validate and tar only the intended evidence. After deletion, verify cleanup:

```bash
runpodctl pod delete "$POD_ID"
runpodctl pod list
runpodctl network-volume list
runpodctl user | jq '{currentSpendPerHr}'
```

## 5. Do Not Repeat

- Do not use ephemeral-only storage for repeated CP1 attempts or CP2+ work.
- Do not leave network volumes around accidentally; check `runpodctl network-volume list` and delete intentional scratch volumes when the reuse window closes.
- Do not assume a ready-made CUDA image means the repo will skip project venv and CUDA/PyTorch dependency setup.
- Do not use paid H100 time to discover `uv` resolver failures.
- Do not assume `ssh "$POD_ID@ssh.runpod.io"` works.
- Do not clone a bundle without `-b main`.
- Do not call CP1 done while the exact acceptance predicate still fails.
- Do not stage run outputs before `uv run python -m scripts.lib.pipeline active-run-status`.
- Do not let an ignored `logs/` path hide the wrapper log from the evidence commit.
- Do not run splits, activations, APIs, or claim-bearing stages until CP1 is accepted.
