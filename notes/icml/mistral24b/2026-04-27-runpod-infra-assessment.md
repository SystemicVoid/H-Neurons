# Mistral-24B Anchor-Replication Infra Assessment

> Superseded for Mistral execution planning and progress tracking by
> `notes/icml/mistral24b/2026-04-28-5.5-pro-l1-mitigation-strategy.md`.
> Retained for historical infra context only.

**Date:** 2026-04-27
**Scope:** RunPod compute selection for the Mistral-Small-24B narrow-extension anchor
([scope memo](../../../paper/icml/reviews/2026-04-21-scope-reassessment-gemma-flagship-mistral-anchor.md)).
Separate from the main Gemma flagship infra; do not generalize the choices here to Gemma runs.

**Decision in one line:**
**H100 SXM 80 GB Pod on RunPod Secure Cloud, Pytorch 2.4 / CUDA 12.4 template, with storage chosen by run stage.**
Fall back to A100 SXM 80 GB if availability or budget pinches.

---

## 1. Workload definition

The Mistral 24B anchor work is *replication*, not novel research:

- Forward passes over a fixed prompt set (TriviaQA + canary; see `data/mistral24b/`).
- Per-layer activation capture for h-neuron probe rebuilding
  (`models/mistral24b_classifier_rebuilt.pkl` indicates this is already a partial pipeline).
- Possible causal interventions (patching/ablation) at single layers — not multi-GPU TP.
- Evaluator pass via external LLM judge ([`logs/mistral24b_step2_llm.log`](../../../logs/mistral24b_step2_llm.log)).

→ Stateful, custom hook code, deterministic seeds. **Pods, not Serverless.**

### Model facts

`mistralai/Mistral-Small-24B-Base-2501` and `-Instruct-2501`:
24B params, 40 layers, hidden 5120, FFN intermediate 32768,
32 attention heads / 8 KV heads, 32k context, BF16 native
([HF config](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501/blob/main/config.json)).

`mistralai/Mistral-Small-3.1-24B-Instruct-2503` keeps the same text stack shape
but is `model_type=mistral3` with `Mistral3ForConditionalGeneration` and a
128k context. It is a deferred third-checkpoint extension, not the canonical
text-only 2501 run.

### Memory budget (BF16, no quant)

| Component | Size |
|---|---|
| Weights | 48.0 GB |
| KV cache, B=8, S=2048 | ~2.7 GB |
| Residual cache, all 40 layers, B=8, S=2048 | ~6.7 GB |
| CETT hook tensors / attn / working buffers | ~8-20 GB |
| **Peak** | **~65-78 GB** |

→ 80 GB is feasible with disciplined batching, 96 GB is comfortable, **48 GB is non-viable for replication** (quantization perturbs the very activations being studied).

### Throughput regime

Mech-interp at batch ≤ 32 is **memory-bandwidth bound**, not compute bound.
Cost-per-result is governed by `$/hr ÷ aggregate HBM bandwidth`, not by FLOPS or FP4 marketing numbers.

---

## 2. Live RunPod state (Secure Cloud, captured 2026-04-27)

Pulled from [console.runpod.io/deploy](https://console.runpod.io/deploy);
catalog at [runpod.io/pricing](https://www.runpod.io/pricing).

| GPU | VRAM | $/hr | HBM BW | Avail | Verdict |
|---|---|---|---|---|---|
| **H100 SXM** | 80 GB | **$2.99** | 3.35 TB/s | **High** | **Primary pick** |
| A100 SXM | 80 GB | **$1.49** | 2.04 TB/s | Medium | **Cheapest viable** |
| RTX PRO 6000 Blackwell | 96 GB | **$1.89** | 1.79 TB/s | High | Headroom fallback only |
| H200 SXM | 141 GB | $3.99 | 4.8 TB/s | Low | Overkill |
| H100 NVL | 94 GB | $3.07 | 3.9 TB/s | Low | Worse than H100 SXM |
| H200 NVL | 143 GB | $3.39 | 4.8 TB/s | Low | Overkill |
| H100 PCIe | 80 GB | $2.39 | 2.0 TB/s | **Unavailable** | N/A |
| A100 PCIe | 80 GB | $1.39 | 1.94 TB/s | Low | Avoid (avail) |
| B200 / B300 | 180/288 GB | $5.49 / $7.39 | 8 TB/s | **Unavailable** | N/A |

Bandwidth sources: [H100 datasheet](https://www.nvidia.com/en-us/data-center/h100/),
[A100 datasheet](https://www.nvidia.com/en-us/data-center/a100/),
[H200 datasheet](https://www.nvidia.com/en-us/data-center/h200/),
[RTX PRO 6000 Blackwell product page](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-blackwell-workstation/).

### Cost-per-result (relative to A100 SXM = 1.0)

Single-token decode rate proxy = `2 × N_params / HBM_BW`. Cost-per-result = `$/hr ÷ relative speed`.

| GPU | $/hr | Rel. speed | $/throughput-unit |
|---|---|---|---|
| A100 SXM | $1.49 | 1.00× | **$1.49** |
| H200 SXM | $3.99 | 2.35× | $1.70 |
| H100 SXM | $2.99 | 1.65× | **$1.81** |
| H100 NVL | $3.07 | 1.91× | $1.61 |
| RTX PRO 6000 Blk | $1.89 | 0.88× | $2.15 |

Two non-obvious observations:

1. **A100 SXM is best $/result** at this catalog. H100 SXM costs ~22% more per unit work, you pay for wall-clock.
2. **RTX PRO 6000 Blackwell is the *worst* of the viable picks per dollar** despite the $1.89 sticker. Its only reason-for-being is the 96 GB VRAM headroom; speed-per-dollar is dominated by both A100 and H100.

---

## 3. Why not Serverless / Hub / Clusters

- **Serverless** (vLLM/SGLang workers, [docs](https://docs.runpod.io/serverless/overview)):
  built for OpenAI-shaped inference. Cold starts, ephemeral storage, fixed handler contract.
  You cannot register hooks across 40 layers, dump residuals to disk,
  or do activation patching without fighting the platform. Skip.
- **Public Endpoints** ([Hub](https://console.runpod.io/hub)): no Mistral 24B endpoint, and no hookability anyway.
- **Instant Clusters** (multi-GPU TP): 24B fits on one 80 GB GPU. Tensor parallelism adds NCCL/PCIe nondeterminism, which is hostile to a *replication* effort.
- **AMD MI300X**: $1.99/hr, 192 GB. Skip — `transformer_lens` hooks and FlashAttention paths are CUDA-first; ROCm parity is not worth the risk on an anchor run.

---

## 4. Stack

### Pod template

[Hub > Pod templates](https://console.runpod.io/hub?tabSelected=templates):

- **Hopper (A100/H100/H200):** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
  → Torch 2.4, CUDA 12.4. Battle-tested for FlashAttention-2/3, `transformer_lens`, vLLM.
- **Blackwell (RTX PRO 6000, B200):** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
  → Torch 2.8, CUDA 12.8. Required for `sm_120`. Use only if forced onto Blackwell.

### Python deps (one-shot, installed onto selected workspace)

```bash
uv sync --no-dev
uv run python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

Use `uv add <package>` plus a same-change `uv export --no-hashes --frozen
--no-emit-project > requirements.txt` only if a dependency is genuinely missing.
Do not use `pip install` instructions in repo docs.
Use `uv sync --frozen` only if a `uv.lock` is intentionally committed.

- [transformer_lens](https://github.com/TransformerLensOrg/TransformerLens) supports the Mistral arch.
- [nnsight](https://nnsight.net/) for declarative tracing if direct PyTorch hooks are too rigid for the intervention pattern.
- [FlashAttention](https://github.com/Dao-AILab/flash-attention): FA2 is fine on Hopper; FA3 wheels exist for H100 SXM.
- [vLLM](https://github.com/vllm-project/vllm): only for canary judge/inference checks, not the hook pipeline.

### Storage plan

[Storage docs](https://docs.runpod.io/pods/storage/types):

- For a first bounded CP1 smoke, ephemeral `/workspace` is acceptable and avoids an
  ongoing storage resource.
- For repeated CP1 attempts or CP2+ work, use a Network Volume in the chosen Secure
  Cloud region. Current docs price it at `$0.07/GB-month` under 1 TB, so 200 GB is
  about `$14/month`.
- Region-lock tradeoff: a single volume preserves caches but constrains pod placement
  to that datacenter. Create per-region volumes only if fallback reliability justifies
  manual synchronization.
- Layout: `hf/` for model cache, `uv-cache/`, `02-h-neurons/`, `.venv` only after
  local dependency proof, and run artifacts under `data/mistral24b/`.

### Region

Pick once, commit. EU-RO-1 or US-OR-1 typically carry both H100 SXM and A100 SXM.
Verify availability in [console at deploy time](https://console.runpod.io/deploy) before creating the volume.

---

## 5. Numerical-reproducibility notes (anchor-specific)

Replicating an anchor result means matching numbers, not just trends. Hardware decisions interact with this:

- **BF16 across NVIDIA generations** is bit-stable for elementwise ops but kernel selection (FA2 vs FA3, cuBLAS LT version) can move logits at ULP scale. Empirically irrelevant for downstream metrics, but worth noting if any anchor metric is decimal-sensitive.
- **A100 SXM is the most likely match** to the original Mistral-24B mech-interp work in the literature; H100 is also common. Blackwell is essentially absent from current published interp pipelines, which is the second reason to deprioritize RTX PRO 6000 here.
- Set `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, fix all seeds.
- Pin `transformers` version in `pyproject.toml` change; record `pip freeze` into the run sidecar (existing pipeline guard convention — see `scripts/lib/AGENTS.md`).

---

## 6. Decision matrix

| Constraint | Pick | Why |
|---|---|---|
| Default | **H100 SXM 80 GB @ $2.99/hr** | Fastest wall-clock, High avail, Hopper = canonical mech-interp target |
| Invoice-sensitive (BlueDot reimbursement) | **A100 SXM 80 GB @ $1.49/hr** | Best $/result, matches paper-era hardware, 50% cheaper |
| 80 GB peaks during all-layer capture | **RTX PRO 6000 96 GB @ $1.89/hr** | Only reason: VRAM headroom. Accept ~1.7× slowdown vs H100 SXM. |
| Need wall-clock fast | H200 SXM if available | 2.35× A100 throughput; expensive |
| Anything else | Don't deviate |

### Expected total cost for one full anchor run (~10 GPU-hours)

| Plan | GPU $ | Storage (prorated) | Total |
|---|---|---|---|
| A100 SXM × 10 h | $14.90 | ~$0.50 | **~$15** |
| H100 SXM × 10 h | $29.90 | ~$0.50 | **~$30** |
| RTX PRO 6000 × 11.4 h | $21.55 | ~$0.50 | ~$22 |

All paid via card → clean RunPod invoice → reimbursable.
Lambda hackathon credits stay reserved (see [billing rationale](#7-billing-and-credit-strategy)).

---

## 7. Billing and credit strategy

Lambda promotional credits auto-apply and produce a **$0 invoice** that BlueDot cannot reimburse.
RunPod card-pay → itemized PDF invoice per pod → reimbursable.

→ **Run the BlueDot-funded Mistral anchor on RunPod. Reserve Lambda credits for unfunded follow-ups.**

---

## 8. Setup status (2026-04-27)

Browser-only console work performed via Claude in Chrome
([console settings](https://console.runpod.io/user/settings)); everything below this
section can now be driven from the CLI on this machine.

### Done — credentials and tooling

- **API key** `cli-mech-interp-lab-2026-04-27`, scope **Read & Write** (full
  `api.runpod.io/graphql` access). Stored host-local at:
  - `~/.config/runpod/credentials` — `RUNPOD_API_KEY=…`, `chmod 600`. Source via
    `set -a; source ~/.config/runpod/credentials; set +a` or systemd `EnvironmentFile=`.
  - `~/.runpod/config.toml` — `apikey`/`apiurl` for `runpodctl` (already populated, no
    `runpodctl config --apiKey` step needed).
  - Project `.env` — `RUNPOD_API_KEY` appended for parity with other provider keys.
- **`runpodctl` v2.1.9** installed to `~/.local/bin/runpodctl` (binary verified via
  upstream SHA256). Smoke-tested: `runpodctl me`, `runpodctl pod list`,
  `runpodctl datacenter list` all return live data.
- **Account SSH key** registered: `~/.ssh/id_ed25519.pub`. RunPod injects it into
  every new pod's `authorized_keys` automatically — no per-pod copy needed.
- **Local SSH config** has a `Host ssh.runpod.io` block (User `root`, IdentityFile
  `~/.ssh/id_ed25519`, `IdentitiesOnly yes`, `StrictHostKeyChecking accept-new`,
  keepalives 30 s × 5), but the 2026-04-29 CP1 smoke showed the relay form can
  fail with `Permission denied (publickey)`. Treat `runpodctl pod get <pod-id>` as
  the source of truth for the direct `root@<ip> -p <port>` SSH endpoint; see
  [2026-04-29-cp1-runpod-postmortem.md](2026-04-29-cp1-runpod-postmortem.md).
- **Hugging Face token** `1-gen-vid` (fine-grained, `canReadGatedRepos: true`,
  identity `SystemicVoid`) added to project `.env` as `HF_TOKEN`. Bonus discovery:
  **`mistralai/Mistral-Small-24B-Base-2501` is no longer gated** as of this date
  (HF API `gated: False`, anonymous `config.json` returns HTTP 200) — the token is
  belt-and-suspenders, not strictly required for the canonical Mistral-24B run.
- **Account verified** via `myself` — `clientBalance: $200.00`,
  `currentSpendPerHr: $0`, GitHub linked as `SystemicVoid`, no pre-existing pods or
  network volumes. Account email is `hugo@hugonguyen.com` (separate from the local
  git identity `nguyenhugo8@gmail.com`). Account-level `spendLimit: $80/hr`
  (RunPod default) — well above the $2.99/hr H100 SXM target, no action needed.

### Done — runtime parameters resolved (no commitments made)

- **Template ID locked.** `runpod-torch-v240` →
  `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (official, public).
  Default container disk is only 20 GB, so override to ≥100 GB at launch.
- **Pricing snapshot** (GraphQL `gpuTypes`, $/hr 1 × GPU on-demand):

  | GPU (80 GB) | Secure | Community | Spot floor |
  |---|---|---|---|
  | **H100 SXM** (HBM3) | $2.99 | $2.69 | $1.50 |
  | H100 PCIe | $2.39 | $1.99 | — |
  | **A100 SXM** (HBM2e) | $1.49 | $1.39 | $0.79 |
  | A100 PCIe | $1.39 | $1.19 | — |

- **Datacenter availability snapshot** for the two SXM 80 GB targets
  (`runpodctl datacenter list`, 2026-04-27, "—" = no current stock):

  | DC | Location | H100 SXM | A100 SXM |
  |---|---|---|---|
  | AP-IN-1 | India | Medium | — |
  | EU-FR-1 | France | Low | — |
  | EU-NL-1 | Netherlands | Low | — |
  | EUR-IS-1 | Iceland | — | Low |
  | EUR-IS-3 | Iceland | Low | — |
  | EUR-NO-2 | Norway | Low | — |
  | US-CA-2 | California | Low | Low |
  | US-KS-2 | Kansas | — | Low |
  | US-MD-1 | Maryland | — | Low |
  | US-MO-1 | Missouri | Low | Low |
  | US-NE-1 | Nebraska | Low | — |
  | US-TX-3 | Texas | Low | — |
  | US-WA-1 | Washington | — | Low |

  US-CA-2 and US-MO-1 are the only datacenters with **both** H100 SXM and A100 SXM
  in stock today — preferred picks if region matters less than fallback safety.
  Re-run `runpodctl datacenter list` at launch time; this snapshot decays fast.

### Out of scope — defer to launch time

- **Local adapter.** Use
  [`scripts/infra/cloudctl.py`](../../../scripts/infra/cloudctl.py) and
  [`scripts/infra/cloud/`](../../../scripts/infra/cloud/) for dry-run launch
  rendering, direct SSH derivation, preflight, and cleanup checks.
- **Network Volume.** Current docs list `$0.07/GB-month` under 1 TB and
  `$0.05/GB-month` over 1 TB. Volumes persist independently of compute, mount at
  `/workspace` for Secure Cloud pods, must be attached at pod deployment time, and
  constrain pod placement to the volume datacenter. Use ephemeral storage for a
  first one-shot CP1 smoke; create/prewarm a network volume for a second CP1 paid
  attempt or CP2+ work. See
  [2026-04-29-cp1-runpod-postmortem.md](2026-04-29-cp1-runpod-postmortem.md).
- **Pod launch.** Owned by the agent driving the workload. Reference invocation:
  Before paid CP1-style smoke launches, read the CP1 RunPod postmortem and guardrails:
  [2026-04-29-cp1-runpod-postmortem.md](2026-04-29-cp1-runpod-postmortem.md).

  ```bash
  runpodctl pod create \
    --name mistral24b-anchor \
    --template-id runpod-torch-v240 \
    --gpu-id "NVIDIA H100 80GB HBM3" \
    --gpu-count 1 \
    --cloud-type SECURE \
    --container-disk-in-gb 120 \
    --data-center-ids US-CA-2 \
    --ports "22/tcp,8888/http"
    # add: --network-volume-id <id from `runpodctl network-volume create`>
    #      --volume-mount-path /workspace
    # if reusing weights across pods (else accept ~6 min re-download per cold start)
  ```
  Fallback: swap `--gpu-id "NVIDIA A100-SXM4-80GB"` and re-pick `--data-center-ids`.

### Health checks (re-runnable)

```bash
# Account + key + balance, single round-trip
runpodctl me                                         # via runpodctl
curl -sS -X POST -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d '{"query":"query { myself { id email clientBalance currentSpendPerHr pubKey } }"}' \
  https://api.runpod.io/graphql                      # or via GraphQL

# Live availability for the picked GPU
runpodctl datacenter list | jq '[.[] | {id, location, h100: ([.gpuAvailability[] | select(.gpuId=="NVIDIA H100 80GB HBM3").stockStatus] | first), a100: ([.gpuAvailability[] | select(.gpuId=="NVIDIA A100-SXM4-80GB").stockStatus] | first)}]'

# Mistral weight access (will 401 if HF re-gates the model)
curl -sS -L -H "Authorization: Bearer $HF_TOKEN" -o /dev/null -w "%{http_code}\n" \
  https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501/resolve/main/config.json
```

---

## 9. Launch checklist

1. Verify H100 SXM availability in target region at [console.runpod.io/deploy](https://console.runpod.io/deploy).
2. Choose storage mode: ephemeral for a first one-shot CP1 smoke; otherwise create
   a **Network Volume** in that region and attach it at deploy time.
3. Launch **H100 SXM** Pod, attach the volume if selected, template = **Pytorch 2.4.0**, SSH on.
4. If using persistent storage, install deps once into the volume with `uv sync --no-dev`; download the model with
   `uv run huggingface-cli download mistralai/Mistral-Small-24B-Instruct-2501 --local-dir /workspace/models/Mistral-Small-24B-Instruct-2501`.
5. Run `DRY_RUN=1 bash scripts/infra/mistral24b_replication.sh` to inspect the
   guarded command sequence, then run `bash scripts/infra/mistral24b_replication.sh`
   in tmux or let the wrapper create its own tmux session; dump outputs under
   `data/mistral24b/`. Use `STAGES=<comma-separated stages>` to resume a
   verified subset.
6. Run `scripts/lib/pipeline active-run-status` before staging outputs (project guard, [root AGENTS.md](../../../AGENTS.md)).
7. **Stop the pod** when done — only $7/mo storage carries.
8. If H100 SXM disappears at launch time → fallback A100 SXM, same volume, same template.

---

## 10. Sources

### Live data
- [RunPod console — Deploy a Pod](https://console.runpod.io/deploy) (pricing + availability, snapshot 2026-04-27)
- [RunPod public pricing page](https://www.runpod.io/pricing)
- [RunPod Hub — Pod templates](https://console.runpod.io/hub?tabSelected=templates)

### Platform docs
- [RunPod docs — Pods](https://docs.runpod.io/pods/overview)
- [RunPod docs — Storage types](https://docs.runpod.io/pods/storage/types)
- [RunPod docs — Network volumes](https://docs.runpod.io/pods/storage/create-network-volumes)
- [RunPod docs — Serverless](https://docs.runpod.io/serverless/overview) (for the why-not)

### Hardware
- [NVIDIA H100 product page](https://www.nvidia.com/en-us/data-center/h100/)
- [NVIDIA A100 product page](https://www.nvidia.com/en-us/data-center/a100/)
- [NVIDIA H200 product page](https://www.nvidia.com/en-us/data-center/h200/)
- [NVIDIA RTX PRO 6000 Blackwell Workstation](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-blackwell-workstation/)

### Model + stack
- [Mistral-Small-24B-Instruct-2501 on Hugging Face](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501)
- [Mistral-Small-24B-Instruct-2501 config](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501/blob/main/config.json)
- [Mistral-Small-24B-Base-2501 on Hugging Face](https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501)
- [Mistral-Small-3.1-24B-Instruct-2503 config](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/blob/main/config.json)
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- [nnsight](https://nnsight.net/) · [repo](https://github.com/ndif-team/nnsight)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [vLLM](https://github.com/vllm-project/vllm)

### Project context
- [Anchor scope memo (2026-04-21)](../../../paper/icml/reviews/2026-04-21-scope-reassessment-gemma-flagship-mistral-anchor.md)
- [`data/mistral24b/`](../../../data/mistral24b/) — existing pipeline artifacts
- [`docs/throughput-assessment.md`](../../../docs/throughput-assessment.md) — Gemma-side throughput baseline (do not transplant numbers)
