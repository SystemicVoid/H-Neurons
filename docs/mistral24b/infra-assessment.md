# Mistral-24B Anchor-Replication Infra Assessment

**Date:** 2026-04-27
**Scope:** RunPod compute selection for the Mistral-Small-24B narrow-extension anchor
([scope memo](../../paper/icml/reviews/2026-04-21-scope-reassessment-gemma-flagship-mistral-anchor.md)).
Separate from the main Gemma flagship infra; do not generalize the choices here to Gemma runs.

**Decision in one line:**
**H100 SXM 80 GB Pod on RunPod Secure Cloud, Pytorch 2.4 / CUDA 12.4 template, 100 GB Network Volume.**
Fall back to A100 SXM 80 GB if availability or budget pinches.

---

## 1. Workload definition

The Mistral 24B anchor work is *replication*, not novel research:

- Forward passes over a fixed prompt set (TriviaQA + canary; see `data/mistral24b/`).
- Per-layer activation capture for h-neuron probe rebuilding
  (`models/mistral24b_classifier_rebuilt.pkl` indicates this is already a partial pipeline).
- Possible causal interventions (patching/ablation) at single layers — not multi-GPU TP.
- Evaluator pass via external LLM judge ([`logs/mistral24b_step2_llm.log`](../../logs/mistral24b_step2_llm.log)).

→ Stateful, custom hook code, deterministic seeds. **Pods, not Serverless.**

### Model facts

`mistralai/Mistral-Small-24B-Base-2501` and `-Instruct-2501`:
24B params, 56 layers, hidden 5120, 32 attn heads / 8 KV heads, 32k context,
BF16 native ([HF model card](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501)).

### Memory budget (BF16, no quant)

| Component | Size |
|---|---|
| Weights | 48.0 GB |
| KV cache, B=8, S=2048 | ~12 GB |
| Residual cache, all 56 layers, B=8, S=2048 | ~9.4 GB |
| Attn / working buffers | ~8 GB |
| **Peak** | **~78 GB** |

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
  You cannot register `transformer_lens` hooks across 56 layers, dump residuals to disk,
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

### Python deps (one-shot, installed onto Network Volume)

```bash
pip install -U \
  transformers==4.* accelerate \
  nnsight transformer_lens \
  flash-attn --no-build-isolation \
  vllm
```

- [transformer_lens](https://github.com/TransformerLensOrg/TransformerLens) supports the Mistral arch.
- [nnsight](https://nnsight.net/) for declarative tracing if `transformer_lens` is too rigid for the intervention pattern.
- [FlashAttention](https://github.com/Dao-AILab/flash-attention): FA2 is fine on Hopper; FA3 wheels exist for H100 SXM.
- [vLLM](https://github.com/vllm-project/vllm): only for the canary judge pass, not the hook pipeline.

### Storage plan

[Storage docs](https://docs.runpod.io/pods/storage/types):

- **Network Volume Standard, 100 GB, $0.07/GB-mo = $7.00/mo.**
  Persists across pod stop/start, no idle premium (unlike Volume Disk at $0.20/GB-mo idle).
  Region-locked — pin a region with H100 SXM availability *and* A100 SXM fallback.
- Layout: `models/mistral-24b/` (~48 GB) + `runs/` (~30 GB captures) + headroom.

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

Optional: email BlueDot ahead of time to confirm RunPod is acceptable, or whether they have direct provider credits (some AI-safety programs do).

---

## 8. Launch checklist

1. Verify H100 SXM availability in target region at [console.runpod.io/deploy](https://console.runpod.io/deploy).
2. Create **100 GB Network Volume — Standard** in that region (`/workspace`, $7/mo).
3. Launch **H100 SXM** Pod, attach volume, template = **Pytorch 2.4.0**, SSH on.
4. Install deps once into the volume; `huggingface-cli download mistralai/Mistral-Small-24B-Base-2501 --local-dir /workspace/models/mistral-24b`.
5. Run replication with deterministic seeds; dump to `/workspace/runs/<ISO-date>/`.
6. Run `scripts/lib/pipeline active-run-status` before staging outputs (project guard, [AGENTS.md](../../AGENTS.md)).
7. **Stop the pod** when done — only $7/mo storage carries.
8. If H100 SXM disappears at launch time → fallback A100 SXM, same volume, same template.

---

## 9. Sources

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
- [Mistral-Small-24B-Base-2501 on Hugging Face](https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501)
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- [nnsight](https://nnsight.net/) · [repo](https://github.com/ndif-team/nnsight)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [vLLM](https://github.com/vllm-project/vllm)

### Project context
- [Anchor scope memo (2026-04-21)](../../paper/icml/reviews/2026-04-21-scope-reassessment-gemma-flagship-mistral-anchor.md)
- [`data/mistral24b/`](../../data/mistral24b/) — existing pipeline artifacts
- [`docs/throughput-assessment.md`](../throughput-assessment.md) — Gemma-side throughput baseline (do not transplant numbers)
