# SAE Decomposition Investigation for H-Neurons

## Status Tracker

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0: Write plan | **DONE** | This file |
| Phase 1: Feasibility spike | **CPU REVALIDATED, GPU HOOK FIXED** | `scripts/spike_sae_feasibility.py` now uses the pinned `sae-lens` API and validates `post_feedforward_layernorm`, not `down_proj`. |
| Phase 2: SAE feature extraction | **SCRIPT READY** | `scripts/extract_sae_activations.py` mirrors CETT `--locations`, writes `output_root/<location>/act_<qid>.npy`, and stores root `metadata.json`. |
| Phase 3: SAE probe training | **SCRIPT READY** | `scripts/classifier_sae.py` now validates extraction metadata before training/evaluation and records it in the summary JSON. |
| Phase 4: Interpretability analysis | PENDING | Blocked on Phase 3 |
| Phase 5: SAE-based steering | **HELPER READY** | `scripts/intervene_sae.py` now preserves the exact activation at `alpha=1.0`; benchmark runs remain gated on Phase 3 results. |

### Phase 1 Findings (CPU spike)
- **Revalidated on 2026-03-19:** CPU smoke run succeeded against the real Gemma Scope artifact after the API/hook fixes.
- **SAE type:** JumpReLUSAE (JumpReLU activation, not standard ReLU)
- **SAE dimensions:** d_in=2560 (hidden_size), d_sae=16384 (16k features)
- **Available widths:** 16k, 262k (with l0_small and l0_big variants)
- **Hook point correction:** SAE is trained on `post_feedforward_layernorm` output, NOT raw `down_proj` output
  - Gemma 3 architecture: `residual → pre_ff_norm → MLP → post_ff_norm → + residual`
  - SAE hooks at post_ff_norm output (after MLP, after norm, before residual add)
  - Our CETT hooks capture down_proj input/output (inside MLP, before post_ff_norm)
  - Feasibility spike and extraction script both validate the correct point (`post_feedforward_layernorm`)
- **Dependency:** `sae-lens==6.39.0` installed cleanly, no conflicts
- **Release name:** `gemma-scope-2-4b-it-mlp-all` (covers all 34 layers)
- **SAE ID format:** `layer_{N}_width_{16k|262k}_l0_{small|big}`

---

## Context

External feedback proposes Sparse Autoencoder (SAE) feature decomposition as the highest-value extension to our H-neuron work. The claim: the 38 neurons are "almost certainly polysemantic," and Gemma Scope SAEs can isolate a "pure truth" feature for zero-degradation steering. Before investing weeks of engineering, we need to validate or falsify these claims empirically. The feedback makes several assumptions that contradict our existing evidence (e.g., verbosity confound — but H-neuron amplification actually SHORTENS responses). This plan designs experiments to resolve these questions through engineering, not argumentation.

### What the feedback got right
- SAE decomposition is a legitimate interpretability direction
- Gemma Scope 2 covers `gemma-3-4b-it` (confirmed: HF `google/gemma-scope-2-4b-it`, 16k/65k/262k widths, all layers, MLP output + attention + residual stream)
- L13:N833 is the best single-neuron predictor (AUC=0.703) — correctly identified

### What the feedback missed or got wrong
1. **Verbosity confound is backwards.** H-neuron amplification shortens responses (FalseQA: -9%, BioASQ: 41→27 chars). If these encoded "verbose falsehood," amplification should lengthen, not shorten.
2. **Ignored negative controls entirely.** H-neuron slope 2.09 pp/α vs random 0.02 pp/α (empirical interval [-0.11, 0.16]) already proves specificity. SAEs aren't needed for this.
3. **Ignored swing characterization.** Disjoint swing populations + AUROC=0.517 from structural features suggest a unitary compliance axis, not polysemantic mix.
4. **"Zero-degradation steering" is unsupported.** SAE features are aspirationally monosemantic, not guaranteed. Reconstruction error means clamping still has off-target effects.
5. **Existing activations can't be reused for SAE projection.** Current `.npy` files are CETT-normalized (|z| * ||W_col|| / ||h||). SAE needs raw MLP outputs. New extraction required.
6. **Doesn't distinguish CETT features from raw activations.** The L1 classifier trains on normalized contribution scores, not raw neuron activations. SAE features are a different representation entirely — not a drop-in replacement.

### Critical technical gap the feedback didn't mention
The current `CETTManager` hooks capture `input[0]` to `down_proj` (intermediate activation z_t) and only the NORM of the output (||h_t||). For SAE projection, we need the full `down_proj` output tensor h_t. This requires modifying the extraction pipeline.

---

## Experimental Design

### Phase 1: Feasibility Spike (CPU/light GPU, ~4 hours)

**Goal:** Confirm Gemma Scope 2 loads, verify dimensions match, design the extraction architecture.

**Steps:**

1. **Load and inspect Gemma Scope SAE for layer 13** (where L13:N833, our best single predictor, lives)
   - Use SAELens to load `google/gemma-scope-2-4b-it` layer-13 MLP-output SAE (16k width)
   - Verify: input dimension matches Gemma-3-4B MLP output dimension (should be model hidden_size, not intermediate_size — check this)
   - Record: SAE encoder/decoder shapes, activation function, feature count

2. **Verify hook point compatibility** *(RESOLVED on CPU, corrected for GPU rerun)*
   - Gemma Scope 2 SAEs are trained on `post_feedforward_layernorm` output (AFTER MLP + norm, BEFORE residual addition)
   - This is NOT the raw `down_proj` output — there's a post-feedforward RMSNorm between them
   - Feasibility and extraction scripts both hook `post_feedforward_layernorm`
   - The corrected GPU hook test should only treat that tensor as the success criterion

3. **Chosen extraction architecture**
   - Load model + SAEs together and project `post_feedforward_layernorm` activations through the SAE encoder in the extraction pass
   - Keep token-region selection shared with CETT extraction so `answer_tokens` and `all_except_answer_tokens` stay definitionally aligned across probe families

**Output:** Feasibility confirmed or blocked. If blocked (SAE dimensions mismatch, VRAM doesn't fit, etc.), document why and stop.

**Critical file:** `scripts/extract_activations.py` line 56–108 (`CETTManager` class)

### Phase 2: SAE Feature Extraction (GPU, ~2-4 hours)

**Goal:** Extract SAE features for the same TriviaQA train/test samples used for the CETT classifier.

**Steps:**

1. **Use `scripts/extract_sae_activations.py`**
   - Reuses the sample-loading, tokenization, and region-selection logic from `extract_activations.py`
   - Hooks at `post_feedforward_layernorm` output (where Gemma Scope SAEs are trained)
   - Projects through SAE encoder: `f = SAE.encode(h_t)` → sparse feature vector per token
   - Aggregate over requested token locations: mean across token positions by default (same as CETT pipeline)
   - Save per-sample `.npy` files under `data/gemma3_4b/pipeline/activations_sae/<location>/`
   - Shape per file: `[num_sae_layers, sae_width]` (e.g., `[34, 16384]` for 16k SAE at all layers, or just key layers)
   - Write `metadata.json` at the extraction root so classifier training and steering validation can verify the exact SAE basis used

2. **Layer selection strategy:**
   - Start with layers where H-neurons concentrate: 0, 5, 6, 7, 13, 14, 15, 16, 17, 20
   - If feasible, extract all 34 layers
   - If VRAM constrained: extract layer-by-layer (load one SAE at a time, still ~130 MB per SAE)

3. **Use the same sample IDs** as the CETT classifier: `data/gemma3_4b/pipeline/train_qids.json` + `data/gemma3_4b/pipeline/test_qids_disjoint.json`

**Output:** SAE feature `.npy` files for all train + test samples. Directory: `data/gemma3_4b/pipeline/activations_sae/`

### Phase 3: SAE Probe Training (CPU, ~1 hour)

**Goal:** Train an L1 logistic regression on SAE features and compare to the CETT-based probe.

**Steps:**

1. **Use `scripts/classifier_sae.py`**
   - Load SAE feature `.npy` files
   - Flatten to `[samples, layers × sae_width]` feature matrix
   - Train L1 logistic regression with C sweep (same C values as CETT classifier)
   - Evaluate on disjoint test set
   - Report: AUROC, accuracy, precision, recall, F1 with bootstrap 95% CIs
   - Record extraction metadata (hook point, layer order, SAE width, aggregation method, source locations) in the summary JSON

2. **Key comparisons:**

   | Metric | CETT Probe (baseline) | SAE Probe (new) |
   |--------|----------------------|-----------------|
   | AUROC  | 0.843 [0.815, 0.870] | ? |
   | Accuracy | 76.5% [73.6, 79.5] | ? |
   | Non-zero features | 38 neurons (0.011‰) | ? SAE features |

3. **Train on 3-vs-1 mode** (same as CETT): false answer-token SAE features = label 1; all else = label 0

4. **Also train 1-vs-1 mode** for comparison: false answer-token SAE features vs true answer-token SAE features only

**Output:** `data/gemma3_4b/pipeline/classifier_sae_summary.json` with metrics, feature indices, weights.

### Phase 4: Interpretability & Verbosity Decorrelation (CPU, ~2 hours)

**Goal:** Test the feedback's verbosity confound claim empirically at the SAE feature level.

**Steps:**

1. **For each SAE feature with positive classifier weight:**
   - Compute Pearson correlation between feature activation and response length
   - Compute correlation with question length, context length (from swing characterization data)
   - If |r| > 0.3 for length: flag as potential verbosity confound

2. **Max-activating examples analysis:**
   - For top 5 SAE features by classifier weight, find the 10 samples with highest activation
   - Manually inspect: do they share semantic content (compliance-related) or surface form (length-related)?

3. **Cross-benchmark activation check:**
   - Run a small batch of FaithEval and FalseQA samples through the model + SAE
   - Check whether the same top SAE features activate on compliance-relevant samples across benchmarks
   - This is the SAE-level analog of the cross-benchmark consistency finding

**Output:** `data/gemma3_4b/pipeline/sae_feature_analysis.json` with per-feature correlations and max-activating sample IDs.

### Phase 5: SAE-Based Steering Experiment (GPU, ~4-8 hours)

**Goal:** Test whether SAE-based steering produces the same compliance effect with fewer side effects.

**Only proceed if Phase 3 shows the SAE probe has comparable or better AUROC.**

**Steps:**

1. **Use `scripts/intervene_sae.py`**
   - `SAEFeatureScaler`:
     - Hooks at `post_feedforward_layernorm` output (where SAEs are trained)
     - Encodes through SAE: `f = SAE.encode(h_t)`
     - Scales target features: `f[target_indices] *= alpha`
     - Decodes back: `h_t_new = SAE.decode(f)`
     - Replaces original h_t with h_t_new in residual stream
     - Returns the original activation unchanged when `alpha=1.0`, so the control point is an exact no-op
   - This is architecturally different from neuron scaling — it operates in SAE feature space

2. **Run FaithEval anti-compliance sweep** (same α range: 0.0 to 3.0)
   - Compare against existing neuron-scaling baseline
   - Measure: compliance rate, response length, parse failures at each α

3. **Run FaithEval negative control** (random SAE features, 3 seeds minimum)
   - Pick same number of random SAE features as H-features identified in Phase 3
   - Same α sweep
   - Compare H-feature slope vs random-feature slope (same analysis as existing negative control)

4. **Key success/failure criteria:**

   | Outcome | Interpretation |
   |---------|---------------|
   | SAE steering matches neuron slope, fewer side effects | SAE refines the mechanism — genuine progress |
   | SAE steering matches neuron slope, same side effects | SAE adds complexity without benefit — not worth it |
   | SAE steering weaker than neuron slope | SAE loses signal in encode/decode — neuron level is better for this task |
   | SAE steering stronger than neuron slope | SAE concentrates the signal — strong publishable result |

**Output:** `data/gemma3_4b/intervention/faitheval_sae/experiment/` with per-α JSONL files and results.json.

---

## Dependencies & Infrastructure

**Pinned dependency:** `sae-lens>=6.39.0` in `pyproject.toml`
- Pulls in: `transformer-lens`, `einops`, potentially `circuitsvis`
- May conflict with existing bare `torch`/`transformers` if versions diverge — test in Phase 1

**Current scripts:**
- `scripts/extract_sae_activations.py` — Phase 2
- `scripts/classifier_sae.py` (or flag on `classifier.py`) — Phase 3
- `scripts/intervene_sae.py` (or class in `intervene_model.py`) — Phase 5

**Data directories:**
- `data/gemma3_4b/pipeline/activations_sae/` — SAE feature files (gitignored, like current activations)
- `data/gemma3_4b/pipeline/classifier_sae_summary.json` — probe comparison (committed)
- `data/gemma3_4b/intervention/faitheval_sae/` — steering results (committed)

**Hardware:**
- Phase 1: CPU + brief GPU check (minutes)
- Phase 2: GPU required (~2-4 hrs, Gemma-3-4B + SAE fit on 16GB 5060 Ti)
- Phase 3-4: CPU only
- Phase 5: GPU required (~4-8 hrs for full sweep)

---

## Go/No-Go Gates

**Gate 1 (end of Phase 1):** Does the SAE load and are dimensions compatible?
- If NO → stop, document incompatibility, SAE direction is blocked for this model

**Gate 2 (end of Phase 3):** Does the SAE probe match CETT probe performance?
- If SAE AUROC < 0.75 (significantly below 0.843 baseline) → stop, SAE loses signal. Document as negative result.
- If SAE AUROC ≥ 0.80 → proceed to Phase 5 (steering)
- If SAE AUROC is 0.75–0.80 → proceed with caution, may still be interesting for interpretability even if detection is weaker

**Gate 3 (end of Phase 5):** Does SAE steering improve over neuron steering?
- Compare compliance slopes AND side effects (length, parse failures)
- Document either outcome as a finding — both are publishable

---

## Verification

1. After Phase 1: confirm SAE encoder input dim == `post_feedforward_layernorm` output dim (Gemma 3 hidden size), not the `down_proj` input dimension
2. After Phase 2: spot-check 5 SAE feature files for correct shape and no NaN/Inf
3. After Phase 3: run `uv run python scripts/audit_ci_coverage.py` to register new claims
4. After Phase 5: verify negative control SAE features don't overlap with H-features
5. All phases: `ruff check scripts && ruff format scripts` before committing

---

## What This Resolves

By the end of this investigation, we'll have empirical answers to:
1. **Does SAE decomposition improve hallucination detection?** (Phase 3 AUROC comparison)
2. **Are the compliance-predictive features monosemantic?** (Phase 4 interpretability analysis)
3. **Does the verbosity confound exist at feature level?** (Phase 4 correlation analysis)
4. **Does SAE steering reduce side effects?** (Phase 5 compliance + length comparison)
5. **Is the SAE direction worth pursuing further?** (Go/no-go gates)

Either outcome advances the research: positive results bridge H-neurons to the SAE interpretability paradigm; negative results demonstrate that neuron-level analysis captures something SAEs don't easily decompose (which would itself be a novel finding about the nature of compliance circuits).
