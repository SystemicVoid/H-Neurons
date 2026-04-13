# SAE Steering Pipeline Audit: Gemma-3-4B FaithEval

**Date:** 2026-03-21
**Model:** `google/gemma-3-4b-it`
**Related reports:** [intervention_findings.md](../../intervention_findings.md), [sae_investigation_plan.md](../../../../docs/archive/sae_investigation_plan.md), [pipeline_report.md](../../pipeline/pipeline_report.md), [falseqa_negative_control_audit.md](../falseqa/falseqa_negative_control_audit.md), [2026-04-13-faitheval-slope-difference-reporting-audit.md](../../../../notes/act3-reports/2026-04-13-faitheval-slope-difference-reporting-audit.md)

> **Update — 2026-04-13:** the later paired slope-difference reporting audit
> ([2026-04-13-faitheval-slope-difference-reporting-audit.md](../../../../notes/act3-reports/2026-04-13-faitheval-slope-difference-reporting-audit.md))
> confirms the site-facing matched-readout statement:
> neuron-minus-SAE **+1.93 pp/α [0.94, 2.92]** on matched items, and
> neuron-minus-random mean **+2.01 pp/α** across 8 paired control seeds.
> Use that report for the narrow reporting-path claim. This audit still owns the
> broader SAE closure argument, especially the delta-only result ruling out
> reconstruction noise as the sole explanation.

---

## Bottom Line

- **SAE-based steering does not work for this task.** The H-feature compliance slope is 0.16 pp/α (CI [-0.51, 0.84]) — indistinguishable from zero and from random SAE features (0.59 pp/α). Neuron-level steering is ~13× stronger (2.09 pp/α) with a clean monotonic dose-response.
- **The SAE encode/decode lossy cycle is the dominant effect.** At α=1.0 (the true no-op, where target features are multiplied by 1.0 and the hook returns the original activation), all 5 configurations produce exactly 66.0% compliance with byte-identical responses. At every other α, where the SAE encode/decode cycle is applied, compliance jumps to ~72–75% regardless of which features are targeted. This ~8–9pp shift is a property of the lossy SAE reconstruction (relative L2 error = 0.1557), not of targeted feature manipulation.
- **H-features paradoxically perform WORSE than random features at high α.** At α=3.0, H-features yield 69.9% compliance versus 74.6% for random features (−4.7pp). This is the opposite of what specific steering would produce — over-amplification of compliance-relevant features disrupts rather than enhances the behavior.
- **SAE detection is not significantly better than neuron-level detection.** The SAE probe (AUROC 0.848, 266 features) marginally exceeds the CETT probe (AUROC 0.843, 38 neurons) but falls within the CETT bootstrap CI [0.815, 0.870] and uses 7× more features. This is a detection-steering dissociation: features that detect hallucination do not necessarily steer it in SAE space.
- **Data integrity is clean.** All 35 JSONL files pass line count, JSON validity, compliance recount, and ID consistency checks.

---

## 1. Design

### 1.1 Experiment Pipeline

**Scientific question:** Can SAE feature-space steering replicate or improve on the neuron-level compliance effect?

**SAE steering mechanism** (`scripts/intervene_sae.py`, class `SAEFeatureScaler`):
1. Hook at `post_feedforward_layernorm` output (the point where Gemma Scope 2 SAEs are trained)
2. Encode the activation through the SAE: `f = SAE.encode(h_t)`
3. Scale target features: `f[:, :, target_indices] *= alpha`
4. Decode back: `h_t_new = SAE.decode(f)`
5. Replace original activation with `h_t_new`
6. **Critical: at α=1.0, the hook returns the original activation unchanged** (early return before encode/decode)

**Target features:** 266 positive-weight SAE features from the L1 logistic regression probe (`models/sae_detector.pkl`, C=0.005, AUROC=0.848 on disjoint test set).

**Benchmark:** FaithEval anti-compliance prompt (n=1,000), 7 alphas (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0). Greedy decoding, deterministic evaluation via regex letter extraction.

**Negative control:** 3 random SAE feature sets (266 features each, drawn from zero-weight classifier positions), same α sweep.

### 1.2 Provenance

| Run | Script | Git SHA | Start | End | Status |
|-----|--------|---------|-------|-----|--------|
| H-feature sweep | `run_intervention.py --intervention_mode sae` | `fabadda` | 2026-03-21 16:51 UTC | 2026-03-21 17:23 UTC | Completed |
| Negative control (3 seeds) | `run_sae_negative_control.py` | `fabadda` | 2026-03-21 17:25 UTC | 2026-03-21 19:22 UTC | Completed |

Full provenance: `experiment/run_intervention.provenance.json`, `control/run_sae_negative_control.provenance.json`.

### 1.3 SAE Configuration

| Parameter | Value |
|-----------|-------|
| SAE release | `gemma-scope-2-4b-it-mlp-all` |
| SAE width | 16k (16,384 features per layer) |
| SAE L0 target | small |
| Hook point | `post_feedforward_layernorm` |
| d_in | 2,560 (model hidden_size) |
| d_sae | 16,384 |
| Layers | 0, 5, 6, 7, 13, 14, 15, 16, 17, 20 |
| Reconstruction error (relative L2) | 0.1557 |

---

## 2. Data Integrity Verification

| Check | Result |
|-------|--------|
| JSONL line counts | 1,000 per file, all 35 files (5 configs × 7 alphas) |
| JSON validity | All 35,000 records parse cleanly |
| Compliance recount vs results.json | All 7 experiment alphas match |
| α=1.0 compliance (all 5 configs) | 660/1,000 = 66.0% (identical) |
| α=1.0 response identity (all 5 configs) | 1,000/1,000 byte-identical |
| Experiment vs control/h_features (all alphas) | 1,000/1,000 byte-identical at every α |
| H-feature / random seed overlap | Zero (all 3 seeds) |
| Random seed-to-seed overlap | 0–1 features (seed 0 vs seed 2: 1 overlap) |
| H-feature count | 266 features across 10 layers |
| Random feature count per seed | 266 features each |
| Feature pool | `zero_weight_only` (all 3 seeds draw from coef==0 positions) |
| Total checks | **12 passed, 0 failed** |

**Schema note:** The experiment JSONL (from `run_intervention.py`) does not include a `parse_failure` boolean field; parse failures are inferred from `chosen=None`. The negative control JSONL (from `run_sae_negative_control.py`) includes an explicit `parse_failure` field. The `results.json` correctly counts `chosen=None` as parse failures in both cases. This is a field-naming inconsistency between scripts, not a data error.

---

## 3. Results

### 3.1 Compliance Rates

| α | Neuron baseline | SAE H-features | SAE random (mean ± std) |
|---|----------------|----------------|-------------------------|
| 0.0 | 64.2% | 72.3% [69.4, 75.0] | 74.9% ± 0.4 |
| 0.5 | 65.4% | 74.7% [71.9, 77.3] | 74.8% ± 0.4 |
| 1.0 | 66.0% | **66.0%** [63.0, 68.9] | **66.0%** ± 0.0 |
| 1.5 | 67.0% | 75.0% [72.2, 77.6] | 75.0% ± 0.2 |
| 2.0 | 68.2% | 75.1% [72.3, 77.7] | 74.9% ± 0.1 |
| 2.5 | 69.5% | 74.9% [72.1, 77.5] | 74.9% ± 0.1 |
| 3.0 | 70.5% | 69.9% [67.0, 72.7] | 74.6% ± 0.5 |

Wilson 95% CIs shown for SAE H-features. α=1.0 is the true SAE no-op (early return, no encode/decode applied).

### 3.2 Slopes and Effects

| Configuration | Slope (pp/α) | Slope 95% CI | Spearman ρ | Monotonic? |
|---------------|-------------|--------------|------------|------------|
| Neuron baseline | **2.09** | [1.38, 2.83] | 1.0 | Yes |
| SAE H-features | 0.16 | [-0.51, 0.84] | 0.18 | No |
| SAE random seed 0 | 0.64 | — | 0.33 | No |
| SAE random seed 1 | 0.58 | — | 0.27 | No |
| SAE random seed 2 | 0.54 | — | −0.04 | No |
| SAE random mean | **0.59** | [0.54, 0.64] | — | No |

The H-feature slope CI [-0.51, 0.84] contains zero. The H-feature slope (0.16) is LOWER than the random mean slope (0.59). The neuron baseline slope (2.09) is ~13× the H-feature slope and ~3.5× the random slope.

### 3.3 Parse Failures

| α | SAE H-features | SAE random (typical) | Neuron baseline |
|---|---------------|---------------------|----------------|
| 0.0 | 21 (2.1%) | 14–23 (1.4–2.3%) | 0 |
| 1.0 | **0** | **0** | 0 |
| 1.5 | 16 (1.6%) | 14–17 (1.4–1.7%) | 0 |
| 3.0 | 23 (2.3%) | 14–19 (1.4–1.9%) | 0 |

Total SAE parse failures across all alphas: 110 (experiment). Zero for the neuron baseline at all alphas. Parse failures are an artifact of the lossy SAE reconstruction degrading response formatting. The zero parse failures at α=1.0 (where no encode/decode occurs) confirms this.

### 3.4 Per-Item Divergence at α=3.0

H-features vs random seed 0 at α=3.0:

|  | Random compliant | Random non-compliant |
|--|-----------------|---------------------|
| **H compliant** | 652 | 47 |
| **H non-compliant** | 99 | 202 |

Net: H-features produce 52 fewer compliant items than random seed 0 (−5.2pp). At α=0.0 (ablation), H-features produce 26 fewer compliant items (−2.6pp). Both directions are consistent with H-features carrying some compliance-relevant signal that acts asymmetrically in SAE space.

---

## 4. Critical Findings

### Finding 1: The SAE encode/decode lossy cycle dominates over targeted feature manipulation

**Data:** At α=1.0 (true no-op), all 5 configs produce exactly 66.0% compliance with byte-identical responses. At α≠1.0 (where encode/decode is applied), compliance jumps to ~72–75% regardless of which features are targeted. The ~8–9pp boost is feature-independent.

**Interpretation:** The lossy SAE reconstruction (L2 error = 0.1557) is itself an uncontrolled intervention on the model's activations. Every token's activation passes through a 16k-dimensional bottleneck and loses ~15.6% of its information content. This global perturbation swamps any signal from scaling 266 out of 163,840 features. This is a fundamental limitation of the encode-modify-decode steering paradigm when reconstruction error is non-negligible.

**Implication:** Any SAE steering experiment that does not account for reconstruction error is confounded. The appropriate control is not α=0 (which also applies encode/decode) but α=1.0 (which bypasses it entirely). The sae_investigation_plan correctly identified α=1.0 as the no-op, and the code correctly implements the early return.

### Finding 2: H-features paradoxically suppress compliance at high α

**Data:** At α=3.0, H-feature compliance is 69.9% vs random mean 74.6% (−4.7pp). At α=0.0, H-feature compliance is 72.3% vs random mean 74.9% (−2.6pp). The H-feature slope (0.16 pp/α) is lower than the random slope (0.59 pp/α).

**Interpretation:** This is the opposite of what feature-specific steering should produce. If the 266 SAE features captured the compliance mechanism, amplifying them (α>1) should increase compliance more than amplifying random features. Instead, amplifying H-features DECREASES compliance relative to random. Two possible explanations:

1. **Over-amplification disruption.** The H-features do encode some compliance-relevant variance, but tripling their magnitude (α=3.0) pushes the reconstructed activation far from the training distribution, degrading coherence more than boosting the target behavior.
2. **Feature space misalignment.** The compliance signal that operates at the neuron level (38 neurons in `down_proj` input space) does not map cleanly onto SAE feature space (post-feedforward-layernorm output space). The L1 probe finds features that correlate with hallucination labels, but correlation in feature space does not imply causal control through encode/decode.

Both explanations are consistent with the data. Distinguishing them would require a reconstruction-error-corrected steering method (modify only the targeted features, pass through the residual unchanged).

### Finding 3: Detection-steering dissociation

**Data:** SAE probe AUROC (0.848) ≈ CETT probe AUROC (0.843), but SAE steering slope (0.16 pp/α) ≪ neuron steering slope (2.09 pp/α).

**Interpretation:** Features that predict hallucination from activations do not necessarily causally control it when manipulated. This is a well-known issue in mechanistic interpretability: correlation-based feature selection (L1 logistic regression on static activations) identifies what covaries with the behavior, not what drives it. The neuron-level intervention works because `down_proj` input scaling directly modifies the MLP's output contribution to the residual stream. The SAE encode/decode cycle adds a lossy transformation that destroys the fine-grained causal link.

**Efficiency comparison:**

| Probe | Features | AUROC | Steering slope (pp/α) | mAUROC per feature |
|-------|----------|-------|-----------------------|--------------------|
| CETT (neurons) | 38 | 0.843 | 2.09 | 22.2 |
| SAE (3-vs-1) | 266 | 0.848 | 0.16 | 3.2 |
| SAE (1-vs-1) | 267 | 0.849 | — | 3.2 |

The CETT probe achieves comparable detection with 7× fewer features and dramatically better steering. The SAE probe adds complexity without benefit for either task.

### Finding 4: Layer coverage gap is real and partially confounding

**Data:** The SAE extraction covers 10 of 34 layers. Of the 38 CETT H-neurons, 18 (47.4%) are in uncovered layers (2, 4, 9, 10, 12, 23-33). These include 5 of the top-10 H-neurons by classifier weight: L33:N8011 (weight 3.071), L24:N7995 (2.603), L26:N1359 (2.456), L9:N5580 (1.824), L10:N4996 (1.705). The uncovered top-10 neurons carry 29.4% of top-10 weight mass and 31.4% of total positive weight mass.

Despite this, the SAE probe matches CETT AUROC (0.848 vs 0.843), meaning the covered layers contain sufficient *discriminative* information. But detection and steering have different requirements: CETT neuron steering operates in all 34 layers simultaneously.

**Interpretation:** The layer coverage gap is a genuine confound for steering, even though it does not affect detection. The SAE steers in 10 layers while the neuron baseline steers in 23 layers (those containing H-neurons). The missing layers include high-weight neurons. However, this cannot be the primary explanation for the steering failure because (a) the encode/decode reconstruction error affects all 10 covered layers and dominates the signal regardless of feature selection, and (b) H-features perform *worse* than random features within the same 10 layers, which layer coverage does not explain.

### Finding 5: Verbosity confounds are low-weight and not driving the classifier

**Data:** 6 of 266 positive-weight SAE features are flagged as verbosity confounds (|r| > 0.3 with response length). All 6 have weights between 0.0001 and 0.0026 (bottom quartile). The top 10 features by weight are all clean (|r| < 0.1).

**Interpretation:** The verbosity confound concern raised in external feedback is empirically real but marginal. The classifier's predictive power comes from non-confounded features. Removing the 6 confounded features would negligibly change the probe's AUROC.

### Finding 6: Layer 20 concentration parallels the neuron 4288 artifact pattern

**Data:** Layer 20 contributes 93 of 266 positive SAE features (35.0%) and 24.4% of total positive weight. By comparison, only 1 of 38 CETT H-neurons (2.6%) is in layer 20. Random feature sets draw ~27 features per layer (10%) under uniform sampling.

L20 features have *lower* mean weight (0.0019 vs 0.0031 for other layers), *lower* positive-separation rate (45% vs 62%), and near-zero median separation. At the CETT neuron level, L20:N4288 was the top-weight neuron (weight 12.169, 30.7% of top-10 weight) but failed all 6 causal tests and was classified as a regularization artifact.

**Interpretation:** The 3.5x over-representation of layer 20 in the SAE probe echoes the neuron 4288 pattern: L1 regularization concentrating on features in a layer where the signal is noisy rather than causal. The lower separation rates and lower weights for L20 features (despite their numerical dominance) are consistent with the probe spreading residual weight across many marginally predictive features rather than identifying a concentrated causal signal. This means the SAE steering experiment amplified ~93 potentially artifactual features in one layer while only modifying ~27 per layer in random controls -- creating a ~3.4x larger perturbation magnitude in layer 20 that likely explains why H-features perform worse than random at extreme alpha values.

**Connection to C=3.0 CETT detector:** The CETT C=3.0 detector (219 neurons, 80.5% accuracy) represents the "distributed mechanism" alternative where neuron 4288 drops to rank #5 and the signal spreads. The SAE probe at C=0.005 (266 features) is structurally analogous -- both are less sparse, both have higher feature counts, and neither has been used for intervention. The SAE C=0.001 probe (62 features, AUROC 0.830) would be the sparser alternative closer to the 38-neuron CETT pattern, but was not selected for steering because it had lower detection AUROC. Detection-optimal feature selection may not be steering-optimal.

---

## 5. Open Confounds and Untested Alternatives

The negative steering result is the primary finding, but four configuration choices remain untested and could in principle change the outcome. They are listed here in order of expected information gain per compute cost.

### Confound 1: Steering architecture — RESOLVED (2026-03-21)

**Status:** Tested. Delta-only steering produces the same null result as full-replacement.

**What was tested:** A delta-only hook (`h + decode(f_modified) - decode(f_original)`) that cancels reconstruction error exactly. Quick-mode sweep (α ∈ {0.0, 1.0, 3.0}) for H-features (266) and 1 random seed (266 zero-weight features). Data: `data/gemma3_4b/intervention/faitheval_sae_delta/`.

**Results:** Delta-only H-feature slope: 0.12 pp/α. Delta-only random slope: -0.09 pp/α. Both indistinguishable from zero. α=1.0 identity check passed (660/1,000 = 66.0%, byte-identical). Zero parse failures (vs 1.4–2.3% for full-replacement). Neuron baseline slope on the same 3 alphas: 2.12 pp/α.

**Interpretation:** The reconstruction error was a nuisance (causing parse failures and the ~8-9pp compliance shift) but was not the cause of the null SAE steering result. The failure is genuinely about feature-space misalignment. This was the highest-priority confound and the cheapest decisive test; its resolution closes the SAE steering line.

### Confound 2: SAE width and reconstruction error

The 16k-width SAE has L2 error of 0.1557. The 262k-width Gemma Scope variant should have substantially lower reconstruction error. If error drops below ~5%, feature-specific effects might emerge above the noise floor.

**Cost:** Download 262k SAEs (~10 layers), re-extract features (~3h GPU), retrain probe, rerun steering. Total ~8h GPU.

### Confound 3: Feature count / C-sweep selection

We steered with 266 features (C=0.005, best detection AUROC). The sparser C=0.001 probe (62 features, AUROC 0.830) concentrates on fewer, potentially more causal features. A sparser feature set creates a smaller per-layer perturbation and may interact less destructively with the decode cycle.

**Cost:** Retrain steering with existing C=0.001 model checkpoint (if saved) or retrain, then rerun. ~2h GPU.

### Confound 4: Layer coverage

The 10-layer extraction misses 47.4% of CETT H-neurons (31.4% of weight). An all-34-layer extraction would provide complete coverage.

**Cost:** ~6h GPU for extraction + retrain + steer. The lowest-priority confound because it cannot explain why H-features perform worse than random within the same 10 layers.

---

## 6. Should We Investigate SAE Steering Further?

**Recommendation: No. The SAE steering line is closed.**

The delta-only falsification test (Confound 1, §5) was run on 2026-03-21 and confirmed that the steering failure is fundamental feature-space misalignment, not reconstruction noise. Both full-replacement and delta-only architectures produce null H-feature slopes (0.16 and 0.12 pp/α respectively), while the neuron baseline is ~18× stronger (2.12 pp/α on the same alphas).

Remaining open confounds (SAE width, feature count, layer coverage) are lower priority and unlikely to change the conclusion: delta-only steering eliminates the dominant confound and still produces a null result. The SAE probe detects hallucination (AUROC 0.848) but the detected features do not causally control compliance when manipulated in SAE space.

---

## 7. Artifacts

| Artifact | Path | Contents |
|----------|------|----------|
| H-feature sweep | `experiment/` | 7 alpha JSONL files + `results.json` + provenance |
| H-feature control run | `control/h_features/` | Redundant run, byte-identical to experiment (reproducibility confirmation) |
| Random seed 0 | `control/seed_0_random/` | 7 alpha JSONL files + `results.json` + feature indices |
| Random seed 1 | `control/seed_1_random/` | 7 alpha JSONL files + `results.json` + feature indices |
| Random seed 2 | `control/seed_2_random/` | 7 alpha JSONL files + `results.json` + feature indices |
| Comparison summary | `control/comparison_summary.json` | Slopes, CIs, cross-config comparison |
| Comparison plot | `control/sae_negative_control_comparison.png` | Visual summary |
| SAE classifier | `models/sae_detector.pkl` | 266-feature L1 probe (C=0.005) |
| SAE classifier summary | `pipeline/classifier_sae_summary.json` | C-sweep, metrics, top features |
| SAE feature analysis | `pipeline/sae_feature_analysis.json` | Verbosity correlations, max-activating samples |
| SAE extraction metadata | `pipeline/activations_sae_hlayers_16k_small/metadata.json` | Hook point, layers, dimensions |
