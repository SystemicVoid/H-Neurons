# Week 3 Log — 2026-03-24 to 2026-03-29

> Post-week3 rerun resolution now lives in
> [2026-04-01-priority-reruns-audit.md](./2026-04-01-priority-reruns-audit.md).
> This file is the historical week-3 basis, not the current D1-vs-D4 ranking source.

> Website update basis. Each section names the headline claim, the raw data files to plot from, and links to the full audit where the claim lives. Do not restate the numbers in derived documents — link here instead.

---

## Overview

Week 3 closed the Baseline A evaluation (H-neuron scaling on FaithEval and jailbreak), built the graded jailbreak judge (CSV-v2), locked the Act 3 measurement contract, stood up the direction-steering infrastructure, and ran D2–D3.5 on Gemma-3-4B-IT. The D3.5 gate resolved: D4 proceeds without refusal orthogonalization.

**Model throughout:** `google/gemma-3-4b-it`
**Classifier:** `models/gemma3_4b_classifier.pkl` (38 H-neurons, L1-regularized logistic probe)
**Sprint contract:** [`notes/act3-sprint.md`](act3-sprint.md) · [`notes/measurement-blueprint.md`](measurement-blueprint.md)

---

## 1. Baseline A — FaithEval (H-neuron scaling)

> Superseded interpretation: after the 2026-03-31 audit, this section should be read as
> **anti-compliance MCQ context-acceptance**, not truthfulness improvement. The anti-compliance
> D1 branch also has `0/1000` parse failures at every alpha; the older parse-failure sentence
> below reflects the separate `faitheval_standard` branch and should not be reused for D1.

**Status:** Decision-complete. Canonical result locked at commit `3f226f8`.

**Claim:** H-neuron scaling monotonically increases compliance on a hallucination/credulous-claim benchmark. Effect is above noise.

### Headline numbers

| α | Compliance | 95% CI (Wilson) |
|---|---|---|
| 0.0 | 64.2% | [61.2, 67.1] |
| 0.5 | 65.4% | [62.4, 68.3] |
| 1.0 | 66.0% | [63.0, 68.9] |
| 1.5 | 67.0% | [64.0, 69.8] |
| 2.0 | 68.2% | [65.2, 71.0] |
| 2.5 | 69.5% | [66.6, 72.3] |
| 3.0 | 70.5% | [67.6, 73.2] |

- **Slope:** +2.09 pp/α, 95% CI [+1.38, +2.83]
- **Δ(0→3):** +6.3 pp, 95% CI [+4.2, +8.5]
- **Spearman ρ = 1.0** (perfectly monotonic)
- **Parse failures:** 0/1000 at every alpha on the anti-compliance D1 branch

### Plot data

| Plot | File |
|---|---|
| Compliance curve (all 7 alphas) | `data/gemma3_4b/intervention/faitheval/experiment/results.json` — field `results[alpha].compliance` |
| Per-item JSONL (paired bootstrap source) | `data/gemma3_4b/intervention/faitheval/experiment/alpha_*.jsonl` |
| Parse failure curve | same `results.json` — field `results[alpha].parse_failure` |

### Full audit
[`data/gemma3_4b/intervention_findings.md`](../data/gemma3_4b/intervention_findings.md) §1.1–1.5

---

## 2. Baseline A — Jailbreak (H-neuron scaling)

**Status:** Decision-complete. 4-alpha, 5000-token, CSV-v2 graded result locked at commit `8cdc80c`.

**History note:** An earlier 7-alpha, 256-token run appeared to show a significant slope (+6.2 pp, slope +2.14 pp/α). A 5000-token rerun on 2026-03-25 falsified that result — the 256-token window truncated responses mid-disclaimer, creating alpha-dependent false positives. The binary judge on the 5000-token run shows a non-significant effect. CSV-v2 recovers a real but structurally different result.

### 2a. Binary judge (5000-token canonical)

| α | Compliance | 95% CI (Wilson) |
|---|---|---|
| 0.0 | 30.4% | [26.5, 34.6] |
| 1.0 | 31.0% | [27.1, 35.2] |
| 1.5 | 32.2% | [28.3, 36.4] |
| 3.0 | 33.4% | [29.4, 37.6] |

- **Δ(0→3):** +3.0 pp, 95% CI [−1.2, +7.2] — **CI includes zero, not significant**
- **Slope:** +1.04 pp/α, 95% CI [−0.27, +2.35] — **not significant**

### 2b. CSV-v2 graded judge (GPT-4o, harmful_binary=yes)

| α | csv2_yes (harmful_binary=yes) | 95% CI (Wilson) | Mean payload share | V=3 rate |
|---|---|---|---|---|
| 0.0 | 94/500 = 18.8% | [15.6, 22.5] | 0.212 | 3.8% |
| 1.0 | 123/500 = 24.6% | [21.0, 28.6] | 0.227 | 9.8% |
| 1.5 | 118/500 = 23.6% | [20.1, 27.6] | 0.242 | 12.2% |
| 3.0 | 132/500 = 26.4% | [22.7, 30.4] | 0.263 | 14.0% |

- **Δ(0→3) csv2_yes:** +7.6 pp, 95% CI [+3.6, +11.6] — **significant**
- **Severity metrics (V, S, payload share, pivot position) are monotonic across all 4 alphas**
- **Structural insight:** 76% of the count increase is **ablation recovery** (α=0→1: +5.8 pp). The α=0.0 baseline suppresses ~30 responses that would otherwise be genuinely harmful. Amplification (α=1→3) adds +1.8 pp in count but drives the entire severity escalation.

**Interpretation:** The binary judge lacks resolution. CSV-v2 separates genuine compliance (C≥2) from disclaimer-wrapped discussion (C=1, constant noise floor of ~60 responses). V=3 (polished, actionable harmful content) nearly quadruples. Payload share rises 58% → 73%. Pivot position recedes 16% → 10% of response length.

**Caveats:** (1) No jailbreak negative control — H-neuron specificity unconfirmed. (2) Stochastic generation (`do_sample=True, temp=0.7`): per-item churn invalidates flip analysis. (3) No judge test-retest reliability for jailbreak.

### Plot data

| Plot | File |
|---|---|
| Binary compliance curve | `data/gemma3_4b/intervention/jailbreak/experiment/results.json` — field `results[alpha].compliance` |
| CSV-v2 per-item JSONL | `data/gemma3_4b/intervention/jailbreak/csv2_evaluation/alpha_*.jsonl` — fields `csv2.harmful_binary`, `csv2.C`, `csv2.S`, `csv2.V`, `csv2.harmful_payload_share`, `csv2.pivot_position` |
| CSV-v2 provenance | `data/gemma3_4b/intervention/jailbreak/csv2_evaluation/evaluate_csv2.provenance.20260327_221648.json` |

### Full audit
[`data/gemma3_4b/intervention/jailbreak/jailbreak_pipeline_audit.md`](../data/gemma3_4b/intervention/jailbreak/jailbreak_pipeline_audit.md) — §1.8 for CSV-v2 breakdown; §3.1 for template/category decomposition
[`data/gemma3_4b/intervention_findings.md`](../data/gemma3_4b/intervention_findings.md) — Finding 3 (binary) + CSV-v2 subsection

---

## 3. Act 3 Sprint Design & Measurement Contract

**Status:** Locked at commit `ca12f43` (2026-03-27), revised after external critique at `fe4d9ab` (2026-03-28).

Key decisions recorded in the sprint doc:

- Reframed as a **comparative steering sprint** (H-neurons as reference row, not the subject).
- **D3.5 methodology fix** adopted: must project neuron weights into residual-stream space via `down_proj` columns before cosine comparison — prior approach compared apples to oranges.
- **D4 uses difference-in-means** with diverse multi-source contrastive data, not single-dataset logistic regression.
- Baseline B (refusal-direction) reframed as a **diagnostic comparator**, not a presumed mitigator.
- TransformerLens available in parent venv but does **not** support Gemma-3-4B-IT (only Gemma-2 variants) — head-level work requires raw HuggingFace hooks.

→ [`notes/act3-sprint.md`](act3-sprint.md)
→ [`notes/measurement-blueprint.md`](measurement-blueprint.md)

---

## 4. D0.5 — Infrastructure & Dataset Freeze

**Status:** Partial. Refusal data frozen ✅. Harmful eval sets materialized ✅. IFEval + perplexity + truthfulness set: open.

### 4a. Refusal contrastive dataset freeze

- Frozen at commit `7ecf5a3` (2026-03-28)
- Source: pinned `refusal_direction` repo snapshot, seed-42 sampling, 128+128 train / 32+32 val / 100 harmful test
- One explicit leakage fix: 2 train-harmful prompts that normalized-overlapped the harmful test pool replaced
- **Not** a row-for-row recovery of the paper's hidden subset; faithful at the level of source pools and split sizes (Appendix A)
- Caveats in [`data/contrastive/refusal/metadata.json`](../data/contrastive/refusal/metadata.json)

### 4b. Harmful eval set materialization

Committed at `71fff94` (2026-03-29). Partitions the 572-record upstream pool into three explicit JSONL assets:

| Set | Count | Source |
|---|---|---|
| `jbb_100.jsonl` | 100 | JailbreakBench behaviors |
| `harmbench_test_159.jsonl` | 159 | HarmBench test split |
| `strongreject_313.jsonl` | 313 | StrongREJECT |

**Data:** `data/contrastive/refusal/eval/` (JSONL files + `eval_sets_metadata.json`)
**Provenance:** `data/contrastive/refusal/eval/materialize_harmful_eval_sets.provenance.20260329_133127.json`

---

## 5. D2 — Refusal Direction Extraction

**Status:** Done. Clean-provenance signoff at commit `bd1877b` (2026-03-29).

**Claim:** Difference-in-means on the frozen 128+128 contrastive set extracts a per-layer refusal direction. Best layer is 25 (98.4% val accuracy). Single-layer ablation at layer 25 reduces harmful refusal rate from 25% to 0%.

### Headline numbers

| Metric | Value |
|---|---|
| Best layer | 25 |
| Val accuracy at layer 25 | 98.4% |
| Separation score at layer 25 | 9,179 |
| Single-layer ablation: harmful refusal rate before | 25.0% (8/32) |
| Single-layer ablation: harmful refusal rate after | 0.0% (0/32) |
| Single-layer addition effect on harmless | 0% added refusals (expected — D3 uses all-layer) |

**Infrastructure note:** Gemma-3 exposes layer count and hidden dim under `model.config.text_config` (not `model.config` directly) and decoder layers at `model.model.language_model.layers`. Both `extract_direction.py` and `intervene_direction.py` were updated at commit `3eabb26` to handle this.

**Clean-rerun verdict:** Fresh extraction from a clean worktree reproduced the same refusal direction tensor hash, best layer (25), val accuracy (98.4%), and sanity gate result with `git_dirty: false`. Residual risk is conceptual (dataset construction, model specificity), not chain-of-custody.

### Plot data

| Plot | File |
|---|---|
| Per-layer accuracy curve (34 layers) | `data/contrastive/refusal/directions/extraction_metadata.json` — field `separation_scores[].accuracy` and `separation_scores[].separation` |
| Sanity gate outputs | `data/contrastive/refusal/directions/sanity_check_results.json` |
| Clean extraction provenance | `data/contrastive/refusal/directions_2026-03-29_dirty_provenance_snapshot/` (archived) and live `data/contrastive/refusal/directions/` |

### Full audit
[`notes/act3-sprint.md`](act3-sprint.md) — Decision log 2026-03-29 entries for D2

---

## 6. D3 — Baseline B: Refusal-Direction Ablation on FaithEval

**Status:** Decision-complete. Closeout at commit `be4d9ad` (2026-03-29).

**Claim:** All-layer refusal-direction ablation has a narrow usable point at β=0.02 but is not a robust alternative to H-neuron scaling. β=0.03 collapses compliance and introduces severe answer-option bias.

**Infrastructure note:** A micro-beta alpha label aliasing bug (`0.005/0.01/0.02` all collapsing to `alpha_0.0.jsonl` due to single-decimal formatting) was found and fixed at commit `9e96072` before the clean calibration run.

### Headline compliance (clean full run, `git_dirty: false`, n=1000 each)

| β | Compliance | 95% CI (Wilson) |
|---|---|---|
| 0.00 | 66.0% | [63.0, 68.9] |
| 0.02 | 70.2% | [67.3, 73.0] |
| 0.03 | 51.1% | [48.0, 54.2] |

### Row-level audit (β=0.02 → β=0.03)

| Transition | Count |
|---|---|
| False → True (gain) | 45 |
| True → False (loss) | 236 |
| Net | −191 |

### Answer-option distribution at β=0.03

| Option | β=0.00 | β=0.02 | β=0.03 |
|---|---|---|---|
| A | 22.2% | 25.4% | 24.0% |
| B | 26.6% | 28.2% | **58.1%** |
| C | 24.5% | 23.1% | 6.9% |
| D | 24.8% | 21.4% | 9.1% |

The compliance collapse at β=0.03 is primarily an answer-option bias failure (massive over-selection of B), not a parser failure (parse failure rate = 0% at all three β values).

### Per-correct-option compliance at β=0.03

| Correct option | β=0.00 | β=0.02 | β=0.03 |
|---|---|---|---|
| A | 60.8% | 73.8% | 65.8% |
| B | 70.5% | 75.1% | **87.3%** |
| C | 64.0% | 69.1% | **22.5%** |
| D | 68.9% | 63.3% | **29.5%** |

**Decision:** Do not broaden D3. Carry β=0.02 only as a scoped externality-check row in D5 if needed. Move directly to D4.

### Plot data

| Plot | File |
|---|---|
| 3-point β compliance curve | `data/gemma3_4b/intervention/faitheval_direction_ablate_d3_calibrated/experiment/results.20260329_192019.json` — field `results[alpha]` |
| Per-item JSONL (option distribution analysis) | `data/gemma3_4b/intervention/faitheval_direction_ablate_d3_calibrated/experiment/alpha_*.jsonl` |
| Clean run provenance | `data/gemma3_4b/intervention/faitheval_direction_ablate_d3_calibrated/experiment/run_intervention.provenance.20260329_192019.json` |
| Exploratory/calibration context (non-headline) | `data/gemma3_4b/intervention/faitheval_direction_ablate_microbeta_calibration/` and `faitheval_direction_ablate_all-layers_refusal-directions_c892775a83/` |

### Full report
[`notes/act3-reports/2026-03-29-d3-faitheval-refusal-direction.md`](act3-reports/2026-03-29-d3-faitheval-refusal-direction.md)

---

## 7. D3.5 — Refusal-Overlap Audit (Baseline A)

**Status:** Done with robustness downgrade. Gate resolved at commit `0a90ad8` (2026-03-29).

**Claim:** The projected 38-neuron residual update overlaps refusal geometry more than a layer-matched random-neuron null — but the mediation signal is dominated by a single layer (layer 33, 43× larger subspace gap than the next layer). The D4 gate is `proceed_with_d4_unchanged`.

**Sign convention:** D2 stores directions as `mean(harmful) − mean(harmless)`. Negative canonical cosine = anti-refusal / harmless-ward alignment. Both the canonical gap and the subspace gap are real departures from null in the expected directions.

### Headline geometry

| Metric | Estimate | 95% CI |
|---|---|---|
| Canonical signed cosine mean | −0.0173 | [−0.0177, −0.0169] |
| Canonical gap vs matched null | **−0.0183** | [−0.0310, −0.0126] |
| Refusal-subspace fraction mean | 0.0388 | [0.0373, 0.0404] |
| Refusal-subspace gap vs matched null | **+0.0361** | [+0.0251, +0.0390] |

### Full-model prompt-level correlations (Spearman ρ)

| Benchmark | Metric | ρ | 95% CI |
|---|---|---|---|
| FaithEval | canonical overlap vs compliance slope | −0.0869 | [−0.160, −0.012] |
| FaithEval | subspace overlap vs compliance slope | +0.0857 | [+0.011, +0.159] |
| Jailbreak | canonical overlap vs csv2_yes slope | −0.1164 | [−0.206, −0.024] |
| Jailbreak | subspace overlap vs csv2_yes slope | +0.1120 | [+0.022, +0.202] |

### Dominant-layer fragility check (layer 33 excluded)

| Benchmark | Metric | ρ after exclusion | 95% CI |
|---|---|---|---|
| FaithEval | canonical overlap vs compliance slope | −0.0048 | [−0.068, +0.060] |
| FaithEval | subspace overlap vs compliance slope | −0.0218 | [−0.086, +0.044] |
| Jailbreak | canonical overlap vs csv2_yes slope | +0.0303 | [−0.049, +0.110] |
| Jailbreak | subspace overlap vs csv2_yes slope | −0.1587 | [−0.237, −0.078] |

Layer 33 dominance: subspace gap +0.6647 vs next layer +0.0155 (43× ratio).

**Gate decision:** `proceed_with_d4_unchanged`. Treat refusal overlap as a live hypothesis, not a settled mechanism. Best next step (deferred): targeted layer-33 / top-neuron robustness pass.

### Plot data

| Plot | File |
|---|---|
| Headline geometry + null distribution | `data/gemma3_4b/intervention/refusal_overlap/analysis/summary.json` — fields `headline_geometry`, `benchmarks` |
| Per-layer overlap scores (34 layers) | `data/gemma3_4b/intervention/refusal_overlap/analysis/layer_scores.csv` |
| Per-prompt overlap and outcome slopes | `data/gemma3_4b/intervention/refusal_overlap/analysis/prompt_scores.csv` |
| Null distribution (100 random-neuron sets) | `data/gemma3_4b/intervention/refusal_overlap/analysis/null_distribution.json` |
| Full provenance | `data/gemma3_4b/intervention/refusal_overlap/analysis/analyze_refusal_overlap.provenance.20260329_223011.json` |

### Full report
[`data/gemma3_4b/intervention/refusal_overlap/refusal_overlap_audit.md`](../data/gemma3_4b/intervention/refusal_overlap/refusal_overlap_audit.md)

---

## 8. Infrastructure & Bug Fixes

| Fix | Commit | Impact |
|---|---|---|
| Gemma-3 nested config compatibility (`extract_direction.py`, `intervene_direction.py`) | `3eabb26` | Unblocked D2/D3 on Gemma-3-4B-IT |
| Micro-beta alpha label aliasing (`0.005/0.01/0.02` → `alpha_0.0`) | `9e96072`, `5012afc` | D3 calibration data was uninterpretable before fix |
| Jailbreak site payload drift (stale 7-point axis vs 4-point data) | `446f826` | Site was misrepresenting the current sweep |
| D3.5 gate hardening: sign-ambiguous evidence, dominant-layer exclusion test | `7a45e4d`, `f97c51c` | Prevented premature D4 orthogonalization |

---

## 9. What Is Open

| Item | Status | Blocks |
|---|---|---|
| Truthfulness contrastive set curation | Not started | D4 |
| IFEval + perplexity capability battery | Not started | D1 completion, D5 |
| Jailbreak negative control (random-neuron baseline) | Not started | H-neuron specificity confirmation |
| D1 graded-jailbreak negative control | Not started | Baseline A completeness |
| Layer-33 / top-neuron robustness pass (D3.5 follow-up) | Deferred | Conditional — only if cheaper than next D4 decision |

---

## Plot Registry for Website Session

These are the files to pull in the next session to generate charts for the website:

```
# Baseline A — FaithEval compliance curve (7 alphas)
data/gemma3_4b/intervention/faitheval/experiment/results.json

# Baseline A — Jailbreak binary compliance (4 alphas)
data/gemma3_4b/intervention/jailbreak/experiment/results.json

# Baseline A — Jailbreak CSV-v2 severity (per-item, 4 alphas)
data/gemma3_4b/intervention/jailbreak/csv2_evaluation/alpha_0.0.jsonl
data/gemma3_4b/intervention/jailbreak/csv2_evaluation/alpha_1.0.jsonl
data/gemma3_4b/intervention/jailbreak/csv2_evaluation/alpha_1.5.jsonl
data/gemma3_4b/intervention/jailbreak/csv2_evaluation/alpha_3.0.jsonl

# D2 — Per-layer accuracy / separation curve (34 layers)
data/contrastive/refusal/directions/extraction_metadata.json

# D3 — Refusal-direction ablation compliance curve (3 β values)
data/gemma3_4b/intervention/faitheval_direction_ablate_d3_calibrated/experiment/results.20260329_192019.json
data/gemma3_4b/intervention/faitheval_direction_ablate_d3_calibrated/experiment/alpha_0.0.jsonl
data/gemma3_4b/intervention/faitheval_direction_ablate_d3_calibrated/experiment/alpha_0.02.jsonl
data/gemma3_4b/intervention/faitheval_direction_ablate_d3_calibrated/experiment/alpha_0.03.jsonl

# D3.5 — Refusal overlap geometry and null distribution
data/gemma3_4b/intervention/refusal_overlap/analysis/summary.json
data/gemma3_4b/intervention/refusal_overlap/analysis/layer_scores.csv
data/gemma3_4b/intervention/refusal_overlap/analysis/prompt_scores.csv
data/gemma3_4b/intervention/refusal_overlap/analysis/null_distribution.json
```
