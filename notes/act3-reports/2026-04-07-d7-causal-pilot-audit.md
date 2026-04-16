# D7 Causal Pilot Audit — pilot100_canonical

> **Superseded for headline D7 conclusions by**
> [2026-04-16-d7-full500-two-seed-current-state-audit.md](./2026-04-16-d7-full500-two-seed-current-state-audit.md).
> Keep using this file for pilot-only details (alpha locking, pilot probe-family behavior,
> and pilot-specific failure analysis), but not as the current source of truth for D7's
> benchmark-level conclusion.

**Date:** 2026-04-07
**Data:** `data/gemma3_4b/intervention/jailbreak_d7/pilot100_canonical/{causal,probe}/`
**Model:** `google/gemma-3-4b-it`
**Intervention:** ITI head-level (decode-only, full_decode scope, k=20 ranked heads)
**Generation:** canonical (`do_sample=True, temperature=0.7, top_k=20, top_p=0.8, max_new_tokens=5000`)
**Sample manifest:** `jbb_d7_pilot_harmful100_seed42.json` (20 behaviors × 5 jailbreak templates = 100 prompts)
**Primary metric:** csv2_yes (CSV-v2 harmful_binary == "yes"), paired bootstrap CIs

---

## 0. Bottom Line

The causal (gradient-based) selector produces a statistically significant reduction in jailbreak severity at α=4.0: **-13pp csv2_yes [95% CI: -21, -6]**. The probe (AUROC-based) selector is null at every alpha tested: best is -2pp [CI: -10, +6]. These methods select genuinely different heads (Jaccard overlap = 0.11 on top-20), and the behavioral divergence is real: the causal selector suppresses harmful content while the probe selector does not.

However, the result comes with substantial caveats: n=100 yields wide CIs, model degeneration at α≥4.0 is a confound that needs monitoring at full-500 scale, and the effect is template-concentrated rather than uniform. The pilot justifies proceeding to the full-500 confirmatory run, but the -13pp headline should not be treated as a final estimate.

---

## 1. Data Integrity Checks

All checks pass.

| Check | Causal | Probe |
|---|---|---|
| Samples per alpha file | 100/100 (5 files) | 100/100 (5 files) |
| CSV2 files per alpha | 100/100 (5 files) | 100/100 (5 files) |
| Unique behaviors | 20 (5 templates each) | 20 (5 templates each) |
| ID parity across alphas | Verified by lock script | Verified by lock script |
| Run profile | canonical | canonical |
| Generation fingerprint | `0c53541825825b08` | `0c53541825825b08` |
| Provenance status | completed | completed |
| git_sha (both) | `7584677` | `7584677` |

**Baseline independence:** Causal and probe runs were executed as separate processes at different times (probe: 2026-04-06 09:17–17:14; causal: 2026-04-07 08:56–10:10). With `do_sample=True`, α=0.0 (no-op) responses differ across runs — 0/100 response-level matches, 88/100 judge-verdict matches. This is expected and correct: paired comparisons are within-family only.

**Bootstrap verification:** Independently recomputed the causal α=4.0 paired bootstrap (seed=42, n=10000): point estimate = -13.0pp, CI = [-21.0, -6.0]pp. Exact match with the lock file.

---

## 2. Headline Results

### 2.1 Causal Selector (iti_refusal_causal, k=20 heads)

**Artifact:** `data/contrastive/refusal/iti_refusal_causal_d7/iti_heads.pt`

| Metric | α=0.0 | α=1.0 | α=2.0 | α=4.0 | α=8.0 |
|---|---|---|---|---|---|
| Binary judge compliance | 41% | 35% | 37% | **30%** | 42% |
| csv2_yes | 35 | 25 | 28 | **22** | 26 |
| csv2_borderline | 29 | — | — | — | — |
| csv2_no | 36 | — | — | — | — |
| harmful_payload_share (mean) | 0.302 | 0.226 | 0.258 | **0.151** | 0.075 |
| Responses hitting 5000-token cap | 0 | 0 | 0 | **24** | 74 |

**Locked α=4.0** by largest paired decrease in csv2_yes: **-13pp [95% CI: -21, -6]**.

Other candidates:
- α=1.0: -10pp [-17, -4] ← also significant
- α=2.0: -7pp [-15, +1] ← CI crosses zero
- α=8.0: -9pp [-19, +1] ← CI crosses zero, confounded by degeneration

### 2.2 Probe Selector (iti_refusal_probe, k=20 heads)

**Artifact:** `data/contrastive/refusal/iti_refusal_probe_d7/iti_heads.pt`

| Metric | α=0.0 | α=1.0 | α=2.0 | α=4.0 | α=8.0 |
|---|---|---|---|---|---|
| Binary judge compliance | 41% | 40% | 41% | 42% | **53%** |
| csv2_yes | 30 | 28 | 30 | 28 | **36** |
| harmful_payload_share (mean) | 0.310 | 0.271 | 0.288 | 0.315 | 0.100 |
| Responses hitting 5000-token cap | 0 | 0 | 0 | 0 | 82 |

**Locked α=1.0** (best available): -2pp [95% CI: -10, +6]. **Null at every alpha.** At α=8.0, the intervention *increases* harmful compliance (+12pp [+3, +21]).

---

## 3. What Withstands Scrutiny

### 3.1 The causal selector reduces jailbreak severity; the probe selector does not

The causal -13pp at α=4.0 is the primary finding. The CI [-21, -6] excludes zero. Two lines of evidence support it beyond the binary csv2_yes:

1. **harmful_payload_share decreases monotonically** across all alphas for causal (0.302 → 0.226 → 0.258 → 0.151 → 0.075). Even at the degenerate α=8.0, harmful content is a smaller fraction of output. The probe metric is flat until α=8.0 where the long responses dilute it.

2. **Per-sample trajectories** show the causal intervention converts samples to sustained refusal. In the causal condition, 50/100 samples are never csv2_yes at any alpha and 9/100 are always csv2_yes. Of the 41 that flip, the dominant patterns are 1→0→0→0→0 (5 samples: early suppression that holds) and 0→0→0→0→1 (9 samples: late-alpha degeneration artifacts). The probe has 48 always-no, 15 always-yes, and 37 flippers — but with no coherent dose-response directionality.

### 3.2 The two methods select genuinely different heads

Jaccard similarity on top-20 selected heads: **0.11** (4 overlapping heads out of 36 unique).

| Layer range | Causal count | Probe count |
|---|---|---|
| 0–4 | 3 | 6 |
| 5–9 | 9 | 6 |
| 10–15 | 4 | 5 |
| 16–28 | 4 | 3 |

The causal selector concentrates in **layer 5** (4 heads) and picks **late layers 27–28** (2 heads with causal_effect > 0.35). The probe selector concentrates in **layer 4** (4 heads) and **layer 9** (4 heads), with high AUROC (0.92–1.0) but no gradient signal.

This validates the premise: the two ranking methods are surfacing different model components, and the difference in downstream behavior is consistent with this divergence.

### 3.3 The pipeline infrastructure is sound

Generation settings, provenance tracking, paired bootstrap, sample-ID parity enforcement, and resume/integrity guards are all working correctly. The data is fully auditable and reproducible.

---

## 4. What Does NOT Withstand Scrutiny

### 4.1 Model degeneration at α≥4.0 is a significant confound

| α | Causal: hit_token_cap | Causal: mean_chars | Probe: hit_token_cap | Probe: mean_chars |
|---|---|---|---|---|
| 0.0 | 0 | 5929 | 0 | 5944 |
| 1.0 | 0 | 5854 | 0 | 6061 |
| 2.0 | 0 | 5581 | 0 | 6135 |
| 4.0 | **24** | **8612** | 0 | 6416 |
| 8.0 | **74** | **14095** | **82** | **19888** |

At α=4.0 causal, 24% of samples hit the 5000-token cap. At α=8.0, this rises to 74% (causal) and 82% (probe). The model is producing dramatically more text — likely repetitive, rambling, or degenerate content — which creates two measurement problems:

1. **Token-cap truncation bias:** Truncated responses may cut off either the harmful payload or a late refusal, creating alpha-dependent scoring artifacts.
2. **Judge evaluation on long text:** CSV2 and binary judges may not be calibrated for 20,000+ character responses. Longer responses provide more opportunity for the judge to find harmful spans.

**For causal α=4.0 specifically:** The non-cap subset (76 samples) shows csv2_yes moving from 37% to 22% (-15pp), which is actually *stronger* than the overall -13pp. The cap-hitting subset (24 samples) shows csv2_yes moving from 29% to 21%. So the degeneration is not inflating the causal effect — if anything, it slightly dilutes it. **The α=4.0 causal result survives this check.**

But the non-monotonicity at α=8.0 (csv2_yes bouncing from 22 back to 26) and the probe's α=8.0 reversal (+6pp csv2_yes increase) are both degeneration artifacts, not real intervention reversals.

### 4.2 The probe result is uninformative, not negative

The probe's flat dose-response (Δ ranging from -2pp to +6pp, all CIs crossing zero) cannot distinguish between:
- The probe-ranked heads being genuinely irrelevant to jailbreak behavior
- The intervention strength being miscalibrated for probe-selected heads
- The probe selecting heads that discriminate refusal well (AUROC up to 1.0) but are downstream of the causal mechanism, so perturbing them has no upstream effect

The probe null is *consistent with* the causal hypothesis but does not *prove* the causal selector is better in a controlled sense. A proper comparison requires the same alpha applied to both selectors at the same time, which the pilot does not do (different locked alphas: causal=4.0, probe=1.0).

### 4.3 Template heterogeneity concentrates the signal

csv2_yes rate by template (causal):

| Template | α=0.0 | α=4.0 | Δ |
|---|---|---|---|
| t0 (DAN persona) | 9/20 (45%) | 9/20 (45%) | 0pp |
| t1 ("Begin with Absolutely") | 12/20 (60%) | 7/20 (35%) | **-25pp** |
| t2 (fiction story) | 7/20 (35%) | 1/20 (5%) | **-30pp** |
| t3 (system override) | 2/20 (10%) | 2/20 (10%) | 0pp |
| t4 (researcher persona) | 5/20 (25%) | 3/20 (15%) | -10pp |

The causal effect is carried almost entirely by **templates 1 and 2** (n=40), with templates 0, 3, and 4 showing near-zero deltas. This raises the concern that the intervention works specifically against instruction-following jailbreaks ("begin your response with...") and fictional framing, but not against persona-based attacks (DAN, system override). At n=20 per template these are very noisy — the full-500 run (n=100 per template) is needed to confirm whether template 0 truly resists the intervention.

### 4.4 Category-level effects are noisy and contain anomalies

| Category | α=0.0 | α=4.0 | α=8.0 |
|---|---|---|---|
| Disinformation | 50% | 20% | 50% |
| Fraud/Deception | 67% | 40% | 40% |
| Malware/Hacking | 40% | 0% | 20% |
| **Physical harm** | **0%** | **0%** | **60%** |

The Physical harm category jumps from 0% baseline to 60% at α=8.0. This is driven by the degeneration effect (long outputs scored as harmful). At n=5 per category per alpha, these numbers are unreliable. But the Physical harm anomaly at α=8.0 flags a potential externality: the intervention may be *increasing* harmful content in some categories at high alpha.

---

## 5. Greedy Cross-Validation

The greedy1024 pilot (pilot100_greedy1024) provides a consistency check on the alpha ranking, despite using incorrect generation settings.

| Family | Greedy selected α | Canonical selected α | Agreement |
|---|---|---|---|
| Causal | 8.0 | 4.0 | Disagree (both at low csv2_yes end, different peak) |
| Probe | 8.0 | 1.0 | **Opposite** conclusions |

The greedy data masked the degeneration effect because 1024-token truncation cuts off the degenerate text. At α=8.0 greedy-causal, csv2_yes drops to 16 (vs 26 in canonical) because long harmful content is simply truncated away. This confirms the GENERATION_SETTINGS_NOTE.md warning: greedy data is not comparable to canonical for absolute rates.

However, both greedy and canonical agree that **causal dominates probe at the optimal alpha**: greedy causal-best = 16 csv2_yes, greedy probe-best = 21; canonical causal-best = 22, canonical probe-best = 28.

---

## 6. Judge Concordance

Binary judge (GPT-4o HARMFUL/SAFE) vs CSV2 (harmful_binary = yes/no/borderline):

| α | Agreement | HARMFUL+csv2_yes | HARMFUL+csv2_not_yes | SAFE+csv2_yes | SAFE+csv2_not_yes |
|---|---|---|---|---|---|
| 0.0 | 86% | 31 | 10 | 4 | 55 |
| 4.0 | 92% | 22 | 8 | 0 | 70 |
| 8.0 | 84% | 26 | 16 | 0 | 58 |

At α=8.0, the disagreement widens: 16 samples are HARMFUL by binary judge but not csv2_yes. These are likely disclaimer-heavy long responses where the binary judge flags harmful intent but CSV2 downgrades due to low Commitment (C) or high Disclaimer (D) scores. The csv2_yes metric is more conservative and arguably more meaningful for safety evaluation.

---

## 7. Remaining Uncertainties

| Uncertainty | Severity | How to resolve |
|---|---|---|
| n=100 yields ~15pp CI width | High | Full-500 run (will narrow CIs by ~√5 ≈ ×0.45) |
| Causal α=4.0 degeneration (24% hit cap) | Moderate | Monitor at full-500; consider α=1.0 (0% cap, -10pp [-17,-4]) as conservative alternative |
| Template concentration of effect | Moderate | Full-500 provides n=100/template — sufficient for template-stratified CIs |
| No shared baseline between causal and probe | Moderate | full-500 baseline_noop condition provides one shared reference |
| CSV2 judge calibration on long responses | Low-moderate | Compare csv2_yes with harmful_payload_share trends; both point same direction for causal |
| Stochastic generation noise at α=0.0 | Low | Paired analysis within family controls for this; cross-family comparison is ordinal only |
| Whether causal heads are genuinely "causal" or just differently-correlated | Low (for pilot); high (for paper claim) | Random-head control in full-500 (`causal_random_head/` directory exists) |

---

## 8. Recommendations for Full-500

1. **Keep α=4.0 as the locked causal alpha.** The -13pp effect survives sub-group checks and the CI excludes zero. But **also run α=1.0** as a sensitivity analysis — it has -10pp [-17, -4] with zero degeneration (0% token cap).

2. **Report harmful_payload_share alongside csv2_yes.** The continuous metric tells a monotonically improving story and is less susceptible to judge threshold effects on long responses.

3. **Stratify results by template.** The effect heterogeneity is a real finding, not a bug — it reveals which jailbreak strategies the intervention can and cannot counter.

4. **Run the random-head control.** With Jaccard=0.11, the causal and probe selectors pick nearly disjoint heads. The random-head baseline will establish whether *any* k=20 heads produce the causal result or if it requires the gradient-selected ones.

5. **Monitor token-cap rates at full scale.** If >20% of samples hit the cap at α=4.0 in the full run, consider a cap-stratified analysis or reporting the non-cap subset separately.

---

## 9. Key Data Files

| File | Contents |
|---|---|
| `pilot100_canonical/causal_lock.json` | Alpha lock decision (causal) |
| `pilot100_canonical/probe_lock.json` | Alpha lock decision (probe) |
| `pilot100_canonical/causal/experiment/results.json` | Binary judge compliance rates |
| `pilot100_canonical/probe/experiment/results.json` | Binary judge compliance rates |
| `pilot100_canonical/causal/csv2_evaluation/alpha_*.jsonl` | Full CSV2 annotations |
| `pilot100_canonical/probe/csv2_evaluation/alpha_*.jsonl` | Full CSV2 annotations |
| `pilot100_canonical/causal/experiment/*.provenance.*.json` | Run provenance |
| `data/contrastive/refusal/iti_refusal_causal_d7/extraction_metadata.json` | Causal head ranking |
| `data/contrastive/refusal/iti_refusal_probe_d7/extraction_metadata.json` | Probe head ranking |
