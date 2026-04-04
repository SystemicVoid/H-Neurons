# E2-A TriviaQA Source-Isolated Audit — 2026-04-03

> **Status: data authority for E2-A (paper-faithful selector overrides).**
>
> **Superseded by:** E2-B diagnostic ([2026-04-03-e2b-triviaqa-familydefault-diagnostic.md](./2026-04-03-e2b-triviaqa-familydefault-diagnostic.md)) tested the family-default selectors. Analytical synthesis closing the full TriviaQA transfer lane: [2026-04-04-e2-triviaqa-transfer-synthesis.md](./2026-04-04-e2-triviaqa-transfer-synthesis.md).
>
> This report is intentionally split into:
> 1. **Data (observations only)**
> 2. **Interpretation (inference from data)**
>
> Use this file for E2-A data. Use the synthesis for E2 conclusions.

## Source Hierarchy

- Strategy context:
  [optimise-intervention-ac3.md](./optimise-intervention-ac3.md)
- Sprint execution board:
  [act3-sprint.md](../act3-sprint.md)
- Predecessor E1 audit:
  [2026-04-02-e1-truthfulqa-modernized-audit.md](./2026-04-02-e1-truthfulqa-modernized-audit.md)
- E2 chain script:
  [scripts/infra/iti_e2_triviaqa_source_isolated_pipeline.sh](../../scripts/infra/iti_e2_triviaqa_source_isolated_pipeline.sh)
- Canonical E2 lock: `data/contrastive/truthfulness/iti_triviaqa_source_isolated_calibration/locked_iti_config.json`
- Consolidated machine-readable report: `notes/act3-reports/e2_triviaqa_source_isolated_report.json`
- Shortlist generation pilot report: `data/contrastive/truthfulness/iti_triviaqa_source_isolated_calibration/shortlist_generation_pilot_report.json`

---

## 1. Data (Observations Only)

### 1.1 Pipeline Execution and Integrity

- The E2 chain completed end-to-end under the pipeline script.
- All extraction, sweep, pilot, locking, fold eval, production, reporting, and judge phases ran without fatal errors.
- Provenance sidecars are present and marked `completed` for all extraction and inference stages.
- The pipeline uses `set -euo pipefail` and `systemd-inhibit` for sleep/idle protection.
- E2 uses the locked canonical scope `first_3_tokens` for all generation and MC decode, consistent with the scope decision from Stage 5.2.

### 1.2 Artifact Identity: All Four Extractions Are Bit-for-Bit Identical

**SHA256 of all E2 extraction artifacts:**

| Artifact | Path | SHA256 prefix |
|---|---|---|
| Calibration | `iti_triviaqa_source_isolated_calibration/iti_heads.pt` | `e24f219e069b77c6` |
| Fold 0 | `iti_triviaqa_source_isolated/final_fold0/iti_heads.pt` | `e24f219e069b77c6` |
| Fold 1 | `iti_triviaqa_source_isolated/final_fold1/iti_heads.pt` | `e24f219e069b77c6` |
| Production | `iti_triviaqa_source_isolated_production/iti_heads.pt` | `e24f219e069b77c6` |

All four artifacts have identical SHA256 hashes (`e24f219e069b77c6...`). This is because the `iti_triviaqa_transfer` family draws from fixed TriviaQA consistency data (`data/gemma3_4b/pipeline/consistency_samples.jsonl`) and fixed train/val QID splits (`train_qids.json` / `test_qids_disjoint.json`), none of which depend on TruthfulQA fold assignments. With identical inputs, seed, and policy overrides (`val_accuracy` + `last_answer_token`), extraction is deterministic.

**Implications:**
- The 2-fold CV is NOT cross-validated in the extraction sense — both folds use the same directions. This is methodologically correct: TriviaQA data cannot leak into TruthfulQA evaluation. The folds exist solely for evaluation coverage of all 655 TruthfulQA questions.
- The production artifact is identical to the calibration artifact. This means the pilot and final evaluations are testing the same object.
- The four extraction GPU passes were redundant. Future source-isolated pipelines could skip fold-specific re-extraction.

### 1.3 Extraction Metadata and Probe Quality

Extraction metadata confirms:
- `family = iti_triviaqa_transfer`
- `source_dataset = triviaqa_consistency`
- `head_ranking_metric = val_accuracy` (override applied; family default is `auroc`)
- `position_policy = last_answer_token` (override applied; family default is `all_answer_positions`)
- `prompt_format = Question: {question}\nAnswer: {answer}` (raw text, not chat-template)
- Training set: 2,642 examples from 2,000 TriviaQA questions
- Validation set: 782 examples

**Probe quality comparison (E2 vs E0):**

| Metric | E2 top-40 (K=40) | E0 top-12 (K=12) |
|---|---|---|
| val_accuracy (mean) | 0.677 | 0.751 |
| val_accuracy (max) | 0.696 | 0.772 |
| val_accuracy (min) | 0.665 | 0.739 |
| AUROC (mean) | 0.731 | 0.820 |
| AUROC (max) | 0.761 | 0.836 |
| Class separation (mean) | 1.111 | 1.140 |

The best E2 head (val_accuracy = 0.696) is below the *worst* E0 head (val_accuracy = 0.739). E2's probes are substantially weaker than E0's on a per-head basis, even though E2 uses 3.3× more heads. Layer distribution clusters in L9–L18 for both, but E0 probes are sharper at every rank position.

### 1.4 Calibration Sweep Characteristics

- Sweep table: `data/contrastive/truthfulness/iti_triviaqa_source_isolated_calibration/sweep_results.json`
- Grid: K ∈ {8, 12, 16, 24, 32, 40} × α ∈ {0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0} = 54 combos
- Calibration sample size: n=81 (MC1 and MC2)
- MC1 one-sample resolution: 100/81 = 1.235 pp
- All α=0.0 baselines are identical (as expected): MC1 = 22.22% (18/81), MC2 = 40.79%

**Sweep landscape:**

| Best MC1 | Candidate | Delta from baseline |
|---|---|---|
| 0.2716 | K=40, α=4.0 | +4.94 pp (4 extra correct) |

| Locked MC1 | Candidate | Delta from baseline |
|---|---|---|
| 0.2593 | K=40, α=6.0 | +3.70 pp (3 extra correct) |

The full MC1 range across the grid is [17.28%, 27.16%], a ±4.94 pp excursion around baseline. In absolute terms, the difference between the worst and best candidates is 4 correct answers on 81 questions.

**Comparison with E0 (paper-faithful) calibration sweep:**

| Metric | E2 (TriviaQA) | E0 (TruthfulQA) |
|---|---|---|
| Calibration baseline MC1 | 22.22% (18/81) | 22.22% (18/81) |
| Best calibration MC1 | 27.16% (22/81) | 38.27% (31/81) |
| Locked calibration MC1 | 25.93% (21/81) | 37.04% (30/81) |
| Calibration MC1 delta | +3.70 pp | +14.82 pp |
| Final held-out MC1 delta | **-0.92 pp** | **+6.3 pp** |

The E0 calibration signal was 4.0× stronger than E2's, and E0's signal replicated at ~43% strength in the final evaluation. E2's weak calibration signal (+3.70 pp = 3 answers on 81) did not replicate at all.

### 1.5 Locking and Pilot Gate

**Shortlist:** 5 candidates within 1.5 pp tolerance of best MC1 (27.16%):

| K | α | MC1 | MC2 | MC1 gap from best |
|---:|---:|---:|---:|---:|
| 40 | 4.0 | 0.2716 | 0.4079 | 0.00 pp |
| 40 | 6.0 | 0.2593 | 0.4381 | 1.23 pp |
| 40 | 8.0 | 0.2593 | 0.4315 | 1.23 pp |
| 16 | 6.0 | 0.2593 | 0.4168 | 1.23 pp |
| 32 | 4.0 | 0.2593 | 0.3969 | 1.23 pp |

All 5 shortlist candidates have K ≥ 16 and α ≥ 4.0. No low-K or low-α candidates survived.

**Pilot poison gate:** All 5 candidates passed with 0 rejections. Gate thresholds were:
- Reject if: (attempt_delta ≤ -10.0 pp AND precision_delta ≤ 0.0 pp) OR not_attempted_delta ≥ 15
- Baseline precision on the 100-ID pilot: 5% (5/100 CORRECT)
- All candidates produced between 3 and 5 CORRECT answers
- No candidate showed any NOT_ATTEMPTED count (all attempted everything)

**Lock decision:** K=40, α=6.0 selected by MC2 tie-break (best MC2 = 0.4381 in the shortlist after the 1.5 pp MC1 tolerance filter). The locked candidate has MC1 that is 1.23 pp (exactly 1 correct answer) below the best MC1 in the shortlist.

### 1.6 Final 2-Fold TruthfulQA Outcomes (Within-E2: α=6.0 vs α=0.0)

Pooled held-out (n=655):

| Metric | Baseline α=0.0 | Intervened α=6.0 | Delta (pp) | 95% CI |
|---|---:|---:|---:|---:|
| MC1 | 26.72% | 25.80% | **-0.92** | [-3.36, +1.53] |
| MC2 truthful mass | 42.86% | 43.59% | **+0.73** | [-1.21, +2.65] |

Both CIs include zero. The intervention produces no statistically significant effect on either MC metric compared to its own α=0 baseline.

### 1.7 SimpleQA-200 Outcomes (Within-E2: α=6.0 vs α=0.0, `first_3_tokens`)

Judge-complete results:

| Alpha | CORRECT | INCORRECT | NOT_ATTEMPTED | Compliance | Attempt rate | Precision |
|---|---:|---:|---:|---:|---:|---:|
| 0.0 | 11 | 187 | 2 | 5.50% | 99.0% | 5.56% |
| 6.0 | 10 | 188 | 2 | 5.00% | 99.0% | 5.05% |

Paired deltas (α=6.0 minus α=0.0):

| Metric | Delta (pp) | 95% CI |
|---|---:|---:|
| Compliance | -0.50 | [-3.50, +2.50] |
| Attempt rate | 0.00 | [-1.50, +1.50] |
| Precision | -0.51 | [-3.66, +2.59] |

All three CIs include zero. The intervention has no detectable effect on generation.

**Transition table (within-E2):**

| Transition | Count |
|---|---:|
| INCORRECT → INCORRECT | 181 |
| CORRECT → INCORRECT | 6 |
| INCORRECT → CORRECT | 5 |
| CORRECT → CORRECT | 5 |
| NOT_ATTEMPTED → INCORRECT | 1 |
| INCORRECT → NOT_ATTEMPTED | 1 |
| NOT_ATTEMPTED → NOT_ATTEMPTED | 1 |

The 6 CORRECT→INCORRECT and 5 INCORRECT→CORRECT transitions nearly cancel. This is consistent with a near-zero net effect at this event rate; the data do not support a directional gain, but they also do not cleanly distinguish zero from very small harm.

### 1.8 Cross-Experiment Comparisons

#### E2 vs Paper (E0: K=12, α=8.0, TruthfulQA-paperfaithful)

| Metric | E0 (paper) | E2 | Delta (pp) | 95% CI | Sig? |
|---|---:|---:|---:|---:|---|
| TruthfulQA MC1 | 32.98% | 25.80% | **-7.18** | [-10.23, -4.12] | Yes |
| TruthfulQA MC2 | 50.36% | 43.59% | **-6.77** | [-9.23, -4.36] | Yes |
| SimpleQA compliance | 4.00% | 5.00% | +1.00 | [-2.00, +4.00] | No |
| SimpleQA attempt rate | 89.50% | 99.00% | **+9.50** | [+5.50, +14.00] | Yes |
| SimpleQA precision | 4.47% | 5.05% | +0.58 | [-2.64, +3.91] | No |

#### E2 vs E1 (K=8, α=8.0, TruthfulQA-modernized)

| Metric | E1 | E2 | Delta (pp) | 95% CI | Sig? |
|---|---:|---:|---:|---:|---|
| TruthfulQA MC1 | 29.47% | 25.80% | **-3.66** | [-6.41, -0.92] | Yes |
| TruthfulQA MC2 | 47.35% | 43.59% | **-3.76** | [-6.10, -1.44] | Yes |
| SimpleQA compliance | 6.00% | 5.00% | -1.00 | [-4.00, +2.00] | No |
| SimpleQA attempt rate | 97.50% | 99.00% | +1.50 | [0.00, +3.50] | Marginal |
| SimpleQA precision | 6.15% | 5.05% | -1.10 | [-4.25, +1.98] | No |

#### E2 vs α=0.0 baselines (all variants share the same baseline)

The α=0.0 baselines for E2, E1, and E0 should be identical for TruthfulQA MC (same unsteered model, same questions). The within-E2 baseline is MC1 = 26.72%, MC2 = 42.86%. E0's baseline at α=0 is not separately reported in the E2 JSON (the cross-comparison uses E0's locked α=8.0, not α=0.0), but the E0 final evaluation showed baseline MC1 = 26.72% confirming consistency.

### 1.9 Head Overlap Diagnostics

#### E2 vs Paper (E0)

| Metric | Value |
|---|---|
| E2 selected heads | 40 |
| E0 selected heads | 12 |
| Intersection | 4 |
| Jaccard index | 0.083 (8.3%) |
| Overlapping heads | L10H2, L10H5, L11H0, L15H2 |
| Shared ranked heads (all positions) | 272 |
| Spearman ρ (rank agreement) | 0.543 (p < 0.001) |
| Direction cosine on shared selected heads (mean) | **-0.163** |
| Direction cosine (range) | [-0.351, +0.068] |

#### E2 vs E1

| Metric | Value |
|---|---|
| E2 selected heads | 40 |
| E1 selected heads | 8 |
| Intersection | 1 |
| Jaccard index | 0.021 (2.1%) |
| Overlapping head | L15H2 |
| Shared ranked heads | 187 |
| Spearman ρ (rank agreement) | 0.354 (p < 0.001) |
| Direction cosine on shared selected head | +0.122 |

### 1.10 Automated Classification

The `_classify_outcome` function classified E2 as `gentler_but_weaker_tradeoff` based on:
- `e2_vs_paper_compliance_delta_pp > 0.0` (True: +1.00 pp)
- `e2_vs_paper_mc1_delta_pp < 0.0` (True: -7.18 pp)

---

## 2. Interpretation (Inference From Data)

### 2.1 What Survives Skeptical Review

1. **E2's intervention is near-inert on all measured surfaces.**
   The within-E2 comparisons (α=6.0 vs α=0.0) show no statistically significant effect on MC1, MC2, SimpleQA compliance, attempt rate, or precision. All 95% CIs include zero. The transition table shows nearly symmetric correct-to-incorrect and incorrect-to-correct flips (6 vs 5), consistent with random perturbation rather than directional steering.

2. **Source isolation successfully prevents contamination.**
   The bit-for-bit identical artifacts across all folds confirm that TriviaQA extraction is independent of TruthfulQA fold assignments. The source-isolated setup shows no evidence of TruthfulQA fold leakage into extraction.

3. **TriviaQA-derived probes are weaker than TruthfulQA-derived probes.**
   The best E2 head (val_accuracy = 0.696) is below the worst E0 head (val_accuracy = 0.739). This is not a ranking metric artifact: AUROC tells the same story (E2 mean 0.731 vs E0 mean 0.820). The TriviaQA consistency signal at the last-answer-token position is noisier than the TruthfulQA best-answer signal at the same position.

4. **The calibration signal did not replicate.**
   Calibration MC1 delta was +3.70 pp (3 extra correct on 81 questions). The final held-out delta was -0.92 pp (CI [-3.36, +1.53]). This is the textbook pattern of fitting to noise on a small calibration set. By contrast, E0's calibration signal (+14.82 pp) replicated at ~43% strength (+6.3 pp) in the final evaluation.

5. **E2 selects fundamentally different heads from both E0 and E1.**
   Jaccard overlap with E0 is 8.3%, with E1 is 2.1%. The direction cosines on the 4 shared selected heads between E2 and E0 average -0.163, suggesting they may point in partially *opposite* directions — though n=4 is too small for this to be more than suggestive. The TriviaQA-trained probes and the TruthfulQA-trained probes agree moderately on which heads carry variance (Spearman ρ = 0.543 across 272 shared ranked heads) but appear to disagree on which direction is "truthful."

6. **Pipeline engineering is solid.**
   The infrastructure — provenance tracking, paired bootstrap CIs, identical-sample-ID enforcement, poison gates, fail-fast assertions, reproducibility via fixed seeds, incremental persistence — is well-designed and internally consistent.

### 2.2 What Does Not Survive Scrutiny

1. **"Gentler but weaker tradeoff" overstates E2's contribution.**
   The automated classifier labels E2 as `gentler_but_weaker_tradeoff` because E2 has slightly higher compliance (+1.00 pp, non-significant) and much higher attempt rate (+9.50 pp, significant) compared to the paper's locked config. But this "gentleness" is the absence of E0's refusal-inducing intervention, not a positive property of E2's intervention. Since E2's own α=6.0 vs α=0.0 comparison shows no effect, E2's generation behavior is simply the unsteered model's behavior. A more precise label would be `near_inert`.

2. **The pilot poison gate provided no information.**
   With 5/100 baseline correct answers and no NOT_ATTEMPTED events at any alpha, the gate thresholds (-10.0 pp attempt, 15 NOT_ATTEMPTED) were non-discriminating in practice. All 5 candidates attempted everything and produced zero NOT_ATTEMPTED responses, so the gate could not distinguish among them. In principle, the rule could have fired if a candidate had dropped attempt rate by 10 pp or created 15+ NOT_ATTEMPTEDs — but no candidate came close. The pilot was properly executed but the gate provided no information at this baseline performance level.

3. **The lock is procedurally valid but brittle.**
   Multiple K values (16, 32, 40) produced MC1 improvements above baseline at moderate α — the shortlist itself contains K=16,α=6.0 with the same MC1 (0.2593) as the locked K=40,α=6.0. The data do not show "only K=40 works"; they show the lock rule selected K=40 after an MC2 tie-break on n=81, where the locked candidate was exactly one MC1 question below the best point and had a smaller-K rival (K=16,α=6.0) with identical MC1. When the entire calibration landscape spans only four correct answers from worst to best, small tie-break wins should be treated as brittle, not structural destiny. Using 40 of 272 ranked heads (14.7%) does suggest the TriviaQA signal is more diffuse than E0's K=12 peak, but this interpretation should not rest on the K=40 lock alone.

4. **The 2-fold extraction is structurally redundant for source-isolated families.**
   Since all four extractions are identical, the pipeline spent 4 GPU passes producing the same artifact. The 2-fold structure adds no robustness to the extraction; it only ensures complete TruthfulQA evaluation coverage. This is correct but should be acknowledged in the pipeline documentation.

### 2.3 Mechanistic Read (Low Confidence — Descriptive, Not Proven Causal)

An analogy for the E2 outcome: E2 applied a **diluted, mis-oriented perturbation** to the model's representations.

- **Diluted**: The TriviaQA probes are weaker (val_accuracy ~0.68 vs E0's ~0.75), meaning the directions are noisier estimates of whatever signal exists. Spreading this across 40 heads compounds the dilution.
- **Mis-oriented** (preliminary): The negative direction cosines (mean -0.163) on the 4 shared selected heads suggest the TriviaQA "truthful" direction and the TruthfulQA "truthful" direction are not aligned — they may reflect partially different latent concepts. However, n=4 is too small for this to be more than suggestive.
- **The rank agreement (ρ = 0.543) is interesting**: the probes agree moderately on *which* attention heads carry truthfulness-relevant variance, but disagree on *which direction* is truthful. This could reflect the fact that TriviaQA closed-book factuality and TruthfulQA misconception resistance, while both "truthfulness-adjacent," activate different geometric structures in the same heads.

This frame is consistent with the data but is not proven causal. The near-inert outcome could also result from the interventions being too weak (low α relative to the noise level of the directions) or the decode scope limiting exposure to only 3 tokens of weak perturbation.

### 2.4 What's Interesting Despite the Null Result

1. **The rank-direction dissociation is intriguing but preliminary.** Moderate rank agreement (ρ = 0.543 across 272 shared ranked heads) but negative direction cosines (mean = -0.163) on the 4 shared *selected* heads suggests the truthfulness concept may not be unitary in this model. However, the direction-cosine summary rests on only 4 overlapping selected heads — too small a sample for a strong conclusion. This is weakly suggestive against the "universal truthfulness hyperplane" hypothesis (Liu et al. 2024) at the attention-head level for this model, but needs replication with a larger overlap set before it can carry interpretive weight.

2. **The calibration landscape is structurally different from E0's.** E0's calibration surface showed a clear hot zone at K ∈ {8, 12, 16} with K ≥ 24 collapsing. E2's positive signal is diffused across multiple K values (16, 32, 40) with small magnitudes, and the lock required an MC2 tie-break among candidates separated by at most one MC1 question. This is consistent with the TriviaQA truthfulness signal living in a wider, weaker subspace than TruthfulQA's, though the flat calibration landscape also means the K choice carries less structural information than in E0.

3. **E2's attempt rate (99.0%) is stable under intervention** — neither α=0 nor α=6.0 produces NOT_ATTEMPTED responses. This contrasts sharply with E0 (attempt rate drops to 89.5% at α=8.0). The TriviaQA directions do not interact with the model's refusal circuitry, consistent with the source isolation design goal.

---

## 3. Uncertainty Register

| Uncertainty | Level | Basis |
|---|---|---|
| Whether val_accuracy ranking is appropriate for TriviaQA heads | **High** | E2 uses paper-faithful selectors (val_accuracy + last_answer_token) as an override on a family that defaults to AUROC + all positions. The family-default selectors might extract a stronger artifact. |
| Whether TriviaQA→TruthfulQA transfer works for this model at any operating point | **High** | The near-null result could mean the transfer fails on Gemma-3-4B-IT specifically, regardless of extraction policy. |
| Whether the calibration sweep (n=81) had sufficient power to detect a real E2 effect | **High** | The entire locking decision rests on 3 extra correct answers out of 81. E0's locking rested on 12 extra correct. |
| Whether the TriviaQA consistency data quality is suitable for head probing | **Medium** | Probe accuracy ~0.68–0.70 indicates moderate signal exists, but it may not be the right signal for TruthfulQA MC discrimination. |
| SimpleQA event rate (n=200, ~10 correct) | **Medium** | CIs are wide (±3 pp). Any effect smaller than ~3 pp is invisible. |
| Seed-specific split sensitivity | **Low** | One fixed seed (42) determines all splits. Unlikely to dominate, but not verified. |

---

## 4. Decision and Next Steps

### 4.1 Decision

- **E2-A is a valid, well-executed null for the source-isolated TriviaQA artifact under paper-faithful override selectors** (val_accuracy ranking + last_answer_token). This is a negative result for E2-A specifically, not yet for the entire TriviaQA transfer lane — the family-default selectors (AUROC + all_answer_positions) remain untested. The near-null within-E2 effect is the primary finding; the "gentler but weaker" cross-comparison label describes the absence of an effect, not the presence of a desirable tradeoff. A more precise label is `near_inert`.
- **The lock is procedurally valid but not robust.** The lock rule was predeclared, which is good. But it picked K=40,α=6.0 after an MC2 tie-break on n=81, where the locked candidate was exactly one MC1 question below the best point and had a smaller-K rival (K=16,α=6.0) with identical MC1. When the entire calibration landscape spans only four answers from worst to best, small tie-break wins should be treated as brittle, not structural destiny.
- **Promote result as:**
  - "E2-A produces no detectable effect on TruthfulQA MC or SimpleQA generation under paper-faithful override selectors"
  - "TriviaQA-derived head directions are weakly anti-correlated with TruthfulQA-derived directions (cosine ≈ -0.16 on 4 shared heads) despite moderate rank agreement (ρ ≈ 0.54 on 272 shared heads) — suggestive but preliminary"
  - "The calibration signal (+3.7 pp on 81 questions) did not replicate in the 655-question held-out evaluation"
  - "Source isolation is methodologically sound but the transferred signal under these selectors is too weak to be useful"
  - "Whether TriviaQA transfer fails entirely on Gemma-3-4B-IT, or whether E2-A simply chose the wrong extraction selectors, remains an open question"

### 4.2 E3 Conditional Assessment

E1 showed a tradeoff (MC↑ / generation↑ vs paper, but MC↓ / generation↑ vs paper). E2-A shows near-null. These are NOT complementary in the way §5.5 of the strategy note envisioned:
- E1 changes something (AUROC-selected, TruthfulQA-sourced directions steer the model, but in a MC-degrading direction relative to paper)
- E2-A changes nothing under paper-faithful selectors (TriviaQA-sourced directions don't engage the TruthfulQA-relevant representations)

Mixing an active-but-wrong-direction signal (E1) with an inert signal (E2-A) is unlikely to help. The E3 conditional gate ("E1 and E2 appear complementary in head selection or benchmark profile") is **not met** for E2-A; untested for family-default selectors.

### 4.3 Highest-Value Next Steps

1. **Diagnostic: E2 with family-default selectors.** Extract a TriviaQA artifact using AUROC ranking + all answer positions (E2's family defaults, without source-isolated overrides). Cost: 1 GPU extraction. This would separate "wrong ranking metric" from "wrong data source" — if the family-default artifact shows a signal, the paper-faithful selectors were the wrong choice for TriviaQA.

2. **Accept the three-variant evidence (conditional on E2-A scope).** E0 (paper-faithful TruthfulQA) works on MC but harms generation. E1 (modernized TruthfulQA) partially trades MC for generation. E2-A (TriviaQA, paper-faithful selectors) is inert. This pattern suggests mass-mean ITI under paper-faithful selectors on Gemma-3-4B-IT is fundamentally limited by a direction-quality ceiling. However, E2-A's negative result does not yet close the TriviaQA lane entirely — the family-default selectors remain untested. Consider whether the D4 ITI path has reached diminishing returns and whether D7 (causal head selection) or adaptive-α work should be prioritized.

3. **Document the rank-direction dissociation (with appropriate caveats).** The moderate rank agreement (ρ = 0.543 on 272 heads) but negative direction cosine (mean = -0.163 on 4 shared selected heads) between TriviaQA and TruthfulQA heads is an intriguing observation worth preserving for the final synthesis (D8). The rank agreement is on solid footing; the direction-cosine claim needs replication with a larger overlap set.
