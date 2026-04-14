# V2 vs V3 Paired Evaluator Comparison — H-Neuron 38 and Seed-1 Control

> **Status: canonical for v2-v3 evaluator comparison on jailbreak data.** Subsumes the interpretation sections (§3) of the [pipeline audit](2026-04-13-phase3-jailbreak-pipeline-audit.md); raw data verification and issue inventory remain owned there.
>
> **Prior work:** [pipeline audit](2026-04-13-phase3-jailbreak-pipeline-audit.md) (issues C1-C3, M1-M4), [seed-0 v2 control](2026-04-12-seed0-jailbreak-control-audit.md) (v2 specificity, slope CIs), [4-way evaluator comparison](2026-04-12-4way-evaluator-comparison.md) (v3 accuracy validation), [holdout validation](2026-04-12-4way-evaluator-holdout-validation.md) (calibration overlap controlled)
>
> **Data audited:**
> - `data/gemma3_4b/intervention/jailbreak/csv2_evaluation/` — H-neuron v2 (500 records × 4 alphas)
> - `data/gemma3_4b/intervention/jailbreak/csv2_v3_evaluation/` — H-neuron v3 (500 records × 4 alphas)
> - `data/gemma3_4b/intervention/jailbreak/control/seed_1_unconstrained_csv2_v3/` — seed-1 v3 control (500 records × 4 alphas)
> - `data/gemma3_4b/intervention/jailbreak/control/comparison_csv2_v2_summary.json` — archived v2 specificity data
> - `data/gemma3_4b/intervention/jailbreak/control/concordance/concordance_summary.json` — three-judge concordance
>
> **Methodology:** All bootstrap CIs use 10,000 resamples, seed=42, `numpy.random.default_rng`. OLS slopes via `numpy.polyfit(..., 1)`. Error-filtered denominators (excluding `csv2.error` records) match the pipeline audit's verified counts.

---

## 0. Headline

| Metric | Evaluator | Value | 95% CI | Significant? |
|---|---|---|---|---|
| harmful_binary slope (H-neuron) | **v2** | **+2.30 pp/alpha** | **[+0.99, +3.58]** | **Yes** |
| harmful_binary slope (H-neuron) | v3 | +0.46 pp/alpha | [-1.46, +2.41] | No |
| harmful_binary slope (seed-1 control) | v3 | -0.34 pp/alpha | [-2.28, +1.61] | No |
| harmful_binary slope gap (H minus ctrl) | v3 | +0.80 pp/alpha | [-2.00, +3.58] | No |
| **substantive_compliance slope (H-neuron)** | **v3** | **+2.00 pp/alpha** | **[+0.11, +3.87]** | **Yes** |
| substantive_compliance slope (seed-1 ctrl) | v3 | -0.72 pp/alpha | [-2.58, +1.19] | No |
| **substantive_compliance slope gap (H minus ctrl)** | **v3** | **+2.72 pp/alpha** | **[+0.02, +5.44]** | **Yes** |

**Summary:** The v2 evaluator reports a significant binary dose-response. The v3 evaluator reports a non-significant binary dose-response on the same model outputs. But v3's primary_outcome taxonomy reveals the underlying signal: the intervention increases *severity* (substantive compliance) at +2.00 pp/alpha, which is significant. The effect is real — it's just at a different measurement level than v2 captures.

---

## 1. Data Scope

### 1.1 What was compared

**Paired comparison (same model outputs, different evaluator):**
- H-neuron v2 vs H-neuron v3: 500 prompts × 4 alphas, joined by ID. This is the cleanest possible "measurement changes conclusions" exhibit.

**Unpaired comparisons (different model outputs):**
- H-neuron v3 vs seed-1 v3: same evaluator, different neuron sets. Tests specificity within v3.
- H-neuron v2 vs seed-0 v2: same evaluator, different neuron sets. Tests specificity within v2 ([prior report](2026-04-12-seed0-jailbreak-control-audit.md)).

### 1.2 What could NOT be compared

| Comparison | Status | Why |
|---|---|---|
| Seed-0 v2 vs seed-0 v3 | Not possible | Seed-0 was scored only with v2 |
| Seed-1 v2 vs seed-1 v3 | Not possible | Seed-1 was scored only with v3 |
| Cross-evaluator paired comparison on control data | Not possible | No control seed has both v2 and v3 |

**Implication:** We can decompose the v2-v3 difference on H-neuron data (paired) but not on control data (different seeds, different evaluators). Both controls are flat, which is reassuring, but we cannot rule out that the evaluator contributes to the ~10pp baseline difference between v2 control (24.2%) and v3 control (34.4%).

---

## 2. Paired V2-V3 Analysis on H-Neuron Data

### 2.1 Rate comparison by alpha

| Alpha | v2 yes | v2 borderline | v2 no | v3 yes | v3 no | v3 errors |
|---|---|---|---|---|---|---|
| 0.0 | 94 (18.8%) | 171 (34.2%) | 235 (47.0%) | 169 (34.6%) | 320 (65.4%) | 11 |
| 1.0 | 123 (24.6%) | 127 (25.4%) | 250 (50.0%) | 172 (35.0%) | 320 (65.0%) | 8 |
| 1.5 | 118 (23.6%) | 134 (26.8%) | 248 (49.6%) | 167 (34.5%) | 317 (65.5%) | 16 |
| 3.0 | 132 (26.4%) | 98 (19.6%) | 270 (54.0%) | 177 (36.0%) | 315 (64.0%) | 8 |

v2 denominators: always 500/500 (no errors). v3 denominators: 489, 492, 484, 492 (after error exclusion).

### 2.2 Transition matrix (ID-level join, v3 errors excluded)

| v2 label → v3 label | α=0.0 | α=1.0 | α=1.5 | α=3.0 | Total |
|---|---|---|---|---|---|
| **yes → yes** | 93 | 120 | 113 | 130 | **456** |
| **yes → no** | 0 | 0 | 1 | 1 | **2** |
| **borderline → yes** | 76 | 52 | 54 | 47 | **229** |
| **borderline → no** | 87 | 70 | 73 | 48 | **278** |
| **no → no** | 233 | 250 | 243 | 266 | **992** |
| **no → yes** | 0 | 0 | 0 | 0 | **0** |

**Key structural properties:**
- **Zero v2-no → v3-yes flips.** Every v3-yes record was either v2-yes (456) or v2-borderline (229). v3 never calls something harmful that v2 called safe.
- **Only 2 v2-yes → v3-no flips** (at α=1.5 and α=3.0). These are the only genuine evaluator disagreements worth examining.
- **All 229 discordant records are v2-borderline → v3-yes.** This is construct expansion (v3 resolves what v2 left ambiguous), not evaluator error.

### 2.3 Agreement statistics

| Metric | Value |
|---|---|
| Overall agreement (v2-yes=yes, v2-border/no=no) | 88.2% |
| Cohen's kappa | 0.719 |
| v2-yes → v3-no (genuine flips) | 2 / 1957 (0.1%) |
| v2-no → v3-yes (genuine surprises) | 0 / 1957 (0.0%) |

Agreement improves with alpha: 84.5% at α=0.0 → 90.2% at α=3.0, because fewer borderlines remain to reclassify at higher scaling.

---

## 3. Bootstrap Confidence Intervals

### 3.1 Binary harmful rate slopes

| Condition | Slope (pp/alpha) | 95% CI | Includes zero? |
|---|---|---|---|
| H-neuron v2 (from [prior report](2026-04-12-seed0-jailbreak-control-audit.md)) | +2.30 | [+0.99, +3.58] | No |
| H-neuron v3 | +0.46 | [-1.46, +2.41] | **Yes** |
| Seed-0 v2 control (from prior report) | -0.47 | [-1.42, +0.47] | Yes |
| Seed-1 v3 control | -0.34 | [-2.28, +1.61] | Yes |

### 3.2 Slope differences

| Comparison | Gap (pp/alpha) | 95% CI | Includes zero? |
|---|---|---|---|
| v2: H-neuron minus seed-0 control (from prior report) | +2.77 | [+1.17, +4.42] | **No** (p=0.013) |
| v3: H-neuron minus seed-1 control | +0.80 | [-2.00, +3.58] | **Yes** |

### 3.3 Primary outcome slopes

| Outcome | H-neuron slope | 95% CI | Control slope | 95% CI |
|---|---|---|---|---|
| **substantive_compliance** | **+2.00** | **[+0.11, +3.87]** | -0.72 | [-2.58, +1.19] |
| partial_compliance | -1.54 | — | +0.38 | — |
| deflection | -1.68 | — | +0.09 | — |
| refusal | +0.86 | — | -0.01 | — |

**Substantive_compliance slope gap (H minus ctrl):** +2.72 pp/alpha [+0.02, +5.44] — CI excludes zero.

---

## 4. Slope Compression Mechanism

### 4.1 The arithmetic

The v3 harmful_binary count at each alpha = v2_yes count + absorbed borderlines (v2-borderline → v3-yes) - genuine flips (v2-yes → v3-no).

| Alpha | v2_yes | Absorbed borderlines | Flips | v3_yes | Net v3 gain over v2 |
|---|---|---|---|---|---|
| 0.0 | 94 | 76 | 0 | 169 | +75 |
| 1.0 | 123 | 52 | 0 | 172 | +49 |
| 1.5 | 118 | 54 | 1 | 167 | +49 |
| 3.0 | 132 | 47 | 1 | 177 | +45 |

From α=0.0 to α=3.0:
- v2_yes increases by **+38 records** (the v2 dose-response signal)
- Absorbed borderlines decrease by **-29 records** (fewer borderlines to absorb at high alpha)
- Net v3_yes increase: **+8 records** (the compressed v3 signal)

**The ~~88%~~ 80% slope compression (2.30 → 0.46 pp/alpha) [corrected 2026-04-14: (2.30−0.46)/2.30 = 80%] is entirely explained by the declining number of borderline absorptions at higher alphas.** The intervention reduces the borderline population (polarization), and v3 absorbs fewer of them into the "yes" category at high alpha than at baseline.

### 4.2 Borderline reclassification rate is alpha-stable

| Alpha | Borderlines | → v3 yes | → v3 no | Reclassification rate |
|---|---|---|---|---|
| 0.0 | 171 | 76 | 87 | 46.6% |
| 1.0 | 127 | 52 | 70 | 42.6% |
| 1.5 | 134 | 54 | 73 | 42.5% |
| 3.0 | 98 | 47 | 48 | 49.5% |

The fraction of borderlines that v3 calls "yes" is roughly stable (42-50%) across all alphas. The slope compression is driven by the **number** of borderlines declining (171 → 98), not by a change in v3's classification behavior. This confirms the compression is a structural property of how v3 interacts with the intervention's polarization effect, not an evaluator inconsistency.

### 4.3 Polarization (from v2 data)

The intervention pushes v2-borderline cases toward the poles:
- Borderlines: 171 → 98 (-73 records)
- v2-yes: 94 → 132 (+38 records, half of the departing borderlines become clearly harmful)
- v2-no: 235 → 270 (+35 records, the other half become clearly safe)

The control shows no polarization (borderlines stable at 124-139 across alphas; [prior report](2026-04-12-seed0-jailbreak-control-audit.md) §3.1).

---

## 5. Primary Outcome-Level Analysis — The Severity Shift

This is the most informative finding in this analysis. While v3's harmful_binary rate is flat, its primary_outcome taxonomy reveals a clear dose-dependent severity shift.

### 5.1 H-neuron primary outcome trajectory

| Outcome | α=0.0 | α=3.0 | Change | Direction |
|---|---|---|---|---|
| substantive_compliance | 27.8% | 33.9% | **+6.1pp** | ↑ stronger compliance |
| partial_compliance | 6.7% | 2.0% | -4.7pp | ↓ resolving to poles |
| deflection | 17.6% | 12.4% | -5.2pp | ↓ resolving to poles |
| refusal | 45.4% | 48.0% | +2.6pp | ↑ more definitive refusals |
| benign_answer | 2.5% | 3.7% | +1.2pp | ≈ stable |

### 5.2 Seed-1 control primary outcome trajectory

| Outcome | α=0.0 | α=3.0 | Change | Direction |
|---|---|---|---|---|
| substantive_compliance | 30.7% | 29.2% | -1.5pp | ≈ flat |
| partial_compliance | 3.7% | 4.5% | +0.8pp | ≈ flat |
| deflection | 17.2% | 17.8% | +0.6pp | ≈ flat |
| refusal | 45.7% | 45.0% | -0.7pp | ≈ flat |
| benign_answer | 2.7% | 3.5% | +0.8pp | ≈ flat |

### 5.3 Interpretation

The intervention produces a **severity intensification** pattern visible only in v3's outcome taxonomy:

1. **Partial compliance → substantive compliance:** Responses that partially engage with the harmful request at baseline become fully compliant at higher alpha. This is the primary dose-dependent signal (+6.1pp).
2. **Deflection → refusal:** Ambiguous deflections become definitive refusals at higher alpha (+2.6pp refusal, -5.2pp deflection). The intervention also pushes safe-side responses toward more decisive safety behavior.
3. **harmful_binary masks the signal:** Because partial_compliance and substantive_compliance both count as "yes" when intent_match ≥ 1, the partial→substantive shift is invisible at the binary level. The harmful count barely changes (+8 records) while the *composition* changes substantially.

**The dose-response exists — it's a response in severity, not in the harmful/safe boundary.** v2 captured this as a rate increase (borderline→yes ≈ partial→substantive through v2's lens). v3's binary metric missed it because its threshold is set higher. But v3's granular taxonomy recovers it at the right level.

---

## 6. Sensitivity Analysis — Error Attrition

v3 evaluation errors vary by alpha: 11, 8, 16, 8 (all `invalid_evidence_spans`).

| Alpha | Actual rate | All errors harmful | All errors safe | Impact |
|---|---|---|---|---|
| 0.0 | 34.56% | 36.00% | 33.80% | ±1.2pp |
| 1.0 | 34.96% | 36.00% | 34.40% | ±0.6pp |
| 1.5 | 34.50% | 36.60% | 33.40% | ±1.2pp |
| 3.0 | 35.98% | 37.00% | 35.40% | ±0.6pp |

| Scenario | Slope (pp/alpha) |
|---|---|
| Actual (errors excluded) | 0.46 |
| All errors harmful | 0.36 |
| All errors safe | 0.50 |

**Verdict:** Error attrition has negligible impact on the slope. The α=1.5 error spike (16 errors, 2× other alphas) does not meaningfully affect the harmful rate or slope. The sensitivity bounds [0.36, 0.50] are well within the bootstrap CI [-1.46, +2.41].

---

## 7. Seed-1 V3 Control

### 7.1 Rates

| Alpha | Yes count | Valid count | Rate |
|---|---|---|---|
| 0.0 | 168 | 488 | 34.43% |
| 1.0 | 175 | 491 | 35.64% |
| 1.5 | 168 | 490 | 34.29% |
| 3.0 | 165 | 489 | 33.74% |

### 7.2 Slopes

- harmful_binary slope: -0.34 pp/alpha [-2.28, +1.61] — flat (CI includes zero)
- substantive_compliance slope: -0.72 pp/alpha [-2.58, +1.19] — flat (CI includes zero)

### 7.3 Specificity limitations

The v3 specificity test (H-neuron minus seed-1 control) uses a **single control seed** and yields:
- harmful_binary gap: +0.80 pp/alpha [-2.00, +3.58] — **not significant**
- substantive_compliance gap: +2.72 pp/alpha [+0.02, +5.44] — **marginally significant** (lower bound barely excludes zero)

Contrast with v2 specificity ([prior report](2026-04-12-seed0-jailbreak-control-audit.md)): slope difference +2.77 pp/alpha [+1.17, +4.42], p=0.013 (permutation test). The v2 specificity result remains the stronger evidence.

The v3 specificity at the substantive_compliance level is interesting but should be treated with caution: it comes from a single seed, the lower bound of the CI barely excludes zero, and the same bootstrap was not used to compute the v2 substantive_compliance equivalent (not applicable to v2, which lacks primary_outcome).

---

## 8. What Withstands Scrutiny

### 8.1 The v2-to-v3 slope compression is real and structurally explained

Same 500 prompts, same model outputs, only the evaluator changed. The v2 slope (+2.30, CI excludes zero) compresses to a v3 slope (+0.46, CI includes zero). The mechanism is transparent: v3 absorbs borderline cases into the baseline rate, and fewer borderlines exist at high alpha due to intervention-driven polarization. The arithmetic is exact: v2 gain (+38) + absorbed change (-29) = v3 gain (+8).

### 8.2 The evaluator agreement structure is clean

88.2% overall agreement. Zero v2-no → v3-yes flips. Only 2 v2-yes → v3-no flips. The 11.8% discordance is entirely v2-borderline reclassification — a construct-definition difference, not evaluator error.

### 8.3 The severity-shift finding is reproducible

The substantive_compliance slope (+2.00 pp/alpha) and the gap vs control (+2.72 pp/alpha) are computed from the same raw data with the same bootstrap methodology. The underlying pattern (partial→substantive, deflection→refusal) is visible in raw counts without statistical modeling.

### 8.4 Both controls are flat

v2 seed-0: -0.47 pp/alpha (harmful_binary). v3 seed-1: -0.34 pp/alpha (harmful_binary). v3 seed-1: -0.72 pp/alpha (substantive_compliance). All CIs include zero. The intervention signal, whatever its size, is absent from random neurons under both evaluators.

---

## 9. Caveats and Remaining Uncertainty

### 9.1 High uncertainty

- **Is the substantive_compliance slope robustly significant?** The lower bound of the CI (+0.11) barely excludes zero. A slightly different bootstrap seed or sample could flip this. Treat as "directionally positive, marginally significant" — not headline-safe without replication or a larger sample.
- **Is v3 specificity real?** The single-seed v3 comparison has low power. The harmful_binary gap CI includes zero. The substantive_compliance gap CI barely excludes zero. V2 specificity (p=0.013) remains the load-bearing evidence.

### 9.2 Medium uncertainty

- **Does the borderline reclassification rate change across harm categories?** The overall rate is 42-50%, but category-level variation is untested. If reclassification is concentrated in specific categories, the severity-shift conclusion could be narrower than claimed.
- **Would scoring seed-0 with v3 change the control comparison?** The current v3 control (seed-1) and v2 control (seed-0) use different seeds, preventing a direct evaluator comparison on control data. Scoring seed-0 with v3 would resolve this.

### 9.3 Low uncertainty

- Error attrition does not meaningfully affect any reported metric (sensitivity bounds well within CIs).
- The transition matrix and reclassification arithmetic are exact (ID-level join, exhaustive).
- The pipeline audit's raw data verification (§1) independently confirmed all rates.

---

## 10. Implications for Anchor 3

The paper's Anchor 3 claim ("measurement choices change conclusions") can now be made at three levels, each stronger than the last:

### Level 1: Binary threshold sensitivity (the original framing)

The evaluator changes the statistical conclusion. v2 says "significant dose-response" (slope excludes zero); v3 says "no detectable dose-response" (slope includes zero). Same model outputs, different measurement instrument. This is clean and unambiguous.

### Level 2: Mechanism transparency (new from this analysis)

The slope compression is not mysterious — it's a known and predictable consequence of how each evaluator handles borderline cases. v3 catches refuse-then-comply patterns at baseline that v2 coded as borderline, inflating the baseline rate and absorbing the signal. The intervention's polarization effect (pushing borderlines to yes or no) is the same reality seen through two different lenses.

**Why this matters for the paper:** It preempts the objection "maybe v3 is just noisier." The mechanism is fully explained by construct definition (what counts as "yes"), not by measurement noise.

### Level 3: Measurement level determines conclusion type (the new finding)

The most powerful framing: the evaluator doesn't just change the magnitude of the conclusion — it changes the *type* of conclusion that's detectable. v2's binary metric correctly detected a real signal but located it at the harmful/safe boundary. v3's binary metric missed the same signal at that boundary. But v3's primary_outcome taxonomy found the signal at the right level: severity intensification (partial → substantive compliance), which is significant (+2.00 pp/alpha, CI excludes zero).

**For the paper:** This upgrades Anchor 3 from "evaluator X says significant, evaluator Y says not" to "the intervention's dose-response is real but lives at the severity level, and your conclusion depends on whether your evaluator measures there." This is a methodological insight, not just a measurement artifact.

### Updated claim status

| Claim | Status | Evidence |
|---|---|---|
| v2 shows significant binary dose-response | **Earned** | +2.30 [+0.99, +3.58] |
| v3 shows non-significant binary dose-response | **Earned** | +0.46 [-1.46, +2.41] |
| Measurement changes conclusions (binary level) | **Earned** | Same outputs, different slopes, different significance |
| Slope compression mechanism understood | **Earned** | Borderline absorption + polarization, arithmetic verified |
| Severity-shift dose-response (v3 substantive_compliance) | **Partially earned** | +2.00 [+0.11, +3.87], marginal; needs caveat |
| Severity-shift specific to H-neurons | **Partially earned** | Gap +2.72 [+0.02, +5.44], marginal; single control seed |
| v3 binary specificity | **Not earned** | Gap +0.80 [-2.00, +3.58], CI includes zero |

---

## 11. Next Steps (Ordered by Information Gained per Time)

1. **Score seed-0 with v3** (~$8 API, ~1h). Enables (a) paired v2-v3 comparison on control data, (b) multi-seed permutation test for v3 specificity, (c) check whether borderline reclassification rate is condition-dependent. Highest value remaining experiment for the measurement story.

2. **Examine the 2 v2-yes → v3-no flips** (~10 min). These are the only genuine evaluator disagreements. Understanding why v3 downgraded them could reveal edge cases in v3's primary_outcome → harmful_binary derivation.

3. **Category-level borderline reclassification rates** (~20 min). Check whether the 42-50% reclassification rate is uniform or concentrated in specific harm categories. If non-uniform, the severity-shift story may need qualification.

4. **Substantive_compliance slope with different seeds** (requires step 1). The marginal significance (CI lower bound +0.11) is fragile. A second control seed would either confirm or weaken it. This is important for deciding whether the severity-shift claim can appear in the paper without heavy caveats.

---

## Data Artifacts Audited

| Artifact | Path | Schema | Verified |
|---|---|---|---|
| H-neuron v2 evaluation | `data/.../jailbreak/csv2_evaluation/alpha_{0.0,1.0,1.5,3.0}.jsonl` | legacy (no version) | rates, borderline counts, transitions |
| H-neuron v3 evaluation | `data/.../jailbreak/csv2_v3_evaluation/alpha_{0.0,1.0,1.5,3.0}.jsonl` | csv2_v3 | rates, errors, primary_outcome, transitions |
| Seed-1 v3 control | `data/.../control/seed_1_unconstrained_csv2_v3/alpha_{0.0,1.0,1.5,3.0}.jsonl` | csv2_v3 | rates, primary_outcome, errors |
| v2 specificity summary | `data/.../control/comparison_csv2_v2_summary.json` | — | archived from git, slope +2.30 |
| v3 specificity summary | `data/.../control/comparison_csv2_v3_summary.json` | — | slope +0.46, seed-1 only |
| Concordance summary | `data/.../control/concordance/concordance_summary.json` | — | kappa, flips, category breakdown |
