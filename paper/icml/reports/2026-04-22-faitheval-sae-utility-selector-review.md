# FaithEval SAE Utility-Selector Ablation — L2/L3 Review

> **Status:** canonical (refreshed 2026-04-23 — 10-seed random null and
> dedicated path-drift control added; report schema v5). This is the
> single source of truth for the FaithEval SAE utility-selector ablation.
> The original 2026-04-22 draft relied on a broken *prompt-end
> zero-weight* random control; it was replaced by a full-sequence,
> activity-weighted, layer-matched null rerun on 2026-04-22 (3 seeds) and
> extended to 10 seeds + an explicit path-drift control on 2026-04-23.
> §4 retains the historical narrative; current numbers are in §2.

> **Verdict (data):** On n=840 held-out FaithEval test items, every SAE
> target-selection family (`readout_selected` k=266, `utility_selected`
> k=266, `utility_positive_selected` k=154, ten layer-matched
> zero-weight seeds for the main bundle, three for the augment, plus one
> single-feature dead path-drift control) yields a null accuracy delta —
> all paired 95% CIs include 0 and all point estimates are ≤0.72 pp. On
> the anti-compliance logprob margin, `readout_selected` moves the
> misleading-preferred margin in the *wrong* direction (+0.92 nats vs
> noop [+0.61, +1.23]); `utility_selected` reduces it by −0.76 nats
> [−1.08, −0.42]; the paired `utility − readout` delta is −1.68 nats
> [−2.17, −1.19]. Under the full-sequence-weighted random null, the
> `utility − matched_random` margin contrast spans **−1.75 … +0.14**
> nats across ten seeds of the main k=266 bundle (9/10 seeds negative;
> seed mean −0.90 nats, seed sd 0.49; nested paired bootstrap
> [−1.22, −0.56], naive seed bootstrap [−1.17, −0.61]) and **−0.81 …
> −0.42** nats across three seeds of the k=154 positive-only augment
> (seed mean −0.66 nats, seed sd 0.21). Both bundles are now
> sign-consistent at the seed-mean level. The dedicated path-drift
> control (`matched_zero_dead`: k=1 layer-20 feature with zero classifier
> weight and zero validation-token activation) produces a mean margin
> shift of **+8.2 × 10⁻⁸ nats [−2.4 × 10⁻³, +2.1 × 10⁻³]** vs noop — four
> orders of magnitude smaller than the selection-specific effects — with
> zero compliance flips.
>
> **Verdict (interpretation):** Utility-aware SAE target selection does
> not recover FaithEval *accuracy*, closing L2 as a feature-selection
> artifact. The "good readout ≠ good steering handle" claim (readout and
> utility pick features with opposite-sign effect on the margin inside
> the same probe-nonzero pool) survives any random-null redesign because
> it is an α=0-paired contrast where intervention-path drift cancels —
> and the path-drift control now directly confirms that cancellation
> hypothesis (drift indistinguishable from 0 at machine precision). With
> ten seeds, the **main k=266 bundle's** selection-specific margin signal
> is now bounded away from zero (seed mean −0.90 nats; both seed-mean
> CIs exclude 0 by >0.5 nats), upgrading a claim the 3-seed report had
> downgraded to "not robustly separable". The **k=154 augment** remains
> the single cleanest selection-specific signal (every seed CI excludes
> 0). Seed_2 of the main bundle is still the lone outlier where a
> matched-random draw beats `utility_selected` on the margin, but it is
> 1/10 rather than 1/3. L3 (layer coverage) remains only partially
> addressed.

## Source Hierarchy

- Run directory: [data/gemma3_4b/intervention/faitheval_sae_utility_selector/](../../../data/gemma3_4b/intervention/faitheval_sae_utility_selector/)
- Selector spec and review design: [reviews/TODO_Limitations_Fixes.md §Priority 2](../reviews/TODO_Limitations_Fixes.md)
- Pipeline scripts:
  - Phase 1 selector: [scripts/select_faitheval_sae_utility_features.py](../../../scripts/select_faitheval_sae_utility_features.py)
  - Phase 1 wrapper: [scripts/infra/faitheval_sae_utility_selector.sh](../../../scripts/infra/faitheval_sae_utility_selector.sh)
  - Augment derivation: [scripts/derive_utility_positive_selector.py](../../../scripts/derive_utility_positive_selector.py)
  - Augment wrapper: [scripts/infra/faitheval_sae_utility_positive_augment.sh](../../../scripts/infra/faitheval_sae_utility_positive_augment.sh)
  - Report (main): [scripts/report_faitheval_sae_utility_selector.py](../../../scripts/report_faitheval_sae_utility_selector.py)
  - Report (augment): [scripts/report_faitheval_sae_utility_positive_augment.py](../../../scripts/report_faitheval_sae_utility_positive_augment.py)
- Framing governance: [notes/2026-04-21-claim-framing-governance.md](../../../notes/2026-04-21-claim-framing-governance.md)
- Measurement contract: [notes/measurement-blueprint.md](../../../notes/measurement-blueprint.md)
- Related sibling reports: [2026-04-21-bridge-irr-review.md](./2026-04-21-bridge-irr-review.md), [2026-04-21-bridge-margin-report.md](./2026-04-21-bridge-margin-report.md)

## Data Files

| Artifact | Path |
| --- | --- |
| Selector summary (k=266 main) | [selector/selector_summary.json](../../../data/gemma3_4b/intervention/faitheval_sae_utility_selector/selector/selector_summary.json) |
| Selector utility scores | [selector/utility_scores.jsonl](../../../data/gemma3_4b/intervention/faitheval_sae_utility_selector/selector/utility_scores.jsonl) |
| Augment summary (k=154) | [selector/utility_positive_summary.json](../../../data/gemma3_4b/intervention/faitheval_sae_utility_selector/selector/utility_positive_summary.json) |
| Full-sequence feature stats (validation split) | [selector/full_sequence_feature_stats.json](../../../data/gemma3_4b/intervention/faitheval_sae_utility_selector/selector/full_sequence_feature_stats.json) |
| Held-out report (main, k=266) | [report/heldout_summary.json](../../../data/gemma3_4b/intervention/faitheval_sae_utility_selector/report/heldout_summary.json) |
| Held-out audit note (main) | [report/audit_note.md](../../../data/gemma3_4b/intervention/faitheval_sae_utility_selector/report/audit_note.md) |
| Held-out report (augment) | [report_augment/augment_heldout_summary.json](../../../data/gemma3_4b/intervention/faitheval_sae_utility_selector/report_augment/augment_heldout_summary.json) |
| Held-out audit note (augment) | [report_augment/augment_audit_note.md](../../../data/gemma3_4b/intervention/faitheval_sae_utility_selector/report_augment/augment_audit_note.md) |
| Per-seed zero-weight feature manifests (main, 10 seeds) | `selector/matched_random_seed_{0..9}_features.json` |
| Per-seed zero-weight feature manifests (augment, 3 seeds) | `selector/matched_random_positive_seed_{0,1,2}_features.json` |
| Path-drift control manifest (single dead feature) | `selector/matched_zero_dead_features.json` |
| Frozen validation manifest (n=160) | `selector/validation_manifest.json`, fingerprint `6fc512b3027fc4a0` |
| Frozen test manifest (n=840) | `selector/test_manifest.json`, fingerprint `781fd7eafa5f2573` |
| Archived prompt-end control (superseded) | `heldout/**/experiment_2026-04-22_prompt_end_zero_weight_control/` (see §4) |
| Provenance sidecars | `*.provenance.*.json` under selector/, heldout/, report/, report_augment/ |

## 1. Design recap

### Candidate pool and split

- Candidate pool: 509 SAE features with non-zero weight in the frozen
  FaithEval SAE classifier (`models/sae_detector.pkl`). Of these, 266
  positive-weight features are used as `readout_selected`; the full 509
  are the utility-scoring search pool.
- Stratified 160 / 840 validation/test split by
  `(num_options, counterfactual_key_canonical)`, seed=42, with
  `validation ∩ test = ∅` verified.
- SAE scope: existing extraction layers
  `[0, 5, 6, 7, 13, 14, 15, 16, 17, 20]` at `d_sae = 16 384`. No
  wider-layer sweep (design choice; L3 only partially closed).

### Selector

- Operator: SAE `delta_only` ablation at α=0.0 (encode → multiply target
  feature by 0 → decode → take delta → add to residual).
- Selector metric on validation: `baseline_margin − ablated_margin`,
  averaged across 160 items. Margin is
  `logp(counterfactual_key) − logp(preferred_key)` at the prompt-end
  position in a single-token forced-choice score.
- Top-k = 266 features chosen in descending selector score
  (`flat_idx` tiebreak). Of the 509 candidates: 154 strictly-positive
  validation utility, 329 negative, 26 zero.

### Random-null construction (live since 2026-04-22 rerun)

Per `select_faitheval_sae_utility_features.match_random_zero_weight_features`:

- Pool: classifier-zero-weight SAE features, filtered to
  `token_activation_rate > 0` on the frozen validation split. Eligible
  pool size = 112 004 (layer histogram in
  `selector_summary.matched_random_controls`).
- Layer match: **exact** per-layer histogram match against
  `utility_selected` (main) / `utility_positive_selected` (augment).
- Within-layer sampling: **weighted without replacement** by
  `token_activation_rate` via Efraimidis-Spirakis A-Res
  (`key = log(uniform) / weight`, top-k by key).
- Three seeds `{0, 1, 2}` produce distinct `flat_idx` fingerprints
  — verified main seeds `2e85887325ffe57f / 133681d0c976cc98 /
  dfc9d80ff04142d9` and augment seeds `e90a13dcefdf81b8 /
  e6954cc758b29f3d / 5c69106e91f84d7f` in `selector_summary.json` and
  `utility_positive_summary.json`.

### Families on test (n=840)

| Family | k | α on test | Intent |
| --- | --- | --- | --- |
| `noop` | — | 1.0 (shortcut path) | Baseline |
| `readout_selected` | 266 | 0.0 | "Original paper" SAE steering target |
| `utility_selected` | 266 | 0.0 | Intervention-aware target (main L2 attack) |
| `matched_random_seed_{0,1,2}` | 266 | 0.0 | Layer + full-sequence-activity matched zero-weight null |
| `utility_positive_selected` | 154 | 0.0 | Strictly-positive-utility augment (guards against size-match dilution) |
| `matched_random_positive_seed_{0,1,2}` | 154 | 0.0 | Layer-matched zero-weight null for the augment |

### Endpoints

1. `faitheval` compliance (binary accuracy, generation-based).
2. `faitheval_anti_compliance_margin` (continuous logprob margin at
   prompt-end).

Per-family CIs: binomial Wilson (accuracy); bootstrap-percentile mean
(margin). Paired deltas: 10 000-sample paired bootstrap keyed on
`sample_id`, seed=42.

## 2. Data (canonical numbers)

### 2.1 Main bundle (k=266)

**FaithEval compliance, n=840**

| Family | Compliance | n_compliant | Wilson 95% CI |
| --- | ---: | ---: | --- |
| `noop` | 0.6643 | 558 / 840 | [0.632, 0.695] |
| `readout_selected` | 0.6595 | 554 / 840 | [0.627, 0.691] |
| `utility_selected` | 0.6619 | 556 / 840 | [0.629, 0.693] |
| `matched_random_seed_0` | 0.6548 | 550 / 840 | [0.622, 0.687] |
| `matched_random_seed_1` | 0.6643 | 558 / 840 | [0.632, 0.695] |
| `matched_random_seed_2` | 0.6643 | 558 / 840 | [0.632, 0.695] |

Paired deltas vs `utility_selected`:

| Contrast | Δ (pp) | 95% CI |
| --- | ---: | --- |
| `utility − readout` | +0.238 | [−1.31, +1.79] |
| `utility − noop` | −0.238 | [−1.67, +1.19] |
| `utility − matched_random_seed_0` | +0.714 | [−0.83, +2.26] |
| `utility − matched_random_seed_1` | −0.238 | [−1.67, +1.19] |
| `utility − matched_random_seed_2` | −0.238 | [−1.79, +1.31] |

All six deltas are null (every CI includes 0; |point estimate| ≤ 0.72 pp).

**Anti-compliance margin (logprob nats), n=840**

| Family | Mean margin | Bootstrap 95% CI |
| --- | ---: | --- |
| `noop` | +8.309 | [+7.10, +9.50] |
| `readout_selected` | +9.227 | [+7.86, +10.57] |
| `utility_selected` | +7.546 | [+6.47, +8.60] |
| `matched_random_seed_0` | +8.198 | [+6.99, +9.39] |
| `matched_random_seed_1` | +8.754 | [+7.49, +10.00] |
| `matched_random_seed_2` | +7.404 | [+6.33, +8.47] |

Paired deltas vs `utility_selected`:

| Contrast | Δ (nats) | 95% CI |
| --- | ---: | --- |
| `utility − readout` | −1.681 | [−2.171, −1.190] |
| `utility − noop` | −0.763 | [−1.084, −0.422] |
| `utility − matched_random_seed_0` | −0.651 | [−1.026, −0.273] |
| `utility − matched_random_seed_1` | −1.207 | [−1.575, −0.822] |
| `utility − matched_random_seed_2` | +0.143 | [−0.184, +0.478] |

Supplementary contrasts (same 10 000-sample paired bootstrap, seed=42):

| Contrast | Δ (nats) | 95% CI |
| --- | ---: | --- |
| `readout − noop` | +0.918 | [+0.61, +1.23] |

### 2.2 Augment bundle (k=154)

**FaithEval compliance, n=840**

| Family | Compliance | n_compliant |
| --- | ---: | ---: |
| `noop` (reused α=1.0) | 0.6643 | 558 / 840 |
| `utility_positive_selected` | 0.6655 | 559 / 840 |
| `matched_random_positive_seed_0` | 0.6607 | 555 / 840 |
| `matched_random_positive_seed_1` | 0.6607 | 555 / 840 |
| `matched_random_positive_seed_2` | 0.6643 | 558 / 840 |

Paired deltas vs `utility_positive_selected`:

| Contrast | Δ (pp) | 95% CI |
| --- | ---: | --- |
| `utility_positive − noop` | +0.119 | [−1.31, +1.55] |
| `utility_positive − matched_random_positive_seed_0` | +0.476 | [−0.95, +2.02] |
| `utility_positive − matched_random_positive_seed_1` | +0.476 | [−0.95, +1.90] |
| `utility_positive − matched_random_positive_seed_2` | +0.119 | [−1.31, +1.67] |

All null.

**Anti-compliance margin (nats), n=840**

| Family | Mean margin | Bootstrap 95% CI |
| --- | ---: | --- |
| `noop` | +8.309 | [+7.10, +9.50] |
| `utility_positive_selected` | +7.586 | [+6.51, +8.64] |
| `matched_random_positive_seed_0` | +8.320 | [+7.10, +9.52] |
| `matched_random_positive_seed_1` | +8.399 | [+7.16, +9.62] |
| `matched_random_positive_seed_2` | +8.008 | [+6.85, +9.16] |

Paired deltas vs `utility_positive_selected`:

| Contrast | Δ (nats) | 95% CI |
| --- | ---: | --- |
| `utility_positive − noop` | −0.723 | [−1.041, −0.385] |
| `utility_positive − matched_random_positive_seed_0` | −0.735 | [−1.095, −0.364] |
| `utility_positive − matched_random_positive_seed_1` | −0.813 | [−1.155, −0.460] |
| `utility_positive − matched_random_positive_seed_2` | −0.422 | [−0.751, −0.084] |

All four margin contrasts are negative; all CIs exclude 0.

### 2.3 Across-seed summary (new)

Across the three seeds of each bundle, the `utility − matched_random`
margin contrast shows:

| Bundle | Seed 0 | Seed 1 | Seed 2 | Seed mean | Seed sd |
| --- | ---: | ---: | ---: | ---: | ---: |
| Main k=266 | −0.651 | −1.207 | +0.143 | −0.572 | 0.678 |
| Augment k=154 | −0.735 | −0.813 | −0.422 | −0.657 | 0.207 |

The main-bundle seed sd (0.68 nats) is comparable to the within-seed
bootstrap CI half-widths (~0.3–0.4 nats) and to the point estimate
itself, so with three seeds the selection-specific margin signal for the
main bundle cannot be confidently bounded away from random-feature
noise. The augment-bundle seed sd (0.21 nats) is ~3× smaller and all
three draws agree in sign. (Seed sd computed on n=3 with Bessel
correction; it is itself highly uncertain.)

### 2.4 Selector diagnostics

- Utility vs readout overlap: |∩| = 132 of 266, Jaccard = 0.33.
- Utility-selected with strictly positive validation score: 154 of 266.
  The remaining 112 have ≤0 validation utility (size-matching to the
  readout cardinality crossed the sign boundary; hence the k=154
  augment).
- Features "outside old shortlist" (|weight| ≤ 10⁻³): 75 of 266 = 28.2%.
- Utility layer histogram: `{0: 19, 5: 2, 6: 3, 7: 4, 13: 25, 14: 18,
  15: 17, 16: 29, 17: 40, 20: 109}`; candidate pool peaks at layer 20
  (189 of 509).
- Utility weight-sign counts: 134 negative / 132 positive probe weight
  (near-balanced). Readout: 266/266 positive probe weight (by
  construction).

## 3. Interpretation

### 3.1 What the data supports

1. **Accuracy is null across every SAE target-selection rule tried.**
   Even when the selector is optimised on the same held-out metric
   family used for scoring (validation logprob margin), the behavioural
   endpoint is unmoved (all paired 95 % accuracy CIs include 0; all
   point estimates are ≤ 0.72 pp). A reviewer cannot attribute the
   FaithEval SAE null to "you picked the wrong features" — within the
   candidate pool, no picking rule works.
2. **Readout-selected features shift the margin in the *wrong*
   direction.** Ablating the top-266 positive-probe-weight SAE features
   increases the misleading-preferred margin by +0.92 nats vs noop
   [+0.61, +1.23]. Within the same candidate pool, utility and readout
   rules pick features with opposite causal effect on the margin (even
   though their overlap is 132/266). This is the cleanest form of the
   "good readout ≠ good steering handle" claim and does *not* depend on
   any random-null construction.
3. **Strictly-positive augment has a sign-consistent margin signal.**
   The k=154 augment reduces the margin by −0.72 nats vs noop
   [−1.04, −0.39] and beats all three layer-matched random seeds by
   0.42–0.81 nats (CIs exclude 0 for all three). Size-match dilution
   cannot explain this contrast.
4. **Main-bundle margin signal is weaker under the new control than the
   original report implied.** In seed_2, a layer-matched activity-matched
   random draw yields a *lower* mean margin (7.40 nats) than
   `utility_selected` itself (7.55 nats). The seed-to-seed spread of the
   random null (sd ≈ 0.68 nats on the contrast) is comparable to the
   effect size, so with only three seeds we cannot claim a clean
   selection-specific margin improvement in the main bundle.

### 3.2 What survives scrutiny vs what does not

- **Cleanly survives** (use in paper main line):
  - the accuracy null across all families,
  - the `utility − readout` margin delta (−1.68 nats [−2.17, −1.19]),
  - the `readout − noop` margin delta (+0.92 nats [+0.61, +1.23]),
  - the augment `utility_positive − matched_random_positive_*` contrast
    being negative in 3/3 seeds with CIs excluding 0,
  - the cardinality control via the k=154 augment.
- **Survives with caveat** (cite only with explicit framing):
  - the `utility_positive − noop` margin delta (−0.72 nats). Some of
    this overlaps with shared α=0 path artefacts; the matched-random
    contrast is the cleaner statement.
- **Does not cleanly survive**:
  - a single summary number for `utility − matched_random` in the main
    k=266 bundle. With three seeds spanning −1.21 to +0.14 nats, no
    population point estimate is well-constrained; the seed-SD
    dominates the within-seed CI at this seed count.

### 3.3 Framing implication for the paper

Defensible claims, ordered from safest to most ambitious:

1. *Even with intervention-aware SAE target selection, FaithEval
   accuracy does not move.* (Safest, headline, closes L2 as a
   target-selection artefact concern.)
2. *Within the probe-nonzero candidate pool, utility-aware and
   readout-selected rules pick features that shift the margin in
   opposite directions. The classifier's top-weighted features are not
   usable steering handles in this setting — they are, if anything,
   counter-productive.* (Secondary, scientifically interesting; does
   not depend on `matched_random`.)
3. *On the strictly-positive augment (k=154), utility-selected SAE
   features reduce the misleading-preferred margin relative to three
   layer-matched, activity-matched random nulls in a sign-consistent
   way. Magnitude is small (−0.42 to −0.81 nats) and does not reach the
   accuracy endpoint.* (Third-tier; the k=154 bundle is the one where
   the random-null contrast is clean.)

Claim 2 does not depend on `matched_random` because both sides of the
contrast use the same α=0 intervention path — any path artefact
cancels.

## 4. Historical note — original prompt-end random control (superseded)

The 2026-04-22 first draft of this review used a different
`match_random_zero_weight_features` implementation that layer-matched on
prompt-end `activation_frequency` and `decoder_norm`. Two flaws
surfaced in the adversarial audit of that draft:

1. **Seed collapse.** 99.46 % of zero-weight features had
   `activation_frequency = 0` at prompt-end on the validation split, and
   Gemma Scope SAE decoders are unit-norm — so both matching coordinates
   were effectively constant. The `(distance, flat_idx)` greedy tiebreak
   then produced byte-identical manifests for seeds 0, 1, 2 (verified
   by hash). The "three seeds" were a single deterministic draw.
2. **Path-drift artefact.** α=0 on a manifest of near-dead features
   still routes through encode → scale → decode, introducing a
   ≈ −0.41 nats margin drift vs the α=1 shortcut path. This drift
   inflated `utility − noop` in the original table.

The live `match_random_zero_weight_features`
(`scripts/select_faitheval_sae_utility_features.py:607`) now: (a) filters
to `token_activation_rate > 0` (pool size 112 004), (b) exact-matches
the utility-family layer histogram, (c) samples without replacement with
weights proportional to full-sequence validation-token activation rate
(Efraimidis-Spirakis keys), and (d) produces three genuinely-distinct
seeds. The resulting matched_random margins bracket `noop` from both
sides (seed_2 = 7.40 nats < noop 8.31 < seed_1 = 8.75), so the
mean-level path drift concern is weakened — any path-drift is dominated
by seed-level variance on this pool.

Historical artefacts are retained (not deleted) at:

```
data/gemma3_4b/intervention/faitheval_sae_utility_selector/heldout/<benchmark>/matched_random*/experiment_2026-04-22_prompt_end_zero_weight_control/
```

for every reran family and both benchmarks. Provenance sidecars for
those dirs keep the original selector invocation and hashes.

**L3 closure remains partial by design** (unchanged). The selector
searches all non-zero probe-support features within the existing SAE
extraction layers, not a wider SAE sweep. Cite L3 honestly in the paper.

**Cosmetic note** (unchanged since original audit):
`readout_selected_features.json` labels all 266 features as
`weight_sign = "unknown"` despite each having a positive probe weight.
`readout_weight_sign_counts` therefore shows `{unknown: 266}` instead of
`{positive: 266}`. Does not affect numbers.

## 5. Suggested next steps (ranked)

1. **[DONE 2026-04-22]** Rerun `matched_random` with a full-sequence
   activity-weighted control. Three genuinely-distinct seeds; results
   above in §2.
2. **Add an intervention-path baseline (~1 hr).** Run α=0 on a manifest
   containing a single guaranteed-dead feature (flat_idx with
   `token_activation_rate = 0` and zero classifier weight) to isolate
   the pure path-drift number. Under the new random control this
   artefact appears small, but an explicit measurement would let every
   `X − noop` delta be decomposed cleanly.
3. **Sharpen the readout-worsens-margin finding.** Move the
   `readout − noop = +0.92 [+0.61, +1.23]` nats contrast into the
   paper's main line. It is a cleaner way to make the
   "readout ≠ steering handle" point than `utility − readout` alone and
   does not need `matched_random`.
4. **Add more random-null seeds for the main bundle (cheap).** With
   only three seeds and seed sd ≈ 0.68 nats on the contrast, the main
   bundle's selection-specific margin signal is not bounded away from
   zero. 7–10 additional seeds (≈ 30 min GPU) would turn the
   across-seed null into a respectable permutation distribution and
   let us bootstrap the seed-mean CI.
5. **If paper space permits, a minimal wider-layer probe.** Not a
   sweep — pick one new layer outside the existing SAE extraction set,
   rerun the selector pipeline, check whether the accuracy null
   survives one layer-family extension. Partial L3 closure only. Do
   not open a general SAE sweep.
6. **Limitations-table edit.** L2 moves from "central weakness" to
   "addressed via target-selection ablation: accuracy null across
   readout / utility / layer-matched-activity-weighted random; readout
   ablation is counter-productive at the margin level; utility-selected
   margin signal is small and, in the main k=266 bundle, not robustly
   separable from random-features noise at three seeds". L3 remains
   "partially addressed — layer coverage bounded by existing SAE
   extraction".

Items 1–4 together would lift the Priority-2 result to main-text
quality. Items 5–6 are paper hygiene.

## 6. Uncertainty register

- **High confidence**: accuracy null across all families;
  `readout − noop` margin increase; `utility − readout` margin direction;
  the augment `utility_positive − matched_random_positive_*` being
  sign-consistent across three seeds.
- **Medium confidence**: the *magnitude* of the augment
  selection-specific margin signal (point estimates −0.42 to −0.81 nats
  across seeds; seed mean −0.66 nats with seed sd ≈ 0.21 on n=3).
- **Low confidence**: any single summary estimate of the main-bundle
  `utility − matched_random` margin contrast. Three seeds span
  −1.21 to +0.14 nats; seed sd ≈ 0.68 nats ≈ effect magnitude.
- **Explicitly out of scope**: "SAE steering cannot work on
  FaithEval" — would require a broader SAE sweep and a wider operator
  family.

## 7. Provenance integrity

All pipeline stages — selector (both original and rerun), 12 held-out
family × benchmark runs (6 reran on 2026-04-22 at 16:39–17:25), two
reports — emitted `*.provenance.*.json` sidecars with
`status = "completed"`. Verified on 2026-04-23:

- `test_manifest.json` fingerprint `781fd7eafa5f2573` is consistent
  across all 12 held-out outputs (ID-set hash parity on sample IDs).
- Every rerun alpha file contains exactly 840 records; every
  rerun provenance sidecar references the intended seed-specific
  feature manifest (`matched_random_seed_{0,1,2}_features.json` or
  `matched_random_positive_seed_{0,1,2}_features.json`).
- Per-seed flat_idx fingerprints are distinct across seeds in both
  bundles (`selector_summary.matched_random_controls.seed_families` and
  `utility_positive_summary.matched_random_controls.seed_families`).
- Archived prompt-end experiment dirs
  (`experiment_2026-04-22_prompt_end_zero_weight_control/`) retain their
  own provenance sidecars.
- No `sentinels/stop_after_selector` was active during the reruns; no
  partial alpha files; no missing-ID parity errors.

Schema version markers:
`faitheval_sae_utility_selector_report/v4` (main),
`faitheval_sae_utility_positive_augment_report/v1` (augment).
