# FaithEval SAE Utility-Selector Ablation — L2/L3 Review

> **Status:** canonical for the original `utility_selected` (k=266),
> `utility_positive` augment (k=154), 10-seed `matched_random`, and
> `matched_zero_dead` path-drift evidence (refreshed 2026-04-23). The
> 2026-04-25 answer-span pool extension lives in
> [2026-04-25-faitheval-answer-span-extension.md](./2026-04-25-faitheval-answer-span-extension.md)
> (selector-summary schema bumped to v6 there); read both reports
> together for the full L2 closure picture. Numbers in §2 below are
> unchanged by the extension. The original 2026-04-22 draft relied on a
> broken *prompt-end zero-weight* random control; it was replaced by a
> full-sequence, activity-weighted, layer-matched null rerun on
> 2026-04-22 (3 seeds) and extended to 10 seeds + an explicit path-drift
> control on 2026-04-23. §4 retains the historical narrative; current
> numbers for the utility / readout / matched_random / path-drift
> families are in §2.

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
- Answer-span pool extension (canonical for `answer_span_selected`, `matched_random_answer_span_*`, `faitheval_answer_span_margin`): [2026-04-25-faitheval-answer-span-extension.md](./2026-04-25-faitheval-answer-span-extension.md)

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

### Random-null construction (live since 2026-04-22 rerun, extended 2026-04-23)

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
- Ten seeds `{0..9}` for the main k=266 bundle produce distinct
  `flat_idx` fingerprints (see `selector_summary.matched_random_controls.seed_families`:
  `2e85887325ffe57f, 133681d0c976cc98, dfc9d80ff04142d9, 99156f41f2719149,
  67e06a2977813bcb, 3ad01f58c2147fa3, 658100286d3965dc, 963cad4de7a11994,
  3c1c1a4ae56d5067, 834c542e463b7ceb`). Pairwise seed-feature Jaccard
  overlaps are 0.019–0.051 (10–26 shared features of 266), confirming
  genuine draw-diversity against the 112 004-feature pool.
- Three seeds `{0, 1, 2}` for the k=154 augment (unchanged from
  2026-04-22): `e90a13dcefdf81b8 / e6954cc758b29f3d / 5c69106e91f84d7f`.

### Path-drift control (new 2026-04-23)

`matched_zero_dead` family: a single SAE feature selected by the
deterministic rule *smallest layer-20 `flat_idx` with zero classifier
weight **and** `token_activation_rate = 0` on the validation split*.
Concretely: `layer=20, feature=17, flat_idx=147 473, decoder_norm≈1.0,
token_active_count=0 / 75 497 validation tokens` (manifest
fingerprint `e5132cf1b03a0386`). Run with `intervention_mode=sae`,
`sae_steering_mode=delta_only`, α=0.0. By design, when the targeted
feature is truly dead, the decoded delta is zero and the encode/decode
roundtrip's reconstruction error cancels — so any residual `X − noop`
contrast measures only the pass-through drift of wiring the SAE hook
onto `post_feedforward_layernorm` and routing through bf16/float32.

### Families on test (n=840)

| Family | k | α on test | Intent |
| --- | --- | --- | --- |
| `noop` | — | 1.0 (shortcut path) | Baseline |
| `readout_selected` | 266 | 0.0 | "Original paper" SAE steering target |
| `utility_selected` | 266 | 0.0 | Intervention-aware target (main L2 attack) |
| `matched_random_seed_{0..9}` | 266 | 0.0 | Layer + full-sequence-activity matched zero-weight null (10 seeds) |
| `matched_zero_dead` | 1 | 0.0 | Path-drift control: single dead zero-weight feature |
| `utility_positive_selected` | 154 | 0.0 | Strictly-positive-utility augment (guards against size-match dilution) |
| `matched_random_positive_seed_{0,1,2}` | 154 | 0.0 | Layer-matched zero-weight null for the augment (3 seeds) |

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
| `matched_zero_dead` | 0.6643 | 558 / 840 | [0.632, 0.695] |
| `matched_random_seed_0` | 0.6548 | 550 / 840 | [0.622, 0.687] |
| `matched_random_seed_1` | 0.6643 | 558 / 840 | [0.632, 0.695] |
| `matched_random_seed_2` | 0.6643 | 558 / 840 | [0.632, 0.695] |
| `matched_random_seed_3` | 0.6583 | 553 / 840 | [0.626, 0.690] |
| `matched_random_seed_4` | 0.6631 | 557 / 840 | [0.630, 0.694] |
| `matched_random_seed_5` | 0.6643 | 558 / 840 | [0.632, 0.695] |
| `matched_random_seed_6` | 0.6619 | 556 / 840 | [0.629, 0.693] |
| `matched_random_seed_7` | 0.6571 | 552 / 840 | [0.624, 0.688] |
| `matched_random_seed_8` | 0.6702 | 563 / 840 | [0.638, 0.701] |
| `matched_random_seed_9` | 0.6786 | 570 / 840 | [0.646, 0.709] |

Paired deltas vs `utility_selected` (10 000-sample paired bootstrap, seed=42):

| Contrast | Δ (pp) | 95% CI |
| --- | ---: | --- |
| `utility − readout` | +0.238 | [−1.31, +1.79] |
| `utility − noop` | −0.238 | [−1.67, +1.19] |
| `utility − matched_random_seed_0` | +0.714 | [−0.83, +2.26] |
| `utility − matched_random_seed_1` | −0.238 | [−1.67, +1.19] |
| `utility − matched_random_seed_2` | −0.238 | [−1.79, +1.31] |
| `utility − matched_random_seed_3` | +0.357 | [−1.07, +1.79] |
| `utility − matched_random_seed_4` | −0.119 | [−1.43, +1.31] |
| `utility − matched_random_seed_5` | −0.238 | [−1.55, +1.07] |
| `utility − matched_random_seed_6` | 0.000 | [−1.19, +1.19] |
| `utility − matched_random_seed_7` | +0.476 | [−1.07, +2.02] |
| `utility − matched_random_seed_8` | −0.833 | [−2.26, +0.60] |
| `utility − matched_random_seed_9` | −1.667 | [−3.10, −0.24] |

All deltas except `utility − matched_random_seed_9` are null (CI crosses
0). seed_9 touches the threshold (upper CI −0.24 pp); this is one of 12
simultaneous contrasts, so cite only as a per-seed descriptive, not as
evidence of a population compliance effect.

**Anti-compliance margin (logprob nats), n=840**

| Family | Mean margin | Bootstrap 95% CI |
| --- | ---: | --- |
| `noop` | +8.309 | [+7.101, +9.499] |
| `readout_selected` | +9.227 | [+7.862, +10.573] |
| `utility_selected` | +7.546 | [+6.475, +8.598] |
| `matched_zero_dead` | +8.309 | [+7.102, +9.499] |
| `matched_random_seed_0` | +8.198 | [+6.986, +9.390] |
| `matched_random_seed_1` | +8.754 | [+7.485, +10.002] |
| `matched_random_seed_2` | +7.404 | [+6.330, +8.474] |
| `matched_random_seed_3` | +8.708 | [+7.461, +9.943] |
| `matched_random_seed_4` | +8.576 | [+7.368, +9.773] |
| `matched_random_seed_5` | +8.578 | [+7.313, +9.822] |
| `matched_random_seed_6` | +8.199 | [+7.016, +9.371] |
| `matched_random_seed_7` | +8.316 | [+7.067, +9.547] |
| `matched_random_seed_8` | +8.399 | [+7.197, +9.582] |
| `matched_random_seed_9` | +9.299 | [+8.024, +10.555] |

Paired deltas vs `utility_selected` (10 000-sample paired bootstrap, seed=42):

| Contrast | Δ (nats) | 95% CI |
| --- | ---: | --- |
| `utility − readout` | −1.681 | [−2.171, −1.190] |
| `utility − noop` | −0.763 | [−1.084, −0.422] |
| `utility − matched_random_seed_0` | −0.651 | [−1.026, −0.273] |
| `utility − matched_random_seed_1` | −1.207 | [−1.575, −0.822] |
| `utility − matched_random_seed_2` | +0.143 | [−0.184, +0.478] |
| `utility − matched_random_seed_3` | −1.162 | [−1.524, −0.777] |
| `utility − matched_random_seed_4` | −1.030 | [−1.378, −0.677] |
| `utility − matched_random_seed_5` | −1.032 | [−1.389, −0.665] |
| `utility − matched_random_seed_6` | −0.653 | [−0.961, −0.345] |
| `utility − matched_random_seed_7` | −0.770 | [−1.151, −0.368] |
| `utility − matched_random_seed_8` | −0.853 | [−1.201, −0.495] |
| `utility − matched_random_seed_9` | −1.753 | [−2.119, −1.371] |

9 of 10 seeds give a negative `utility − matched_random_seed_k` paired
delta; seed_2 is the only positive (opposite-sign) contrast. 9 of 10
per-seed bootstrap CIs exclude 0 (seed_2 CI [−0.18, +0.48] crosses 0).

Supplementary contrasts (same 10 000-sample paired bootstrap, seed=42):

| Contrast | Δ (nats) | 95% CI |
| --- | ---: | --- |
| `readout − noop` | +0.918 | [+0.61, +1.23] |
| `matched_zero_dead − noop` *(path drift)* | +8.2 × 10⁻⁸ | [−2.4 × 10⁻³, +2.1 × 10⁻³] |

See §2.4 for the path-drift analysis.

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

### 2.3 Across-seed summary

`utility − matched_random` margin contrast at the seed-mean level:

| Bundle | Seeds | Seed-mean | Seed SD | Seed range | Nested paired bootstrap CI¹ | Naive seed bootstrap CI² |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Main k=266 | 10 (`{0..9}`) | **−0.897** | 0.489 | [−1.753, +0.143] | [−1.218, −0.558] | [−1.174, −0.606] |
| Augment k=154 | 3 (`{0,1,2}`) | −0.657 | 0.207 | [−0.813, −0.422] | n/a (3-seed) | descriptive only³ |

¹ Nested paired bootstrap resamples test samples with replacement
jointly across `utility_selected` and all matched-random seed outputs;
the statistic is the seed-mean of the per-row (util − mean-of-seeds)
contrast. Resamples n=10 000, seed=42.

² Naive seed bootstrap resamples the per-seed point estimates i.i.d.
with replacement (k=10 or k=3). Treats seeds as the experimental unit.

³ With only 3 seeds, a naive seed bootstrap resamples a discrete
support of 3 values: the 2.5th–97.5th percentiles are tightly bounded
by the seed range and the statistic is not a calibrated coverage
statement. Treat the range [−0.813, −0.422] as descriptive, not as a
95% CI.

**Main bundle (k=266).** With 10 seeds (up from 3), the
seed-to-seed SD is 0.489 nats — roughly half the effect size
(0.897 nats), rather than comparable to it as with 3 seeds. 9 of 10
seeds agree in sign. Both seed-mean CI methods (nested paired,
naive seed) exclude zero by >0.5 nats. The main-bundle
selection-specific margin signal is now well-bounded, inverting the
2026-04-22 3-seed conclusion. (seed_2 remains the outlier that
underperforms `utility_selected` at +0.143 nats; the 10-seed
distribution dilutes its influence.)

**Augment bundle (k=154).** Unchanged since 2026-04-22. Sign-consistent
across all 3 seeds with seed SD 0.21 nats. The 3-seed "CI" equals the
seed range and is descriptive only. An extension to 10 seeds remains
open (see §5 item 7).

### 2.4 Path-drift control (`matched_zero_dead`)

Design target: measure the residual mean-level shift that arises purely
from registering the SAE hook, encoding → decoding, and routing the
post-layernorm activation through bf16/float32 conversions when the
targeted feature contributes zero to the decoded delta.

| Quantity | Value |
| --- | --- |
| Feature | layer=20, feature=17, `flat_idx`=147 473, `decoder_norm`≈1.0, `token_activation_rate`=0.0 (validation) |
| Family mean margin | +8.308 956 nats, bootstrap 95% CI [+7.102, +9.499] |
| `noop` mean margin (same split) | +8.308 956 nats, bootstrap 95% CI [+7.101, +9.499] |
| Paired `matched_zero_dead − noop` mean | **+8.22 × 10⁻⁸ nats** (paired bootstrap CI [−2.4 × 10⁻³, +2.1 × 10⁻³]) |
| Samples with `metric_value` difference ≠ 0 | 8 / 840 (0.95 %) |
| Difference distribution on those 8 samples | ±0.25 nats (6 cases), ±0.50 nats (2 cases); 5 positive / 3 negative; sum ≈ +6.9 × 10⁻⁵ nats |
| Compliance flips vs `noop` | 0 / 840 |

**Interpretation.** At the mean level, path drift is indistinguishable
from zero by any measure and is four orders of magnitude below the
`utility − noop` effect (0.76 nats) and the matched-random seed SD
(0.49 nats). The 8 per-sample deltas are quantised at ±0.25 / ±0.50
nats, consistent with bf16 mantissa precision of the softmax logits at
these magnitudes (the single-token forced-choice margin is computed in
bf16 before the log-softmax), and they cancel almost exactly in the
mean — so the observation is consistent with pure floating-point
rounding noise from the extra dtype round-trip in the SAE hook path,
not with a systematic intervention-path artifact.

**What this rules out.** The `utility_selected`, `readout_selected`,
and matched-random margin effects (all ≳0.5 nats) cannot be explained
by "the SAE code path being invoked at α=0 produces a mean margin shift
vs the α=1.0 shortcut". That hypothesis would predict a systematic
mean-level drift of the same order as the effects; we observe ~10⁻⁸
nats instead.

**What this does not control for.** When a targeted feature *does*
fire, `delta_only` mode still computes
`decode(f_modified) − decode(f)`; the reconstruction error of the
SAE on active features is shared across `decode(·)` calls but its
cancellation depends on numerical precision. This is the mechanism
by which matched-random seeds produce genuinely different mean margins
from each other (seed_2 = +7.40 nats vs seed_9 = +9.30 nats; range
≈1.9 nats, far above the path-drift scale). The `matched_random`
family is the right control for that component; `matched_zero_dead`
isolates the orthogonal "invocation-only" component.

### 2.5 Selector diagnostics

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
4. **Main-bundle margin signal is bounded away from zero at 10 seeds.**
   With the 10-seed extension (2026-04-23), the seed-mean
   `utility − matched_random` margin contrast is −0.897 nats with
   nested paired bootstrap CI [−1.22, −0.56] and naive seed bootstrap CI
   [−1.17, −0.61]. 9 of 10 seeds are negative; seed_2 is the lone
   outlier at +0.143 nats. Seed SD (0.489 nats) is now roughly half the
   effect size rather than comparable to it, and the within-seed
   per-sample paired bootstrap CIs exclude 0 for 9/10 seeds. This
   reverses the 3-seed downgrade from 2026-04-22: the main-bundle
   signal is real and sign-consistent, while remaining smaller and
   noisier than the augment signal.
5. **Path drift is empirically ~0.** The dedicated `matched_zero_dead`
   control (§2.4) measures the mean-level contribution of the SAE
   hook invocation when the targeted feature is truly dead. It is
   +8.2 × 10⁻⁸ nats [−2.4 × 10⁻³, +2.1 × 10⁻³]: four orders of
   magnitude below all selection-specific effects, and consistent with
   bf16 rounding on 1 % of samples. This removes "maybe it's just a
   code-path artifact" as an alternative explanation for the
   `utility`, `readout`, and `matched_random` mean margin shifts.

### 3.2 What survives scrutiny vs what does not

- **Cleanly survives** (use in paper main line):
  - the accuracy null across all families (including the 10-seed
    matched-random ensemble and the path-drift control),
  - the `utility − readout` margin delta (−1.68 nats [−2.17, −1.19]),
  - the `readout − noop` margin delta (+0.92 nats [+0.61, +1.23]),
  - the main-bundle seed-mean `utility − matched_random` contrast at
    10 seeds (−0.897 nats, nested CI [−1.22, −0.56], naive seed CI
    [−1.17, −0.61]) — this is the upgrade from the 2026-04-22 report,
  - the augment `utility_positive − matched_random_positive_*` contrast
    being negative in 3/3 seeds with CIs excluding 0,
  - the cardinality control via the k=154 augment,
  - the path-drift null (`matched_zero_dead − noop` at machine precision).
- **Survives with caveat** (cite only with explicit framing):
  - the `utility_positive − noop` margin delta (−0.72 nats) and the
    `utility − noop` margin delta (−0.76 nats). The path-drift control
    rules out the "α=0 SAE path alone shifts the margin" hypothesis,
    so both deltas now include only a "features-actively-firing"
    component; the matched-random contrast is still the cleaner
    statement because it also controls for "any 266 zero-weight
    layer-matched features shift the margin".
  - the seed_2 main-bundle outlier (+0.143 nats). It is 1/10 of the
    null distribution, not 1/3; cite it as a reminder that the
    matched-random pool is heterogeneous, not as evidence the main
    contrast is fragile.
- **Does not cleanly survive**:
  - any per-seed contrast treated in isolation as a "significant
    result" — there are 12 simultaneous paired contrasts per endpoint,
    and the Bonferroni-scaled per-seed CIs would be wider than
    reported. Always cite the seed-mean, not a cherry-picked per-seed
    number.
  - a seed-mean CI for the augment bundle. With 3 seeds, the naive
    seed bootstrap is descriptive (equals the sample range at the
    extreme percentiles); treat augment uncertainty qualitatively
    ("sign-consistent, range −0.42 to −0.81 nats") until a 10-seed
    extension lands.

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
3. *Utility-aware selection produces a genuine but small margin-level
   advantage over layer-matched, token-activation-weighted random
   zero-weight features.* On the main k=266 bundle at 10 seeds, the
   seed-mean contrast is −0.90 nats, seed-mean CI [−1.22, −0.56]
   (nested) / [−1.17, −0.61] (naive). On the k=154 augment at 3 seeds,
   every per-seed contrast is negative with CI excluding 0. Neither
   effect reaches the accuracy endpoint.
4. *Path-drift from the SAE intervention wiring alone is essentially
   zero* (`matched_zero_dead − noop` at machine precision). All
   observed margin shifts therefore come from either SAE reconstruction
   error on active features (indexed by the random-null spread) or
   genuine selection-specific causal effect (indexed by the
   `utility − matched_random` contrast).

Claim 2 does not depend on `matched_random` because both sides of the
contrast use the same α=0 intervention path — any path artefact
cancels. Claim 4 provides a direct empirical check of that cancellation
and removes "α=0 SAE path introduces artifacts" as an outstanding
concern.

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
(`scripts/select_faitheval_sae_utility_features.py`) now: (a) filters
to `token_activation_rate > 0` (pool size 112 004), (b) exact-matches
the utility-family layer histogram, (c) samples without replacement with
weights proportional to full-sequence validation-token activation rate
(Efraimidis-Spirakis keys), and (d) produces genuinely-distinct seeds
(10 for the main bundle and 3 for the augment; see §1 "Random-null
construction" and §2.3 for seed counts and fingerprints). The resulting matched_random margins
bracket `noop` from both sides (10-seed range 7.40 … 9.30 nats around
noop = 8.31), and the dedicated path-drift control (§2.4) now measures
the pure intervention-path contribution at ~10⁻⁸ nats — so the
2026-04-22 "path drift" concern is no longer a hypothesis but a measured
zero.

Historical artefacts are retained (not deleted) at:

```
data/gemma3_4b/intervention/faitheval_sae_utility_selector/heldout/<benchmark>/matched_random*/experiment_2026-04-22_prompt_end_zero_weight_control/
```

for every reran family and both benchmarks. Provenance sidecars for
those dirs keep the original selector invocation and hashes.

**L3 closure remains partial by design** (unchanged). The selector
searches all non-zero probe-support features within the existing SAE
extraction layers, not a wider SAE sweep. Cite L3 honestly in the paper.

(Cosmetic note about `readout_weight_sign` moved to §7.)

## 5. Suggested next steps (ranked)

1. **[DONE 2026-04-22]** Rerun `matched_random` with a full-sequence
   activity-weighted control. Three genuinely-distinct seeds; results
   above in §2.
2. **[DONE 2026-04-23]** Add an intervention-path baseline: α=0 on a
   manifest containing a single guaranteed-dead feature. Results in
   §2.4: paired drift +8.2 × 10⁻⁸ nats [−2.4 × 10⁻³, +2.1 × 10⁻³], 0
   compliance flips. "Path-drift artifact" is now empirically falsified
   at the mean level.
3. **Sharpen the readout-worsens-margin finding.** Move the
   `readout − noop = +0.92 [+0.61, +1.23]` nats contrast into the
   paper's main line. It is a cleaner way to make the
   "readout ≠ steering handle" point than `utility − readout` alone and
   does not need `matched_random`.
4. **[DONE 2026-04-23]** Add more random-null seeds for the main bundle.
   Extended from 3 to 10 seeds. Seed SD 0.489 nats (≈half the effect
   size), seed-mean CI [−1.22, −0.56] (nested) / [−1.17, −0.61]
   (naive) — main-bundle selection-specific margin signal is now
   bounded away from zero.
5. **If paper space permits, a minimal wider-layer probe.** Not a
   sweep — pick one new layer outside the existing SAE extraction set,
   rerun the selector pipeline, check whether the accuracy null
   survives one layer-family extension. Partial L3 closure only. Do
   not open a general SAE sweep.
6. **Limitations-table edit.** L2 moves from "central weakness" to
   "addressed via target-selection ablation: accuracy null across
   readout / utility / 10-seed layer-matched activity-weighted random /
   path-drift control; readout ablation is counter-productive at the
   margin level; utility-selected selection-specific margin signal is
   small (seed-mean −0.90 nats) but bounded away from zero at 10
   seeds". L3 remains "partially addressed — layer coverage bounded by
   existing SAE extraction".
7. **Extend the augment bundle to 10 seeds (~45 min GPU).** The k=154
   augment is currently the strongest selection-specific result but
   only has 3 seeds. A 10-seed extension would give it a proper
   seed-mean CI rather than a descriptive range. Parallel to item 4.

Items 1–4 and 6 together lift the Priority-2 result to main-text
quality (items 1, 2, 4 already done). Items 3, 5, 7 are paper hygiene
/ robustness extensions.

## 6. Uncertainty register

- **High confidence**: accuracy null across all families (12
  simultaneous comparisons); `readout − noop` margin increase;
  `utility − readout` margin direction; the augment
  `utility_positive − matched_random_positive_*` being sign-consistent
  across three seeds; **new**: the 10-seed main-bundle seed-mean
  `utility − matched_random` being negative with seed-mean CIs that
  exclude 0 by >0.5 nats; **new**: the `matched_zero_dead − noop`
  path-drift null at machine precision.
- **Medium confidence**: the *magnitude* of the augment
  selection-specific margin signal (point estimates −0.42 to −0.81 nats
  across seeds; seed mean −0.66 nats with seed sd ≈ 0.21 on n=3). A
  10-seed extension of the augment remains open.
- **Lower confidence**: any single per-seed `utility −
  matched_random_seed_k` CI read as a "population-level" effect. The
  10-seed distribution is the correct reference; per-seed reports are
  descriptive.
- **Explicitly out of scope**: "SAE steering cannot work on
  FaithEval" — would require a broader SAE sweep and a wider operator
  family.

## 7. Provenance integrity

All pipeline stages — selector (original, rerun, and 2026-04-23
extension), 36 canonical held-out family × benchmark runs, and three
reports — emitted `*.provenance.*.json` sidecars with
`status = "completed"`. Run-count breakdown (per benchmark — each
family runs on both `faitheval` and `faitheval_anti_compliance_margin`):

| Period | Families refreshed | Runs (× 2 benchmarks) |
| --- | --- | --- |
| 2026-04-22 morning | noop, readout_selected, utility_selected, utility_positive_selected | 8 |
| 2026-04-22 evening (rerun, replaces archived prompt-end control) | `matched_random_seed_{0,1,2}`, `matched_random_positive_seed_{0,1,2}` | 12 |
| 2026-04-23 extension | `matched_random_seed_{3..9}`, `matched_zero_dead` | 16 |

Verified on 2026-04-23 after the extension:

- `test_manifest.json` fingerprint `781fd7eafa5f2573` is consistent
  across all held-out outputs (ID-set hash parity on sample IDs).
- Every alpha file contains exactly 840 records; every provenance
  sidecar references the intended family feature manifest.
- Per-seed `flat_idx` fingerprints are **distinct across all 10 main
  seeds and all 3 augment seeds** (pairwise Jaccard overlap for the
  10 main seeds is 0.019–0.051, far from the deterministic collapse
  that plagued the original prompt-end control).
- `matched_zero_dead` provenance (`data/gemma3_4b/intervention/faitheval_sae_utility_selector/heldout/*/matched_zero_dead/experiment/run_intervention.provenance.*.json`)
  references `selector/matched_zero_dead_features.json` with fingerprint
  `e5132cf1b03a0386` and the k=1 layer-20 feature described in §2.4.
- Cache reuse: the 2026-04-23 selector regeneration fingerprints
  classifier + classifier-summary contents and refuses stale cache;
  `selector_scoring.cache_status = "reused"` with
  `input_hash = 10dd5a34e5ae50f779a2a67079fec9d93d9eab39f119183c96447e83d8a5ed98`
  in `selector/selector_summary.json` confirms the underlying utility
  scores, feature stats, and readout/utility manifests were carried
  forward unchanged across the 3→10 seed extension.
- Raw jsonl mean margin and compliance counts re-derived directly from
  `heldout/*/experiment/alpha_*.jsonl` match
  `report/heldout_summary.json` entries exactly (|diff| = 0.0 for
  margins; 14/14 compliance counts agree) for all 14 main-bundle
  families on `faitheval_anti_compliance_margin`.
- Archived prompt-end experiment dirs
  (`experiment_2026-04-22_prompt_end_zero_weight_control/`) retain their
  own provenance sidecars.
- No `sentinels/stop_after_selector` was active during any run; no
  partial alpha files; no missing-ID parity errors.
- `scripts/audit_ci_coverage.py` passes with zero diagnostics.

Schema version markers:
`faitheval_sae_utility_selector/v5` (selector summary),
`faitheval_sae_utility_selector_report/v5` (main held-out report),
`faitheval_sae_utility_positive_augment_report/v1` (augment, unchanged).

Cosmetic: `readout_selected_features.json` still labels all 266 features
as `weight_sign = "unknown"` despite each having a positive probe
weight; `readout_weight_sign_counts` therefore shows `{unknown: 266}`
instead of `{positive: 266}`. Does not affect numbers. Tracked for a
future minor cleanup.
