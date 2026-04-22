# FaithEval SAE Utility-Selector Ablation — L2/L3 Review — 2026-04-22

> **Verdict (data):** On n=840 held-out FaithEval test items, every SAE
> target-selection family (readout-selected k=266, utility-selected k=266,
> utility-positive k=154, layer-matched zero-weight k∈{266,154}) yields a
> null accuracy delta (all paired 95% CIs include 0; point estimates ≤0.5 pp).
> On the anti-compliance margin endpoint, utility-selected reduces the
> counterfactual-minus-preferred logprob margin by −0.76 nats vs noop
> [−1.08, −0.42] and by −0.35 nats vs layer-matched-zero-weight
> [−0.65, −0.05]; readout-selected moves the margin in the *wrong* direction
> by +0.92 nats vs noop [+0.61, +1.23].
>
> **Verdict (interpretation):** Utility-aware SAE target selection, directly
> optimized on a held-out selection criterion, does not recover FaithEval
> behavioral control in this setting. This closes L2 as "the SAE null is a
> feature-selection artifact." It does not fully close L3 (layer coverage
> restricted to existing extraction scope). The readout-vs-utility sign
> divergence at the margin level is an independently interesting finding
> that supports the paper's "good readout ≠ good steering handle"
> thesis — but see the known-issues section before citing the
> `matched_random` comparison as a clean null.

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
| Held-out report (main, k=266) | [report/heldout_summary.json](../../../data/gemma3_4b/intervention/faitheval_sae_utility_selector/report/heldout_summary.json) |
| Held-out audit note (main) | [report/audit_note.md](../../../data/gemma3_4b/intervention/faitheval_sae_utility_selector/report/audit_note.md) |
| Held-out report (augment) | [report_augment/augment_heldout_summary.json](../../../data/gemma3_4b/intervention/faitheval_sae_utility_selector/report_augment/augment_heldout_summary.json) |
| Held-out audit note (augment) | [report_augment/augment_audit_note.md](../../../data/gemma3_4b/intervention/faitheval_sae_utility_selector/report_augment/augment_audit_note.md) |
| Frozen validation manifest (n=160) | `selector/validation_manifest.json`, fingerprint `6fc512b3027fc4a0` |
| Frozen test manifest (n=840) | `selector/test_manifest.json`, fingerprint `781fd7eafa5f2573` |
| Provenance sidecars | `*.provenance.*.json` under selector/, heldout/, report/, report_augment/ |

## 1. Design recap

### Candidate pool and split

- Candidate pool: 509 SAE features with non-zero weight in the frozen FaithEval
  SAE classifier (`models/sae_detector.pkl`). Of these, 266 positive-weight
  features are used as `readout_selected`; the full 509 are used as the search
  pool for utility scoring.
- Split: stratified 160 / 840 validation/test by `(num_options,
  counterfactual_key_canonical)`, seed=42. `validation ∩ test = 0` confirmed.
- SAE scope: existing extraction layers `[0, 5, 6, 7, 13, 14, 15, 16, 17, 20]`
  at `d_sae = 16384`. No wider-layer sweep (design choice; L3 is therefore
  only partially addressed).

### Selector

- Operator: SAE `delta_only` ablation at α=0.0 (encode, multiply target
  feature by 0, decode, take delta, add to residual).
- Selector metric on validation: `baseline_margin − ablated_margin`,
  averaged across 160 items. Margin is
  `logp(counterfactual_key) − logp(preferred_key)` at the prompt-end position
  in a single-token forced-choice score.
- Top-k = 266 features picked in descending selector score (ties broken by
  `flat_idx`). Of the 509 candidates: 154 have strictly positive validation
  utility, 329 negative, 26 zero.

### Families on test (n=840)

| Family | k | α on test | Intent |
| --- | --- | --- | --- |
| `noop` | — | 1.0 (shortcut path) | Baseline |
| `readout_selected` | 266 | 0.0 | "Original paper" SAE steering target |
| `utility_selected` | 266 | 0.0 | Intervention-aware target (main L2 attack) |
| `matched_random_seed_{0,1,2}` | 266 | 0.0 | Layer / activation-freq / decoder-norm matched null — *see §4* |
| `utility_positive_selected` | 154 | 0.0 | Only strictly-positive-utility augment (prevents "size-match dilution" objection) |
| `matched_random_positive_seed_{0,1,2}` | 154 | 0.0 | Layer-matched zero-weight null for the augment — *see §4* |

### Endpoints

1. `faitheval` compliance (binary accuracy, generation-based).
2. `faitheval_anti_compliance_margin` (continuous logprob margin at prompt-end).

Per-family CIs are binomial Wilson for accuracy; bootstrap-percentile mean
CIs for margin. Paired deltas use 10 000-sample paired bootstrap by
sample_id, seed=42.

## 2. Data

### 2.1 Main bundle (k=266)

**FaithEval compliance, n=840**

| Family | Compliance | Wilson 95% CI |
| --- | --- | --- |
| `noop` | 0.6643 (558/840) | [0.632, 0.695] |
| `readout_selected` | 0.6595 (554/840) | [0.627, 0.691] |
| `utility_selected` | 0.6619 (556/840) | [0.629, 0.693] |
| `matched_random_seed_{0,1,2}` | 0.6667 (560/840) | [0.634, 0.698] |

Paired deltas vs `utility_selected` (paired bootstrap, 10 000 resamples):

| Contrast | Δ (pp) | 95% CI |
| --- | ---: | --- |
| `utility − readout` | +0.238 | [−1.31, +1.79] |
| `utility − noop` | −0.238 | [−1.67, +1.19] |
| `utility − matched_random_seed_*` | −0.476 | [−1.90, +0.95] |

All three deltas are null (CIs include zero, point estimates ≤0.5 pp).

**FaithEval anti-compliance margin (logprob nats), n=840**

| Family | Mean margin | Bootstrap 95% CI |
| --- | ---: | --- |
| `noop` | +8.309 | [+7.10, +9.50] |
| `readout_selected` | +9.227 | [+7.86, +10.57] |
| `utility_selected` | +7.546 | [+6.47, +8.60] |
| `matched_random_seed_{0,1,2}` | +7.900 | [+6.77, +9.03] |

Paired deltas vs `utility_selected`:

| Contrast | Δ (nats) | 95% CI |
| --- | ---: | --- |
| `utility − readout` | −1.681 | [−2.17, −1.19] |
| `utility − noop` | −0.763 | [−1.08, −0.42] |
| `utility − matched_random_seed_*` | −0.353 | [−0.65, −0.05] |

Supplementary contrasts (this review, same 10 000-sample paired bootstrap,
seed=42):

| Contrast | Δ (nats) | 95% CI |
| --- | ---: | --- |
| `readout − noop` | +0.918 | [+0.61, +1.23] |
| `matched_random_seed_0 − noop` | −0.409 | [−0.56, −0.25] |

### 2.2 Augment bundle (k=154)

**FaithEval compliance, n=840**

| Family | Compliance |
| --- | --- |
| `noop` (reused α=1.0) | 0.6643 (558/840) |
| `utility_positive_selected` | 0.6655 (559/840) |
| `matched_random_positive_seed_{0,1,2}` | 0.6655 (559/840) |

Paired deltas vs `utility_positive_selected`:

| Contrast | Δ (pp) | 95% CI |
| --- | ---: | --- |
| `utility_positive − noop` | +0.119 | [−1.31, +1.55] |
| `utility_positive − matched_random_positive_*` | 0.000 | [−1.43, +1.43] |

**FaithEval anti-compliance margin (nats), n=840**

| Family | Mean margin |
| --- | --- |
| `noop` | +8.309 |
| `utility_positive_selected` | +7.586 |
| `matched_random_positive_seed_{0,1,2}` | +7.993 |

Paired deltas vs `utility_positive_selected`:

| Contrast | Δ (nats) | 95% CI |
| --- | ---: | --- |
| `utility_positive − noop` | −0.723 | [−1.04, −0.38] |
| `utility_positive − matched_random_positive_*` | −0.407 | [−0.71, −0.09] |

### 2.3 Selector diagnostics

- Utility vs readout overlap: |∩| = 132 of 266, Jaccard = 0.33.
- Utility-selected with strictly positive validation score: 154 of 266.
  The remaining 112 of 266 have ≤0 validation utility (size-matching to the
  readout cardinality crossed the sign boundary; hence the k=154 augment).
- Features "outside old shortlist" (|weight| ≤ 10⁻³): 75 of 266 utility-
  selected = 28.2%.
- Utility layer histogram concentrates in layer 20 (109 of 266), 17 (40),
  16 (29), 13 (25), 14 (18), 15 (17), 0 (19), 5 (2), 6 (3), 7 (4).
  Candidate-pool layer histogram peaks at layer 20 (189 of 509). Utility
  and readout pick from roughly the same layer support.
- Utility weight-sign counts: 134 negative / 132 positive probe weight
  (near-balanced). Readout: 266/266 positive probe weight (by construction).

## 3. Interpretation

### 3.1 What the data supports

1. **Accuracy is null across every SAE target-selection rule tried.**
   Even when the selector is optimized on the same held-out metric family
   used for scoring (validation logprob margin), the behavioral endpoint
   is unmoved (all paired 95% accuracy CIs include 0; all point estimates
   are ≤0.5 pp). A reviewer cannot attribute the FaithEval SAE null to
   "you picked the wrong features" — within the candidate pool, no picking
   rule works.
2. **Utility selection transfers to test on the margin endpoint.**
   The validation-selected utility set reduces misleading-preferred margin
   by −0.76 nats vs noop on test. Confidence interval excludes 0. The
   generalization-to-test gap is not measured directly here but the signal
   is non-vanishing.
3. **Readout-selected features shift the margin in the *wrong* direction.**
   Ablating the top-266 positive-probe-weight SAE features increases the
   misleading-preferred margin by +0.92 nats vs noop
   [+0.61, +1.23]. This is a symmetric but *opposite-sign* effect to
   utility selection. Within the same candidate pool, utility and readout
   rules pick features with opposite causal effect on the margin (even
   though their overlap is 132/266).
4. **Size-match dilution is not driving the null.** The k=154
   strictly-positive augment, which the "diluted-with-harmful-features"
   objection cannot attack, replicates the accuracy null and the margin
   direction (−0.72 nats vs noop).

### 3.2 What survives scrutiny vs what does not

- **Cleanly survives**: the accuracy null across families; the
  `utility − readout` margin delta (−1.68 nats); the `readout − noop`
  margin delta (+0.92 nats); the cardinality control via the k=154 augment.
  These are the claims that support the paper's thesis.
- **Survives with caveat**: the `utility − noop` margin delta (−0.76 nats).
  About half of this is an intervention-path artifact shared with
  `matched_random` (−0.41 nats path drift, §4). The selection-specific
  increment is −0.35 nats — real but small.
- **Does not cleanly survive**: the `utility − matched_random` comparison
  as a bona-fide random-features null. The matched_random families are
  compromised by the issues in §4.

### 3.3 Framing implication for the paper

A defensible set of claims, ordered from safest to most ambitious:

1. *Even with intervention-aware SAE target selection, FaithEval
   accuracy does not move.* (Safest, headline, closes L2 as an artifact
   concern.)
2. *Within the probe-nonzero candidate pool, utility-aware and
   readout-selected rules pick features that shift the margin in opposite
   directions. The classifier's top-weighted features are not usable
   steering handles in this setting — they are, if anything,
   counter-productive.* (Secondary, but scientifically interesting; this
   is the "good readout ≠ good steering handle" claim in its cleanest
   form.)
3. *A weak but statistically detectable selection-specific margin signal
   exists over a near-noop random null.* (Third-tier; cite with the
   matched-random caveats from §4 if used at all.)

Note that claim 2 *does not depend* on `matched_random` because both sides
of the contrast (`utility` and `readout`) use the same α=0 intervention
path on probe-nonzero features — the path artifact cancels in the paired
delta.

## 4. Issues surfaced by this review

### 4.1 `matched_random` "three seeds" are byte-identical *(material)*

**Finding.** `scripts/select_faitheval_sae_utility_features.py:479`
defines `match_random_zero_weight_features(seed=...)` which is intended
to produce three independent random draws of layer-matched zero-weight
features. In practice all three seeds produce byte-identical feature
manifests:

```
seed 0 hash = 70b9a73733042251 (k=266)
seed 1 hash = 70b9a73733042251 (k=266)
seed 2 hash = 70b9a73733042251 (k=266)
```

And the same holds for `matched_random_positive_seed_{0,1,2}` in the
augment. The three held-out held-out results are therefore byte-identical
as well (e.g. FaithEval compliance = 560/840, margin = +7.8996 across all
three main seeds).

**Mechanism.** The matching function layer-matches, processes targets in
a shuffled order, and greedy-picks the nearest zero-weight candidate by
`(distance, flat_idx)` where `distance = hypot(Δ activation_frequency, Δ
decoder_norm)`. But in the FaithEval zero-weight pool:

- All ~16 000 zero-weight features per layer have `decoder_norm ≈ 1.0`
  (Gemma Scope SAE decoders are unit-norm; verified 0 of 163 331 features
  deviate by >0.01 from 1.0).
- 99.46% of zero-weight features have `activation_frequency = 0` at
  prompt-end on the validation split (882 of 163 331 activate at all).

Because both matching coordinates are effectively ~constant, every
candidate tied on `distance`. The `flat_idx` tiebreak then deterministically
picks the lowest-indexed zero-weight features per layer, regardless of
target shuffle order — so the three seeds degenerate to the same fixed
set. Confirmed by direct simulation.

**Implications.**

- Any "mean ± seed variance" interpretation of the three matched-random
  numbers is unsupported. There is effectively one matched-random draw,
  not three. The provenance / audit claim of "3 matched random seeds" is
  wrong as written.
- The CI on `utility − matched_random` reflects only sample-level
  variability under a fixed single-feature-set null, not
  selection-variability.
- The "lowest-flat_idx zero-weight features per layer" is not a
  meaningful layer-matched random control: it is a deterministic pick
  that happens to be almost entirely dead features.

### 4.2 The `matched_random` null is a near-noop intervention *(material)*

**Finding.** The `matched_random` families at α=0 produce a margin shift
of −0.41 nats vs noop [−0.56, −0.25] — not ≈0. This is an
intervention-path artifact: the α=1 shortcut in `SAEFeatureScaler` returns
the residual unchanged, while α=0 runs encode → multiply-by-0 → decode →
add-delta, which on dead features is mathematically zero but numerically
introduces small drift (float precision, bf16/fp32 conversion in the
encode/decode path, minor reconstruction noise on non-target features).

**Implication.** Any comparison that uses noop as its baseline conflates
"selection-specific effect" with "intervention-path drift". In the main
bundle, roughly 54% of the `utility − noop` margin effect is attributable
to path drift (−0.41 nats) and 46% to selection specificity (−0.35 nats).
The paper must either (a) use `utility − matched_random` as the primary
contrast (clean on path drift but compromised by §4.1), (b) use a
properly-constructed α=0 ablation-at-no-target baseline (e.g. ablate a
single dummy feature that is definitely dead everywhere), or (c) restrict
claims to `utility − readout`, where both sides share the α=0 path.

### 4.3 `matched_random` is matched on prompt-end stats only *(moderate)*

The stats used for layer-matching (`activation_frequency`, `decoder_norm`)
are computed at the prompt-end position only
(`PromptEndFeatureCollector`). Many features that appear dead at
prompt-end are active at intermediate positions and thus contribute
meaningfully through subsequent attention. The intended "match by
activation frequency" is therefore not matching the quantity that
actually drives the intervention. Pool-level richness is not preserved
in the match.

### 4.4 Stats field not propagated into matched manifests *(cosmetic)*

`readout_selected_features.json` labels all 266 features as
`weight_sign = "unknown"` despite each having a positive probe weight (the
`weight` field is present; `weight_sign` is not set by
`get_positive_sae_features_from_classifier`). `readout_weight_sign_counts`
in the selector summary therefore says `{unknown: 266}` instead of
`{positive: 266}`. Purely cosmetic; does not affect results.

### 4.5 L3 closure is partial by design *(known and acknowledged)*

The design docstring and `selector_summary.json:selector_design.layer_coverage_note`
both state this explicitly: *"Partial L3 closure only: selection searches
all non-zero probe-support features within the existing SAE extraction
layers, not a wider SAE sweep."* Good hygiene, but cite L3 honestly in the
paper. A wider SAE sweep would be a separate, larger experiment.

## 5. Suggested next steps (ranked)

1. **Rerun `matched_random` with a proper control (≤½ day, high
   information).** Either (a) sample uniformly at random from the
   probe-nonzero candidate pool minus `utility_selected`, layer-matched
   to the utility histogram (same number of active features per layer);
   or (b) sample from the zero-weight pool but weight by full-sequence
   activation probability (not just prompt-end). Three *actually
   independent* seeds. This closes §4.1, §4.2, §4.3 in one artifact and
   yields a clean `utility − random` null that the paper can cite without
   caveats.
2. **Add an intervention-path baseline (~1 hr).** Run α=0 on a manifest
   containing a single *guaranteed-dead* feature (flat_idx picked to have
   activation_frequency = 0 and zero classifier weight). This isolates
   the pure path-drift number and lets every `X − noop` delta be
   decomposed into "path drift + selection specificity".
3. **Sharpen the readout-worsens-margin finding.** Move the
   `readout − noop = +0.92 [+0.61, +1.23]` nats contrast into the paper's
   main line. It is a cleaner way to make the "readout ≠ steering handle"
   point than "utility − readout", and does not need `matched_random`.
4. **If paper space permits, a minimal wider-layer probe.** Not a sweep —
   pick one new layer outside the existing SAE extraction set, re-run the
   selector pipeline, and check whether the accuracy null survives one
   layer-family extension. Partial L3 closure only. Do not open a general
   SAE sweep.
5. **Limitations-table edit.** L2 moves from "central weakness" to
   "addressed via target-selection ablation; readout features are
   counter-productive at the margin level; utility-selected features show
   detectable but tiny margin signal insufficient to shift accuracy". L3
   remains "partially addressed — layer coverage is bounded by existing
   SAE extraction".

Items 1–3 together would tighten the Priority-2 result to genuine main-
text quality. Items 4–5 are paper-hygiene.

## 6. Uncertainty register

- **High confidence**: accuracy null across all families; readout-noop
  margin increase; utility-noop margin decrease at population level.
- **Medium confidence**: the *magnitude* of selection-specific margin
  signal (current estimate −0.35 nats, but the random null it is
  compared to is biased in the §4 ways).
- **Low confidence**: any seed-variance interpretation of the reported
  matched-random bootstraps (§4.1 collapses this to a single-draw null).
- **Explicitly out of scope**: "SAE steering cannot work on FaithEval"
  (would require a broader SAE sweep and wider operator family).

## 7. Provenance integrity

All seven pipeline stages (selector, six held-out families per benchmark,
two reports) emitted `*.provenance.*.json` sidecars with
`status = "completed"`. The test manifest fingerprint `781fd7eafa5f2573`
is consistent across all held-out outputs (verified by direct ID set
equality against `selector/test_manifest.json`). No
`sentinels/stop_after_selector` was active during the runs. No partial
alpha files; no missing ID parity errors.

The `schema_version` markers
(`faitheval_sae_utility_selector_report/v4` and
`faitheval_sae_utility_positive_augment_report/v1`) distinguish the main
and augment summaries cleanly.
