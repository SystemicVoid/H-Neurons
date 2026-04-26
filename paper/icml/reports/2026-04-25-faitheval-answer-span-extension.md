# FaithEval SAE Selector — Answer-Span Pool Extension (Held-Out Review)

> **Status:** canonical for the 2026-04-25 answer-span extension of the
> FaithEval SAE utility-selector ablation. The
> [2026-04-22 review](./2026-04-22-faitheval-sae-utility-selector-review.md)
> remains canonical for the original `utility_selected` (k=266) and
> `utility_positive` augment (k=154) families and the 10-seed
> matched-random + path-drift baseline. This report adds the
> `answer_span_selected` family (k=266), its layer-matched random null
> (`matched_random_answer_span_seed_{0..9}`), the new
> `faitheval_answer_span_margin` endpoint (first-3 assistant-token
> answer-text logprob margin), and the cross-metric contrasts. Schema
> version bumped to v6 (selector summary and held-out report).
>
> **Verdict (data):** On the locked n=840 FaithEval test split, three
> SAE target-selection families are now compared (`readout_selected`,
> `utility_selected`, `answer_span_selected`, all k=266) against a
> 10-seed layer-matched activity-weighted random null per-pool, a
> single-feature path-drift control, and `noop`. Compliance moves are
> null across the full bundle (every paired 95 % CI includes 0; point
> estimates ≤ 1.43 pp). On the **answer-span margin** endpoint (the
> selector's own metric), `answer_span_selected − noop` =
> **−0.33 nats [−0.61, −0.06]**, `answer_span − readout` =
> **−0.98 [−1.38, −0.57]**, and the across-seed `answer_span −
> matched_random_answer_span` seed-mean is **−0.53 nats** with seed sd
> 0.15 nats; primary nested CI [−0.81, −0.26], naive seed CI
> [−0.63, −0.45]; both exclude 0; range [−0.80, −0.37]. On the
> **anti-compliance margin** endpoint (the original prompt-end choice
> margin, the metric `utility_selected` was tuned for), the new pool
> moves the *opposite* way relative to `utility_selected`:
> `answer_span − utility_selected` = **+0.46 nats [+0.16, +0.75]** —
> CI excludes 0 — i.e. the answer-span pool is **worse** than the
> utility pool on the prompt-end metric. Across-seed `answer_span −
> matched_random_answer_span` on the same anti-compliance metric is
> still negative (seed-mean −0.45 nats; primary nested CI [−0.86,
> −0.03]; naive seed CI [−0.72, −0.13]) but ≈half the size of the
> original utility-bundle effect (−0.90 nats). On **compliance**, the
> seed-mean `answer_span − matched_random_answer_span` is +0.82 pp;
> the **primary nested CI [−0.80, +2.45] includes 0** while the naive
> seed bootstrap [+0.38, +1.19] excludes 0 — read the nested CI as the
> headline (see §3.5). The path-drift control
> (`matched_zero_dead − noop`) is unchanged at +8.22 × 10⁻⁸ nats
> [−2.4 × 10⁻³, +2.1 × 10⁻³] and remains four orders of magnitude
> below all selection-specific signals.
>
> **Verdict (interpretation):** Three things to take away. (1) The
> answer-span selector behaves as designed *on its own metric*: it
> reduces the misleading-vs-preferred answer-text margin in the
> first-3 generation tokens, with both seed-mean CIs excluding 0 and
> a tight cross-seed dispersion (0.15 nats). This is generalization
> from a 160-item validation split to an 840-item disjoint test
> split, but on the **same scoring code path** as the selector — so
> it is a within-metric generalization claim, not a transfer claim
> (§3.4). (2) The cross-metric contrast is the new clean *negative*
> finding. The same selection rule that wins on its own metric loses
> by +0.46 nats vs `utility_selected` on the original prompt-end
> margin (CI excludes 0). The two pools share 167/266 features
> (Jaccard 0.46) but their disjoint 99 features each pull the
> prompt-end margin in opposite directions. Selecting features for
> "shifts the answer-text continuation" is therefore not the same
> objective as "shifts the prompt-end forced-choice margin": even
> within the probe-nonzero candidate pool, optimising one metric
> trades off against the other. (3) Compliance remains uniformly
> null across every selection rule and every random null tried —
> readout, utility, utility-positive (augment), answer-span, two
> 10-seed matched-random pools, and the path-drift control. No
> selection inside the existing 509-feature candidate pool yet moves
> behavioural compliance. The L2 closure is unchanged in direction,
> sharpened by a third selection rule, and gains an unexpected
> result on cross-metric inconsistency that the paper can use as a
> standalone observation about the SAE basis on FaithEval.

## Source Hierarchy

- Run directory: [data/gemma3_4b/intervention/faitheval_sae_utility_selector/](../../../data/gemma3_4b/intervention/faitheval_sae_utility_selector/)
- Sibling canonical reports:
  - [2026-04-22-faitheval-sae-utility-selector-review.md](./2026-04-22-faitheval-sae-utility-selector-review.md) (utility / readout / matched-random / path-drift; this report extends it with the answer-span pool)
  - [2026-04-21-bridge-margin-report.md](./2026-04-21-bridge-margin-report.md), [2026-04-21-bridge-irr-review.md](./2026-04-21-bridge-irr-review.md)
- Limitations table: [paper/icml/reviews/TODO_Limitations_Fixes.md §Priority 2](../reviews/TODO_Limitations_Fixes.md)
- Pipeline scripts:
  - Selector: [scripts/select_faitheval_sae_utility_features.py](../../../scripts/select_faitheval_sae_utility_features.py)
  - Wrapper: [scripts/infra/faitheval_sae_utility_selector.sh](../../../scripts/infra/faitheval_sae_utility_selector.sh)
  - Held-out scoring (incl. answer-span): [scripts/run_intervention.py](../../../scripts/run_intervention.py) (`run_faitheval_answer_span_margin`, `score_faitheval_answer_text_targets_from_prompt_ids`, constant `FAITHEVAL_ANSWER_SPAN_PRIMARY_WINDOW_TOKENS = 3`)
  - Reporting: [scripts/report_faitheval_sae_utility_selector.py](../../../scripts/report_faitheval_sae_utility_selector.py)
- Framing governance: [notes/2026-04-21-claim-framing-governance.md](../../../notes/2026-04-21-claim-framing-governance.md)
- Measurement contract: [notes/measurement-blueprint.md](../../../notes/measurement-blueprint.md)
- Quantitative reporting standards: [docs/quantitative-reporting-standards.md](../../../docs/quantitative-reporting-standards.md)

## Data Files

| Artifact | Path |
| --- | --- |
| Selector summary (v6) | [selector/selector_summary.json](../../../data/gemma3_4b/intervention/faitheval_sae_utility_selector/selector/selector_summary.json) |
| Selector scoring state (cache hash linkage) | [selector/selector_scoring_state.json](../../../data/gemma3_4b/intervention/faitheval_sae_utility_selector/selector/selector_scoring_state.json) |
| Answer-span selector scores (per-feature, 509 rows) | [selector/answer_span_scores.jsonl](../../../data/gemma3_4b/intervention/faitheval_sae_utility_selector/selector/answer_span_scores.jsonl) |
| Answer-span selected manifest (k=266) | [selector/answer_span_selected_features.json](../../../data/gemma3_4b/intervention/faitheval_sae_utility_selector/selector/answer_span_selected_features.json) |
| Per-seed answer-span random-null manifests (10 seeds) | `selector/matched_random_answer_span_seed_{0..9}_features.json` |
| Held-out report (v6, three metrics) | [report/heldout_summary.json](../../../data/gemma3_4b/intervention/faitheval_sae_utility_selector/report/heldout_summary.json) |
| Held-out audit note | [report/audit_note.md](../../../data/gemma3_4b/intervention/faitheval_sae_utility_selector/report/audit_note.md) |
| Augment (k=154, separate v2 schema, 3 seeds, unchanged) | [report_augment/augment_heldout_summary.json](../../../data/gemma3_4b/intervention/faitheval_sae_utility_selector/report_augment/augment_heldout_summary.json) |
| Pre-extension report snapshot | `report_2026-04-23_pre_answer_span_heldout/` |
| Frozen test manifest (n=840) | `selector/test_manifest.json`, fingerprint `781fd7eafa5f2573` |
| Frozen validation manifest (n=160) | `selector/validation_manifest.json`, fingerprint `6fc512b3027fc4a0` |

## 1. What changed since the 2026-04-22 review

### 1.1 New family: `answer_span_selected` (k=266)

Same candidate pool (509 SAE features with non-zero classifier weight,
layer set `{0, 5, 6, 7, 13, 14, 15, 16, 17, 20}`, fingerprint
`27a0b55d10f64700`) and same 160 / 840 stratified split as the 2026-04-22
review. Selector metric is the **mean validation reduction in the first-3
assistant-content-token answer-text logprob margin**:

```
selector_score(f) = baseline_first3_margin − ablated_first3_margin
                   averaged across the 160 validation samples
margin_first3     = sum_{i=0..2} logp(counterfactual_text_token_i)
                   − sum_{i=0..2} logp(preferred_text_token_i)
```

Higher score = ablating that feature reduces the misleading-vs-preferred
answer-text margin more on validation. Top-k=266 selected by descending
score, `flat_idx` tiebreak. Schema also stores
`selector_score_full_span` (full answer text) for diagnostic use; ranking
is by `first3` only. Constant
`FAITHEVAL_ANSWER_SPAN_PRIMARY_WINDOW_TOKENS = 3` is shared between the
selector and the held-out benchmark scorer (single source of truth in
`run_intervention.py`).

### 1.2 New random null: `matched_random_answer_span_seed_{0..9}`

Same construction as the original `matched_random` (Efraimidis–Spirakis
weighted-without-replacement on the
`token_activation_rate > 0` zero-weight pool, eligible-pool n = 112 004,
weights proportional to full-sequence token activation rate on the frozen
validation split). The only difference: the layer histogram is
exact-matched to `answer_span_selected` instead of `utility_selected`.
All 10 seed manifests have distinct `flat_idx` fingerprints
(`6facf30b3819f97f, 75e54426fffb7fe3, 4820fa9ea5f84acb,
88d3c3b17644f609, 04b710c6d108ba78, b76827f42847a8ab,
7fdafe1bb92af84c, 959e03159b966854, 223adb930476ba00, 674bcfc36c726b39`).

### 1.3 New endpoint: `faitheval_answer_span_margin`

Teacher-forced scoring of the counterfactual / preferred answer-text
continuations, summing per-token logprobs over the first 3 assistant
tokens, returning `margin_first3 = sum(counterfactual) − sum(preferred)`
per sample. Same prompt template and `prompt_style="anti_compliance"` as
the original `faitheval_anti_compliance_margin` benchmark. Per-record
`metric_value` is the first-3 margin in nats; the full-span margin is
also stored under `answer_text_diagnostics` for diagnostics.

### 1.4 Pipeline guard tightening (commit d335492)

`selector_stage_complete()` and `heldout_stage_complete()` now validate
**content invariants** (manifest counts vs `selector_summary.families[*].k`,
hash match between `selector_summary.selector_scoring.input_hash` and
`selector_scoring_state.input_hash`, score-file required-field coverage,
non-empty validation/test manifests, family-aware
alpha/manifest pairing) instead of relying on file-mtime freshness, which
was flaky after artifact rewrites. Re-runs now refuse to skip when any
expected manifest, score column, or candidate-pool ID-set is missing or
inconsistent. This addresses a real correctness gap that the older
`artifact_is_fresh` check masked.

### 1.5 What did **not** change

- Candidate pool, validation/test split, classifier (`models/sae_detector.pkl`),
  SAE extraction layers, intervention operator (`delta_only`, α=0).
- Original `utility_selected` and `readout_selected` k=266 manifests,
  including their selector scores and per-row evaluation outputs.
- 10-seed `matched_random_seed_{0..9}` (utility-pool layer-matched), the
  `matched_zero_dead` path-drift control, and all derived contrasts in
  the 2026-04-22 report.
- Augment k=154 (`utility_positive_selected` and 3-seed
  `matched_random_positive_seed_{0..2}`); the augment lives in a separate
  schema (`faitheval_sae_utility_positive_augment/v2`) under
  `report_augment/` and is unchanged.

## 2. Data (canonical numbers)

All numbers below were re-derived from raw `alpha_*.jsonl` files
independently of the report code; family means, paired delta point
estimates, and per-seed contrasts agree with `report/heldout_summary.json`
to floating-point precision. CI methods: Wilson (binomial accuracy) /
percentile bootstrap mean, n=10 000 (margins); paired deltas via
10 000-sample paired bootstrap keyed on `sample_id`, seed=42; seed-mean
CIs via nested paired bootstrap (primary) and naive iid seed bootstrap
(supporting), each with n=10 000 resamples and seed=42.

### 2.1 Family means on n=840 test

| Family | k | α | Compliance (Wilson 95 % CI) | Anti-compliance margin (boot 95 % CI) | Answer-span margin (boot 95 % CI) |
| --- | ---: | ---: | --- | --- | --- |
| `noop` | — | 1.0 | 0.6643 (558/840) [0.632, 0.695] | +8.309 [+7.101, +9.499] | +2.126 [+1.383, +2.884] |
| `readout_selected` | 266 | 0.0 | 0.6595 (554/840) [0.627, 0.691] | +9.227 [+7.862, +10.573] | +2.775 [+1.934, +3.624] |
| `utility_selected` | 266 | 0.0 | 0.6619 (556/840) [0.629, 0.693] | +7.546 [+6.475, +8.598] | +2.024 [+1.306, +2.759] |
| `answer_span_selected` | 266 | 0.0 | 0.6738 (566/840) [0.641, 0.705] | +8.004 [+6.897, +9.089] | +1.800 [+1.069, +2.536] |
| `matched_zero_dead` | 1 | 0.0 | 0.6643 (558/840) [0.632, 0.695] | +8.309 [+7.102, +9.499] | n/a |

Two consistency checks:
- `noop` and `matched_zero_dead` are identical to bf16 rounding on every
  metric; their per-sample anti-compliance-margin difference is non-zero
  in only 8/840 cases, all at ±0.25 / ±0.50 nats and balancing to a
  signed sum ≈ +6.9 × 10⁻⁵ nats (paired mean +8.22 × 10⁻⁸ nats). The
  path-drift hypothesis remains empirically falsified at machine
  precision.
- `readout_selected` worsens both margin metrics relative to `noop`
  (anti-compliance margin: +0.92 nats; answer-span margin: +0.65 nats),
  consistent with the 2026-04-22 finding that *readout-weight-top
  features are not steering handles*.

### 2.2 Paired deltas vs `answer_span_selected` (anchor)

Bold rows have paired-bootstrap CIs that exclude 0.

**Compliance, paired delta in pp (n=840)**

| Contrast | Δ (pp) | 95 % CI |
| --- | ---: | --- |
| `answer_span − noop` | +0.952 | [−0.833, +2.738] |
| `answer_span − readout` | +1.429 | [−0.357, +3.214] |
| `answer_span − utility_selected` | +1.190 | [−0.357, +2.738] |

All three null. The point estimates are positive; the upper CI for the
`answer_span − readout` contrast is the largest at +3.21 pp.

**Anti-compliance margin, paired delta in nats (n=840)**

| Contrast | Δ (nats) | 95 % CI |
| --- | ---: | --- |
| `answer_span − noop` | −0.305 | [−0.728, +0.120] |
| **`answer_span − readout`** | **−1.223** | **[−1.773, −0.665]** |
| **`answer_span − utility_selected`** | **+0.458** | **[+0.161, +0.750]** |

The cross-metric anomaly: `answer_span_selected` is *worse* than
`utility_selected` by ≈0.46 nats on the prompt-end choice margin, with
CI excluding 0. It still beats `readout_selected` by 1.22 nats (CI
excludes 0).

**Answer-span margin (selector-native metric), paired delta in nats (n=840)**

| Contrast | Δ (nats) | 95 % CI |
| --- | ---: | --- |
| **`answer_span − noop`** | **−0.326** | **[−0.608, −0.055]** |
| **`answer_span − readout`** | **−0.976** | **[−1.380, −0.566]** |
| `answer_span − utility_selected` | −0.224 | [−0.468, +0.008] |

`answer_span_selected` reduces its native metric vs `noop` and
`readout_selected`, with CIs excluding 0. The contrast vs
`utility_selected` is borderline negative (point −0.224 nats, upper CI
+0.008) — `utility_selected` and `answer_span_selected` move the
answer-span margin in the same direction, with `answer_span_selected`
nudging slightly further.

### 2.3 Across-seed summaries (10 seeds, `matched_random_answer_span_seed_{0..9}`)

Per-seed contrasts on each metric, plus seed-mean summary statistics:

**Compliance, paired delta in pp**

| Quantity | Value |
| --- | --- |
| Per-seed range | [−0.476, +1.429] pp |
| Seed mean | +0.821 pp |
| Seed sd | 0.684 pp |
| Primary (nested bootstrap) CI | [−0.798, +2.452] |
| Supporting (naive seed bootstrap) CI | [+0.381, +1.190] |

8 of 10 per-seed contrasts are positive (range +0.83 … +1.43 pp); 2 are
negative (seed_8 = −0.48 pp, seed_9 = −0.36 pp). The primary nested CI
**includes** 0; the supporting naive seed CI **excludes** 0. See §3.5
for why the headline reading must use the nested CI.

**Anti-compliance margin, paired delta in nats**

| Quantity | Value |
| --- | --- |
| Per-seed range | [−1.220, +0.724] nats |
| Seed mean | −0.446 nats |
| Seed sd | 0.509 nats |
| Primary (nested bootstrap) CI | [−0.857, −0.032] |
| Supporting (naive seed bootstrap) CI | [−0.718, −0.134] |

7 of 10 per-seed CIs exclude 0 on the negative side; seed_2 has a
positive sign with CI excluding 0 on the positive side (single outlier).
Both seed-mean CIs exclude 0; the answer-span pool is a real but small
selection-specific signal on the prompt-end margin — about half the size
of the original utility-pool effect (−0.90 nats; see 2026-04-22 §2.1).

**Answer-span margin, paired delta in nats**

| Quantity | Value |
| --- | --- |
| Per-seed range | [−0.801, −0.369] nats |
| Seed mean | −0.530 nats |
| Seed sd | 0.154 nats |
| Primary (nested bootstrap) CI | [−0.811, −0.262] |
| Supporting (naive seed bootstrap) CI | [−0.625, −0.447] |

Every per-seed paired CI excludes 0 in the negative direction. Seed sd
0.15 nats is well below the effect size (0.53 nats), the tightest
across-seed spread of any contrast in this bundle. Both seed-mean CIs
exclude 0 by ≥0.26 nats. This is the cleanest selection-specific
signal in the new bundle, **but it is on the same metric the selector
optimised** (see §3.4 for the circularity caveat).

### 2.4 Selector pool diagnostics

Layer histogram (count of selected SAE features per layer, with
candidate-pool baseline):

| Layer | Pool | `utility_selected` | `answer_span_selected` | `readout_selected` |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 39 | 19 | 17 | 26 |
| 5 | 6 | 2 | 1 | 3 |
| 6 | 9 | 3 | 3 | 4 |
| 7 | 13 | 4 | 3 | 5 |
| 13 | 49 | 25 | 18 | 26 |
| 14 | 42 | 18 | 13 | 23 |
| 15 | 38 | 17 | 17 | 19 |
| 16 | 49 | 29 | 30 | 28 |
| 17 | 75 | 40 | 47 | 39 |
| 20 | 189 | 109 | 117 | 93 |
| **Total** | **509** | **266** | **266** | **266** |

Layer-20 fraction: pool 37.1 %, utility 41.0 %, answer-span **44.0 %**,
readout 35.0 %. Answer-span concentrates more strongly in the last
extraction layer — consistent with "features that move the
generated-token continuation cluster near the readout" — but the shift
is modest.

Pairwise overlap (Jaccard / intersection / union):

| Pair | Jaccard | ∩ | ∪ |
| --- | ---: | ---: | ---: |
| `utility_selected` ∩ `readout_selected` | 0.330 | 132 | 400 |
| `answer_span_selected` ∩ `readout_selected` | 0.357 | 140 | 392 |
| `answer_span_selected` ∩ `utility_selected` | **0.458** | **167** | 365 |
| `utility ∩ answer_span ∩ readout` (3-way) | — | 80 | — |

`utility_selected` and `answer_span_selected` share 167/266 features
(63 % of either set). The remaining 99 features per pool are
selector-disjoint — and those are where the cross-metric divergence in
§2.2 lives.

Weight-sign distribution within each selected family:

- `utility_selected`: 134 negative / 132 positive probe weight.
- `answer_span_selected`: 126 negative / **140 positive**.
- `readout_selected`: all 266 labelled `unknown` (cosmetic bug,
  no impact on numbers; tracked since 2026-04-22 §7).

`answer_span_selected` skews slightly toward features with positive
probe weight (52.6 % vs `utility_selected`'s 49.6 %). Both are
near-balanced; neither is a "positive readout" subset.

Outside the legacy |w|>10⁻³ shortlist:

- `utility_selected`: 75 / 266 (28.2 %).
- `answer_span_selected`: 87 / 266 (32.7 %).
- `readout_selected`: 0 / 266 (by construction).

Both intervention-aware selectors pick a third of their features from
*outside* the historical readout-weight shortlist, indicating the
candidate-pool gap is real for both selection rules.

## 3. Interpretation

### 3.1 What the data supports cleanly

1. **Compliance is null across every selection rule attempted.**
   Adding a third intervention-aware selector (`answer_span_selected`)
   to the readout / utility / utility-positive / 10-seed matched-random
   / path-drift bundle did not move the behavioural endpoint. Every
   paired 95 % CI on the compliance benchmark includes 0; per-family
   point estimates cover 0.655–0.679 = ±1 % around `noop` = 0.664. The
   strongest L2 closure claim — *no SAE target-selection rule inside
   the probe-nonzero candidate pool can recover FaithEval accuracy* —
   is sharper, not weaker, after this extension.
2. **`answer_span_selected` reduces its own selector metric on
   held-out test, beyond what random matching achieves.** Seed-mean
   contrast against `matched_random_answer_span` is −0.53 nats on the
   first-3-token answer-text margin, with both nested
   (CI [−0.81, −0.26]) and naive seed (CI [−0.63, −0.45]) bootstrap
   intervals excluding 0, and a cross-seed sd of 0.15 nats — the
   tightest seed dispersion in this bundle. Generalisation from
   validation (n=160) to test (n=840) is real for the metric the
   selector was tuned on.
3. **Cross-metric inconsistency is real and bounded.** The same
   `answer_span_selected` rule that wins on the answer-span margin
   *loses* by +0.46 nats vs `utility_selected` on the original
   prompt-end anti-compliance margin (CI excludes 0), and is
   indistinguishable from `noop` on compliance. Two SAE selection
   rules within the same probe-nonzero pool, sharing 167/266 features,
   pull the prompt-end metric in opposite directions when they
   disagree.
4. **The path-drift control still rules out
   "α=0 SAE-path artifact"** as an alternative explanation for any of
   the new effects. `matched_zero_dead − noop = +8.22 × 10⁻⁸` nats with
   8/840 ±bf16-quantised per-row diffs — unchanged from
   2026-04-23, and four orders of magnitude below the smallest
   selection-specific effect in the table.

### 3.2 What survives with caveat

- **`answer_span_selected − utility_selected` on the answer-span
  margin** (point −0.224, CI [−0.468, +0.008]). The CI just barely
  spans 0 and the contrast is not paired against the same null, so the
  natural reading is: "answer-span and utility selectors point in the
  same direction on the answer-span margin; the answer-span selector
  goes slightly further." Don't promote it past that.
- **`answer_span_selected − matched_random_answer_span` on
  anti-compliance margin** (seed-mean −0.45, primary nested CI
  [−0.86, −0.03]). Real but small and noisy; nested CI's upper bound
  hugs 0. Do not lead with this number; lead with the
  `answer_span − utility_selected = +0.46` cross-metric finding, which
  is the more striking and methodologically tighter contrast.
- **`answer_span_selected` compliance lift** (seed-mean +0.82 pp,
  point +0.95 pp vs `noop`). Naive seed bootstrap excludes 0 [+0.38,
  +1.19]; nested bootstrap [−0.80, +2.45] does not. The naive-vs-nested
  divergence is methodologically informative (§3.5). Do not cite this
  as evidence of a compliance effect; cite as a non-monotonicity in the
  null distribution that is worth tracking.

### 3.3 What does not survive

- **Any per-seed contrast cited in isolation** as a "significant
  result". With three metrics × {three pairwise headline contrasts +
  ten per-seed contrasts} the bundle has dozens of simultaneous tests;
  a Bonferroni-style adjustment would loosen per-seed CIs measurably.
  Always cite the seed-mean (with primary nested CI) for population
  claims; per-seed numbers are descriptive.
- **The naive seed bootstrap CI for compliance answer_span_selected**
  ([+0.38, +1.19] pp) read as a population-level effect. The naive
  seed bootstrap collapses when the 10 per-seed point estimates happen
  to cluster (here within ±0.95 pp of the seed mean) and stops
  reflecting per-sample variability. The nested bootstrap is the
  documented `primary_ci_method` and includes 0 here.

### 3.4 Cross-metric tradeoff (cleanest novel finding)

Stacking the 2026-04-22 result and the new pool, the picture across
SAE selection rules is:

| Selector | Anti-compliance margin vs `noop` | Answer-span margin vs `noop` | Compliance vs `noop` |
| --- | ---: | ---: | ---: |
| `readout_selected` | **+0.918 [+0.609, +1.229]** (worse) | **+0.649 [+0.303, +0.988]** (worse) | −0.476 pp [−1.548, +0.595] |
| `utility_selected` | **−0.763 [−1.084, −0.422]** (better) | −0.102 [−0.302, +0.095] (null) | −0.238 pp [−1.667, +1.190] |
| `answer_span_selected` | −0.305 [−0.728, +0.120] (null) | **−0.326 [−0.608, −0.055]** (better) | +0.952 pp [−0.833, +2.738] |

Three different selection rules within the *same* candidate pool
produce three different behaviours on the two margin metrics:
`readout_selected` worsens both, `utility_selected` improves the
prompt-end margin and is null on the answer-span margin,
`answer_span_selected` improves the answer-span margin and is null on
the prompt-end margin. The paired `answer_span − utility = +0.46
[+0.16, +0.75]` nats on anti-compliance margin (§2.2) makes the
tradeoff direct: optimising for one metric trades against the other.
*This is not a path-drift artifact* (the zero-dead control is +8 × 10⁻⁸
nats), and it is *not a layer-coverage artifact* (both selectors operate
on the same 509-feature pool spanning the same 10 layers). It reflects
a real difference between (i) features that perturb the
prompt-end forced-choice softmax and (ii) features that perturb the
generated answer-text continuation, even though the two perturbations
are evaluated on the same underlying single-token-vs-text contrast
(misleading vs preferred).

**Caveat: circularity on the answer-span endpoint.** The
`faitheval_answer_span_margin` benchmark uses the same scoring code
path (`score_faitheval_answer_text_targets_from_prompt_ids`,
`primary_window_tokens=3`) as the `answer_span_selected` selector
metric. The held-out result on this metric is a generalization-from-
160-validation-to-840-test claim under the same metric definition, not
a transfer-to-different-metric claim. The cross-metric contrasts
(`answer_span − utility_selected` on anti-compliance margin and
compliance) are the methodologically conservative tests, and on those
metrics the answer-span pool is at best mixed (anti-compliance margin)
or null (compliance).

### 3.5 Reading the compliance signal honestly

For `answer_span_selected − matched_random_answer_span` on compliance,
the seed-mean is +0.82 pp with naive seed bootstrap CI [+0.38, +1.19]
(excludes 0) but nested paired bootstrap CI [−0.80, +2.45] (includes
0). The two methods answer different questions:

- **Naive seed bootstrap** resamples the 10 per-seed point estimates
  iid with replacement. It asks: "if we drew different sets of 10
  random-null seeds, how would the seed-mean estimate vary?" When
  per-seed point estimates happen to cluster (here, 8/10 between +0.83
  and +1.43 pp; 2/10 around −0.4 pp), this CI is narrow.
- **Nested paired bootstrap** resamples test samples (sample IDs)
  jointly across `answer_span_selected` and all 10 seeds, computing
  the per-sample seed-mean contrast and bootstrapping over rows. It
  asks: "conditional on these 10 seeds, how variable is the per-sample
  (answer_span − mean-of-seeds-baseline) across the 840-item test
  set?" When per-sample variability dominates, this CI is wide.

The report code documents `primary_ci_method = "nested_bootstrap"` for
exactly this reason: a tightly clustered set of seed point estimates
should not buy a tight headline CI when the underlying per-sample
variability swamps the mean shift. **The headline reading is therefore
"compliance is null even on the seed-mean test"**, with a footnote that
the naive seed bootstrap excludes 0 — interesting but not the
population-level claim.

(For the anti-compliance margin and answer-span margin contrasts,
both naive and nested CIs exclude 0, so the issue does not arise; the
signal is real on those metrics.)

### 3.6 Framing implication for the paper

The 2026-04-22 review's claim ladder (most defensible to most
ambitious) extends naturally:

1. *Compliance does not move under any SAE target-selection rule
   inside the probe-nonzero candidate pool — readout, utility,
   utility-positive (k=154 augment), answer-span, two 10-seed
   matched-random pools, or path-drift control.* (Strongest L2 closure;
   headline.)
2. *Within the same candidate pool, three selection rules pick
   feature sets with three different signed effects on the two margin
   metrics. Readout-weighted selection moves both margins the wrong
   way. Utility selection improves the prompt-end forced-choice
   margin and is null on the answer-span margin. Answer-span selection
   improves the answer-span margin and is null on the prompt-end
   margin. The cross-metric tradeoff is bounded: `answer_span −
   utility_selected = +0.46 nats [+0.16, +0.75]` on anti-compliance
   margin.* (New, secondary; does not depend on `matched_random`.)
3. *Both intervention-aware rules produce real but small selection-
   specific margin signals against layer- and activity-matched zero-
   weight nulls, with effect sizes ordered: utility (≈0.90 nats vs
   matched random) > answer-span (≈0.45 nats vs matched random). The
   path-drift control rules out "α=0 SAE wiring is the explanation"
   at four orders of magnitude.* (Tertiary; needs the random-null
   construction.)

Claim 2 is the new clean observation: it stands without any
random-null comparison and without any cross-bundle pooling.

## 4. Limitations and uncertainties

- **High confidence:**
  - Compliance null across the full 14-family bundle (existing
    utility-pool 12 families + answer-span pool 12 families on
    compliance, including 10 + 10 random-null seeds).
  - `answer_span_selected − utility_selected = +0.46 nats [+0.16,
    +0.75]` on anti-compliance margin (the cross-metric tradeoff).
  - `answer_span_selected − matched_random_answer_span` seed-mean on
    answer-span margin, both CIs exclude 0 by ≥0.26 nats.
  - All numbers in §2.1–§2.3 reproduce from raw `alpha_*.jsonl` files
    independently of the report code (verified: 14 family-mean margins
    exact to floating-point; 14 compliance counts exact; per-seed
    paired deltas exact).
  - Path-drift null at 8 × 10⁻⁸ nats (carried over from 2026-04-23).
  - Schema and hash linkage: `selector_summary.selector_scoring.input_hash`
    matches `selector_scoring_state.input_hash` exactly; both schema
    versions at v6.

- **Medium confidence:**
  - `answer_span_selected − matched_random_answer_span` on
    anti-compliance margin (seed-mean −0.45 nats, nested CI [−0.86,
    −0.03]). The nested CI's upper bound is 3 × 10⁻² nats from 0, so
    a different sample draw could pull it over.
  - The *interpretation* of the cross-metric tradeoff as "the
    candidate pool contains genuinely different feature sets for
    answer-text vs prompt-end perturbation" is supported by the
    data but not proven by it. An equally consistent narrative is "the
    99 disjoint features per pool are picked from a noisier tail and
    the tradeoff is regression-to-the-mean noise dressed up as
    structure". A direct causal experiment (apply only the 99
    disjoint-to-utility features and re-measure both metrics) would
    discriminate between these hypotheses; not run yet.

- **Lower confidence:**
  - The compliance trend (`answer_span − noop = +0.95 pp`,
    seed-mean +0.82 pp). Treating the naive seed bootstrap CI as
    population-level evidence would over-claim; the nested CI includes
    0. Best read: "no population-level compliance effect; the per-seed
    point estimates happen to cluster at +1 pp this draw, worth
    re-checking on additional seeds or a different held-out split".
  - The "answer-span concentrates more in layer 20" observation.
    44 % vs 41 % is real, but the candidate pool is itself 37 % layer
    20; the 3-pp gap is consistent with both "answer-span features
    really live downstream" and "the selector is mildly noisier and
    over-samples the largest layer".

- **Explicitly out of scope:**
  - "SAE steering cannot work on FaithEval." Would require a wider
    layer / wider operator / wider model sweep. L3 remains
    *partially* addressed.
  - "Answer-span margin is a strictly better selection objective."
    The cross-metric anti-compliance loss (−0.46 nats vs utility) is
    direct evidence against this, and we have not run an L1 (multi-
    model) confirmation.
  - "The cross-metric inconsistency is generalisable to other tasks."
    Single-task observation on FaithEval; not yet a pattern claim.

- **Methodological caveats:**
  - Same-metric circularity on `faitheval_answer_span_margin` (§3.4
    caveat). Cite the cross-metric (anti-compliance, compliance)
    contrasts as the conservative test of the answer-span selector.
  - Augment k=154 still has only 3 seeds (unchanged from 2026-04-22).
    Item 7 of that review's §5 remains open.
  - Two stale provenance sidecars in the heldout tree:
    `heldout/faitheval/matched_random_seed_0/experiment/run_intervention.provenance.20260423_171837.json`
    (`status="running"`) and
    `heldout/faitheval/noop/experiment/run_intervention.provenance.20260422_125700.json`
    (`status="interrupted"`, `error="KeyboardInterrupt"`). Both are
    leftover from earlier failed attempts; the canonical alpha
    `jsonl`s for both directories are backed by completed sidecars
    that arrived later. Cosmetic only; the existing 2026-04-22 review's
    claim that "all sidecars have status=completed" is technically
    inaccurate after these stale leftovers and should be tightened to
    "every canonical run has a completed sidecar".

## 5. Suggested next steps (ranked)

1. **Causal disjoint-feature experiment.** Apply only the 99
   features in `answer_span_selected ∖ utility_selected` (and
   separately, only the 99 in `utility_selected ∖ answer_span_selected`)
   under the same `delta_only` α=0 operator. Re-measure on all three
   metrics. Predicts: the disjoint-to-utility set drives the
   `answer_span − utility = +0.46` anti-compliance margin loss; the
   disjoint-to-answer-span set drives the `utility` advantage. ≈ 30 min
   GPU. Discriminates "real cross-metric structure" from "noisy-tail
   features" hypothesis. Highest scientific upside.
2. **10-seed extension of the augment k=154 bundle.** Carries over
   from 2026-04-22 §5 item 7. ≈45 min GPU. Unblocks a proper seed-
   mean CI on the augment selection-specific margin claim.
3. **Move the cross-metric tradeoff (claim 2 in §3.6) into the paper's
   main line.** This is a standalone observation: "within the same
   probe-nonzero pool, two intervention-aware selectors pick feature
   sets with opposite signed effects on the two margin metrics, with
   the cross-metric contrast bounded at +0.46 nats [+0.16, +0.75]". It
   does not depend on matched-random or path-drift constructions. It
   is more striking than the
   `readout − noop = +0.92 nats` finding (already recommended for
   main-text move in 2026-04-22 §5 item 3) because both endpoints
   *here* are intervention-aware: the bug is not "readout is bad" but
   "different intervention objectives within the same SAE basis are
   non-aligned".
4. **Sharpen the L2 closure paragraph in
   `paper/icml/reviews/TODO_Limitations_Fixes.md`.** Add a one-line
   pointer to this report and the cross-metric tradeoff. Done in this
   commit cycle.
5. **Cosmetic: stale sidecar cleanup.** The two non-completed sidecars
   in §4 should be moved into a `experiment_<date>_aborted_attempts/`
   sibling directory (per `data/AGENTS.md`'s convention for archived
   re-runs) so the audit-line claim "all sidecars have
   status=completed" can stay clean. Low priority; numbers are not
   affected.
6. **L3 minimal layer extension.** Unchanged from 2026-04-22 §5 item
   5: pick one new layer outside the existing extraction set, rerun
   the selector pipeline, check whether compliance moves. The
   answer-span result reinforces that the existing layer set already
   contains *some* metric-specific features; whether wider coverage
   contains a compliance handle remains the open L3 question.

## 6. Provenance integrity

Verified independently of the report code on 2026-04-25:

- **Sample-ID parity.** Every alpha `jsonl` in
  `heldout/{faitheval,faitheval_anti_compliance_margin,faitheval_answer_span_margin}/<family>/experiment/`
  contains exactly the 840 IDs in
  `selector/test_manifest.json` (fingerprint `781fd7eafa5f2573`).
  Mismatches: 0 over 84 directories.
- **Provenance sidecars.** 92 sidecars in `heldout/`; 90 with
  `status="completed"`. Two non-completed (one "running", one
  "interrupted") are stale leftovers; all 84 canonical alpha `jsonl`s
  are backed by a completed sidecar (verified by walking each
  experiment directory and matching the most-recent
  `completed_at_utc` provenance file's `args.alphas` to the present
  `alpha_*.jsonl` filenames).
- **Manifest fingerprints.** All 10 `matched_random_answer_span_seed_X`
  manifests have distinct `flat_idx` fingerprints (listed in §1.2).
  Pairwise per-feature Jaccard overlaps follow the same draw-diversity
  pattern as the original `matched_random` (negligible accidental
  overlap relative to the 112 004-feature eligible pool).
- **Selector hash linkage.** `selector_summary.selector_scoring.input_hash
  = 83e9fdbb65d9a3de…` matches `selector_scoring_state.input_hash`
  exactly. The `selector_stage_complete` invariant in commit d335492
  enforces this match on every re-entry.
- **Schema versions.** Both `selector_summary.json` and
  `report/heldout_summary.json` are `*_selector*/v6` (or
  `*_selector_report/v6`); audit note + provenance sidecars include
  the version markers. Augment is unchanged at
  `faitheval_sae_utility_positive_augment/v2`.
- **Re-derived numbers vs report.** 14 family means and 14 compliance
  counts on each of three metrics reproduce from raw
  `alpha_*.jsonl` files to floating-point precision. Per-seed paired
  deltas across the three metrics reproduce exactly (30 contrasts
  checked end-to-end). The seed-mean estimate, seed sd, and seed
  range for the answer-span margin contrast are
  identical to the report values.
- **`audit_ci_coverage.py`** passes with zero diagnostics.
