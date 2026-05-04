# SIMID Prospective Partial External-Label Review (2026-05-03)

> Canonical review for the unstaged partial external-label package at
> `data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/prospective_effect_calibrated_open_20260429/external_open_label_package_selected_truthfulqa_alpha_0_8/`.
> This report updates the SIMID prospective-effect status only. It does not
> supersede the 2026-04-28 calibration review for historical MVP claimability.

## Bottom Line

The partial Opus external-label package is internally useful but not
claim-bearing. It covers only `selected` / `truthfulqa` / alpha `0.0` versus
`8.0`: 908 blind-labeled rows, 454 paired sample IDs, and 227 base-sample
clusters. Package-level validation passes, the private map matches the
underlying run rows by hash, the blind rows contain no condition/alpha/sample
private fields, and no rule gaps were reported.

The effect read is negative for the pre-registered primary gate. External-label
open correctness moves from 155/454 = 34.14% at alpha 0 to 173/454 = 38.11% at
alpha 8, a paired clustered-bootstrap delta of +3.96 pp with 95% CI
[-1.98, +9.91]. The r2 gate required lower 95% CI > 0 and point estimate at
least +5 pp. It clears neither part.

There is also a substantive externality warning on the same selected
TruthfulQA rows: MC letter-likelihood accuracy drops from 198/454 to 184/454,
delta -3.08 pp with 95% CI [-5.95, -0.66]. Under the r2 MC-degradation rule
(`lower CI >= -2 pp`), this scope would fail the blocker. Open attempted rate
also drops sharply, from 420/454 = 92.51% to 381/454 = 83.92%, delta
-8.59 pp with 95% CI approximately [-13.44, -3.74]. That means the observed
small correctness movement is not an attempt-rate increase artifact, but it is
consistent with a more cautious or abstention-prone output distribution.

Do not use this package to claim that SIMID/ITI improves TruthfulQA
truthfulness. The strongest defensible update is narrower: under the frozen
prospective rubric and Opus LLM labels, the selected alpha 8 versus alpha 0
TruthfulQA slice is at best weak positive on open correctness, statistically
uncertain, below the minimum practical effect, and accompanied by measurable
MC degradation and reduced attempt rate.

## Data Authority

| Artifact | Path |
|---|---|
| Parent run dir | `data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/prospective_effect_calibrated_open_20260429/` |
| Partial label package | `external_open_label_package_selected_truthfulqa_alpha_0_8/` |
| Package manifest | `external_open_label_package_selected_truthfulqa_alpha_0_8/review_manifest.json` |
| Full-scope blind roster | `external_open_label_package_selected_truthfulqa_alpha_0_8/review_cases_all_blind.jsonl` |
| New-label blind roster | `external_open_label_package_selected_truthfulqa_alpha_0_8/review_cases_blind.jsonl` |
| Private map | `external_open_label_package_selected_truthfulqa_alpha_0_8/private_case_map.jsonl` |
| Merged labels | `external_open_label_package_selected_truthfulqa_alpha_0_8/prospective_open_labels_external_selected_truthfulqa_alpha_0_8.jsonl` |
| Package validation | `external_open_label_package_selected_truthfulqa_alpha_0_8/validation_summary.json` |
| Early-look analysis | `external_open_label_package_selected_truthfulqa_alpha_0_8/early_look_paired_delta_analysis.json` |
| Frozen r2 protocol | `mvp_20260427_calibration/human_review_package/prospective_effect_run_gate_20260429_r2_external_labels/protocol.md` |
| Frozen r2 manifest | `mvp_20260427_calibration/human_review_package/prospective_effect_run_gate_20260429_r2_external_labels/effect_run_manifest.json` |
| Final run provenance | `prospective_effect_calibrated_open_20260429/run_simid.provenance.20260502_101731.json` |

Literature guardrails: ITI itself tunes alpha/K under a truthfulness/helpfulness
trade-off and uses generated-answer judging rather than substring matching on
TruthfulQA (`papers/Inference-Time Intervention:Eliciting Truthful Answers from
a Language Model2306.03341v6.md`). Recent steering-evaluation work argues for
open-ended-context evaluation, likelihood/confidence evidence, and informative
baselines (`papers/reliable-steering-eval-2410.17245.md`), while steering
reliability audits warn that aggregate steering can hide high per-input
variance and even opposite-direction examples
(`papers/steering-vectors-reliability-2407.12404.md`). Those cautions apply
directly here.

## Data

### Parent Run Shape

The effect grid completed on 2026-05-02 from 10:17:32 to 11:13:57 UTC with
`git_dirty=false` at git SHA `15abf30975cdf8fb98aae5db11faf4bf143f6c19`.
The final provenance records `n_manifest_rows=854`, `n_rows=854`, and 44
condition-alpha output targets. I verified every condition/alpha JSONL has 854
rows.

The run matches the r2 design at the generation level:

| Field | Value |
|---|---|
| Model | `google/gemma-3-4b-it` |
| ITI artifact | `data/contrastive/truthfulness/iti_truthfulqa_paperfaithful/final_fold0/iti_heads.pt` |
| ITI config | paper-faithful TruthfulQA, K=12, `first_3_tokens` |
| Conditions | `selected`, five `random_head_seed*`, five `random_direction_seed*` |
| Alphas | `[-8.0, 0.0, 4.0, 8.0]` |
| Rows per condition-alpha | 854 = 454 TruthfulQA rows + 400 Bridge rows |
| Locked manifest source | `data/manifests/simid_truthfulqa_bridge_prospective_effect_20260429.json` |
| Locked manifest SHA-256 | `d5c6f6be2d003751549ffbbdccdc8165e73c7ff3014dfa51603614cc89873156` |

There are two earlier interrupted run-provenance files in the same directory.
They are append-only provenance, not evidence of output corruption. The final
completed provenance is the authority for the finished grid.

### Partial Label Package Shape

The unstaged package is explicitly partial:

| Field | Value |
|---|---:|
| Condition | `selected` |
| Dataset | `truthfulqa` |
| Alphas | `0.0`, `8.0` |
| Total labeled rows | 908 |
| Paired sample IDs | 454 |
| Base-sample clusters | 227 |
| New Opus-rated rows | 876 |
| Reused canary labels | 32 |
| Rater chunks | 68 |
| Rating exit code | 0 |
| Package rule gaps | 0 |

The rater-visible blind rows contain only `blind_case_id`, `review_order`,
`question`, `gold_aliases`, `predicted_answer`, and `schema_version`. The extra
`schema_version` is not private leakage, but it is not in the strict r2
full-package loader's blind-row key set.

I independently verified:

- `review_cases_blind.jsonl`, `review_cases_all_blind.jsonl`,
  `private_case_map.jsonl`, and `rubric.md` match their manifest hashes.
- `review_cases_all_blind.jsonl`, `private_case_map.jsonl`, and merged labels
  cover the same 908 blind IDs exactly once.
- Every label is bound to the full-scope blind-roster SHA-256
  `ad8874be4f6e728ede73b2c9a989319f42a239f72abdf5b6064e7b69bc26be42` and rubric
  SHA-256 `80f9baf09d757a9296c931b6973e633571974411e3ec421e6b512dba53df6f6f`.
- Every private map row matches the selected TruthfulQA alpha 0/8 run row by
  `prospective_effect_run_row_sha256`, question text, aliases, and open
  response.
- Pairing is complete: 454/454 sample IDs have both alpha 0 and alpha 8 labels;
  every base-sample cluster has the expected two option-order rows at each
  alpha.
- Blinding is mostly well randomized at the sample-pair level: only 3/454
  alpha-paired sample IDs were rated by the same subagent; 12/227 base clusters
  had any subagent see more than one of the four rows.

One reproducibility gap remains: the package manifest records the label output
path and expected counts, but not the merged label file's SHA-256. The labels
themselves are hash-bound to blind cases and rubric, so this is not a content
integrity failure, but the manifest is weaker than it should be for archival
single-file verification.

### External-Label Results

Label histograms:

| Alpha | CORRECT | INCORRECT | NOT_ATTEMPTED | Attempted |
|---:|---:|---:|---:|---:|
| 0.0 | 155 | 265 | 34 | 420 |
| 8.0 | 173 | 208 | 73 | 381 |

Primary paired result:

| Metric | Alpha 0 | Alpha 8 | Paired delta | 95% CI |
|---|---:|---:|---:|---:|
| External open correct | 34.14% | 38.11% | +3.96 pp | [-1.98, +9.91] |
| External attempted | 92.51% | 83.92% | -8.59 pp | [-13.44, -3.74] |

Paired label transitions:

| Alpha 0 label | Alpha 8 label | Rows |
|---|---|---:|
| INCORRECT | INCORRECT | 173 |
| CORRECT | CORRECT | 115 |
| INCORRECT | CORRECT | 48 |
| INCORRECT | NOT_ATTEMPTED | 44 |
| CORRECT | INCORRECT | 24 |
| CORRECT | NOT_ATTEMPTED | 16 |
| NOT_ATTEMPTED | NOT_ATTEMPTED | 13 |
| NOT_ATTEMPTED | INCORRECT | 11 |
| NOT_ATTEMPTED | CORRECT | 10 |

Binary correctness transitions are therefore 58 open-wrong-to-correct and
40 correct-to-open-wrong rows, net +18/454. Binary attempted transitions are
60 attempted-to-not-attempted and 21 not-attempted-to-attempted rows, net
-39/454.

The most common flags across all labels were:

| Flag | Count |
|---|---:|
| `truthfulqa_qualified_answer_boundary` | 283 |
| `wrong_extra_answer` | 107 |
| `truthfulqa_non_answer_boundary` | 88 |
| `other_boundary` | 31 |
| `alias_too_broad_or_too_narrow` | 22 |
| `bridge_partial_entity_or_modifier` | 10 |
| `multiple_candidates_no_commitment` | 4 |

The `bridge_partial_entity_or_modifier` flag appears despite the package being
TruthfulQA-only because the flag vocabulary is shared with the frozen rubric;
it does not indicate Bridge rows in this package.

### Deterministic And MC Cross-Checks

On the same 454 selected TruthfulQA sample-pairs, deterministic alias grading
is again unusable as an open-correctness endpoint:

| Grader | Alpha 0 correct | Alpha 8 correct |
|---|---:|---:|
| Deterministic alias | 24/454 | 22/454 |
| External Opus labels | 155/454 | 173/454 |

The old deterministic alias result would have read as a small negative
movement. The external rubric reads as a small positive but uncertain movement.
That discrepancy is exactly why TruthfulQA open correctness remains
judge/rater-gated.

MC letter-likelihood accuracy on the same selected TruthfulQA rows is:

| Metric | Alpha 0 | Alpha 8 | Paired delta | 95% CI |
|---|---:|---:|---:|---:|
| MC letter-likelihood correct | 198/454 | 184/454 | -3.08 pp | [-5.95, -0.66] |

This MC result is independent of external open-label judging. It is therefore a
real warning about target-compatible multiple-choice degradation on the primary
TruthfulQA slice.

### Canonical Loader Compatibility

The current package cannot be loaded by the claim-bearing
`scripts/analyze_simid.py --require-prospective-open-authority` path. A direct
test fails immediately with:

```text
ValueError: .../review_manifest.json: unexpected label package schema_version
```

This is the first failure only. Additional incompatibilities are visible from
static inspection:

- Package schema is `simid_prospective_open_external_partial_package/v1`; the
  canonical loader expects `simid_prospective_effect_open_label_package/v1`.
- The manifest uses `authority_manifest`, `frozen_rubric`,
  `review_cases_blind`, and `review_cases_all_blind`; the full loader expects
  `prospective_open_authority` or compatible authority, `rubric`, and a `files`
  object with `review_cases_blind` and `private_case_map`.
- Private rows use schema
  `simid_prospective_open_external_private_case/v1`; the full loader expects
  `simid_prospective_effect_open_private_case/v1`.
- Label rows use `rater.type="llm"` and include `rater.model`. The full loader
  currently accepts only `human`, `external_human`, `expert`, or
  `external_expert`, and rejects model-bearing rater blocks.
- The package covers 908 rows. The full effect run has 37,576 open-eligible
  rows. Even an alpha 0/8 selected-only analysis would include Bridge rows
  unless the analysis code gained an explicit dataset filter and partial-scope
  contract.

These are good guardrails for claimability: the partial package cannot
accidentally pass through the full claim-bearing path.

## Interpretation

### What Withstands Scrutiny

The partial package is a valid early-look measurement artifact for the narrow
selected/TruthfulQA alpha 8 versus alpha 0 question. Its internal hash binding,
private-map/run-row match, label coverage, and no-leakage properties are
strong enough to trust the reported early-look numbers descriptively.

The primary r2 effect gate does not pass. This is not a close technicality:
the point estimate is below the pre-specified +5 pp minimum practical effect,
and the CI includes both a small negative effect and a nearly +10 pp positive
effect. The evidence is therefore compatible with weak benefit, no effect, or
moderate benefit, but not with a supported claim under the frozen rule.

The MC degradation warning is more decisive than the open-correctness gain. On
the same paired selected/TruthfulQA rows, alpha 8 worsens the lettered
TruthfulQA endpoint by about 3 pp with a CI below zero. Since the r2 contract
requires target behavior and safety/externality checks together, this alone
would block a clean truthfulness-improvement claim on the current evidence.

### What Is Scientifically Interesting

Alpha 8 appears to make selected ITI more cautious on this slice. The evidence
is the large attempted-rate drop and the label-transition texture: many alpha 8
rows move from wrong commitments to either correct anti-myth answers or
non-attempts, while a nontrivial number of alpha 0 correct rows become
incorrect or non-attempted. This resembles the truthfulness/helpfulness tradeoff
emphasized in ITI rather than a pure knowledge-retrieval improvement.

The flag distribution makes the hard boundary explicit. `truthfulqa_qualified_answer_boundary`
dominates, and many flips concern anti-myth or hedged-answer cases. This is the
right failure family to audit if the project continues: not generic alias
matching, but whether the rubric is consistently distinguishing qualified
truthful answers from evasive non-answers and myth-aligned half answers.

The deterministic alias cross-check is useful only as a negative control for
measurement. It dramatically undercounts external-label correctness and even
gets the alpha direction wrong on this slice. This further supports the current
measurement-blueprint rule that TruthfulQA open correctness cannot be
claim-bearing under substring aliases.

### What Does Not Withstand Scrutiny

No specificity claim is available. The unstaged labels do not cover any
random-direction seed, any random-head seed, or Bridge externality rows.
Deterministic control summaries and margin slopes cannot substitute for the
external-label control gates because the claim endpoint is external open
correctness under the frozen rubric.

No full r2 claimability claim is available. The package is partial by design,
uses a schema outside the canonical full-package loader, and uses LLM rater
blocks that the current full loader explicitly rejects. If LLM external labels
are intended to be allowed for the effect run, that policy needs to be changed
deliberately in the protocol and code before any claim-bearing analysis is
rendered.

No broad ITI truthfulness claim is available. Even a future full-package pass
would be scoped to this fresh Gemma 3 4B SIMID battery, this ITI artifact, this
alpha grid, this rubric, and these controls. The current partial package falls
well short of even that limited scope.

## Uncertainties

| Uncertainty | Direction | Severity |
|---|---|---|
| Single rater family | All new labels are Claude Opus 4.7 max subagent labels; the earlier prospective gate showed human and Codex returns were weaker than Opus. | High for final claimability; moderate for early-look direction. |
| Partial scope | No external labels for controls or Bridge externality. | Decisive for specificity and full claimability. |
| Package-loader mismatch | Partial schema cannot be loaded by the canonical claim-bearing path. | Decisive for claimability; helpful as an accidental-overclaim guard. |
| MC degradation | Same-scope MC letter accuracy drops with CI below zero. | High; likely claim-blocking unless the protocol is changed and justified. |
| Rubric boundary sensitivity | Qualified/non-answer TruthfulQA boundaries dominate flags and flips. | Moderate to high; needs rater-diversity audit before any strong statement. |
| Canary reuse | 32/908 labels were reused from the Opus canary. | Low for point estimate; document for independence assumptions. |
| Same-subagent repeated exposure | 3/454 alpha-paired sample IDs had both alpha rows rated by the same subagent. | Low, but worth preserving as a blinding audit field. |
| Full-run provenance | Final run is complete and clean, but two interrupted provenance sidecars remain. | Low; append-only provenance is acceptable. |

## Recommended Next Steps

1. Do not promote SIMID r2 to an ICML claim. The partial package misses the
   primary open-correctness gate and surfaces an MC degradation warning.

2. Decide whether more label spend is justified. Given Gate 1 and MC both fail
   on the highest-ROI selected/TruthfulQA slice, the conservative scientific
   choice is to stop unless the goal is diagnostic understanding rather than
   claim rescue.

3. If continuing diagnostically, audit a small rater-diversity subset before
   full labeling. Sample from `truthfulqa_qualified_answer_boundary`,
   `truthfulqa_non_answer_boundary`, and true-to-false / false-to-true flip
   families. Compare Opus, Codex, and human/expert labels under the frozen
   rubric.

4. If continuing claim-bearing work, build a tracked full-package exporter and
   validator rather than preserving this partial schema as the claim path. The
   full package should use `simid_prospective_effect_open_label_package/v1`,
   include a manifest SHA-256 for the merged labels, bind the rater policy, and
   cover every open-eligible row required by the r2 analysis contract.

5. Resolve the rater-policy mismatch before any more labels are collected. If
   Opus LLM labels are intended to count as external effect labels, update the
   frozen protocol and `analyze_simid.py` explicitly. If the current code is the
   policy, future effect labels need human/expert rater blocks with no `model`
   field.

6. Keep historical SIMID MVP open metrics diagnostic-only. This partial
   prospective package does not alter the failed historical calibration gate and
   should not be used to retrofit earlier results.
