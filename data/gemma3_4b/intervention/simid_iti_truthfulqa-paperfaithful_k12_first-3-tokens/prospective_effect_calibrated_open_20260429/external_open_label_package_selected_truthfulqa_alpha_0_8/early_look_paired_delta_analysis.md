# Early-Look Paired Delta Analysis

This early look evaluates `primary_truthfulqa_open_delta_positive` only. It does not satisfy
`selected_exceeds_random_direction`, `selected_exceeds_random_head`, `mc_degradation_blocker`,
or the full `attempt_shift_blocker` requirement; it only partially informs the latter via the attempted-rate delta.

## Design

- Scope: `selected` / `truthfulqa` / alpha `0.0` vs `8.0`
- Total labeled rows: 908
- Paired sample_id rows: 454
- Distinct base_sample_id clusters: 227
- CI method: clustered paired bootstrap over base_sample_id clusters (10000 resamples, 95% percentile CI)

## Paired Deltas

- CORRECT rate at alpha=0: 34.14%
- CORRECT rate at alpha=8: 38.11%
- Paired CORRECT delta (alpha=8 - alpha=0): 3.96 pp (95% CI -1.98 to 9.91 pp)
- ATTEMPTED rate at alpha=0: 92.51%
- ATTEMPTED rate at alpha=8: 83.92%
- Paired ATTEMPTED delta (alpha=8 - alpha=0): -8.59 pp (95% CI -13.44 to -3.74 pp)

## Gate 1 Read

- Manifest rule: `selected alpha=8 minus selected alpha=0 paired external-label open-correctness delta has lower 95% CI > 0 and point estimate >= +5 pp`
- Read: Gate 1 does not yet clear the manifest rule on this partial package.

## Full-Scale Diagnostics

- Rule-gap count: 0
- Label histogram: {'CORRECT': 328, 'INCORRECT': 473, 'NOT_ATTEMPTED': 107}
- Confidence histogram: {2: 23, 3: 141, 4: 236, 5: 508}
- Flag histogram: {'alias_too_broad_or_too_narrow': 22, 'bridge_partial_entity_or_modifier': 10, 'multiple_candidates_no_commitment': 4, 'other_boundary': 31, 'truthfulqa_non_answer_boundary': 88, 'truthfulqa_qualified_answer_boundary': 283, 'wrong_extra_answer': 107}
