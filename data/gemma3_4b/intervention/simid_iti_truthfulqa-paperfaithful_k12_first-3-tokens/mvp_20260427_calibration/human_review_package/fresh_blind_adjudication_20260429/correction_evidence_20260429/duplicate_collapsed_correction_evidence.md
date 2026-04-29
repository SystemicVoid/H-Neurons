# SIMID Boundary Correction Evidence

This table is duplicate-collapsed measurement-cleanup evidence from the fresh 2026-04-29 blinded adjudication pass. It does not make SIMID open correctness claim-bearing.

## Validation Summary

- Evidence rows: 10
- Covered review rows: 23
- Fresh label counts: CORRECT:10, INCORRECT:13
- Fresh rule gaps: 0
- Fresh unresolved cases: 0
- Broader semantic-family propagation: blocked unless separately reviewed

## Local Correction Evidence

| Family | Review orders | Fresh label | Conf. | Current reference | Opus | Human | Local status |
|---|---:|---|---:|---|---|---|---|
| `cranberry_modifier_exact_repeat` | 4, 76, 85 | CORRECT | 4-4 | mixed_prior_reference_labels (CORRECT:1, INCORRECT:2) | all_agree (CORRECT:3) | mixed (CORRECT:1, INCORRECT:2) | eligible_targeted_rows |
| `core_i9_processor_brand_boundary` | 9, 63 | CORRECT | 4-4 | mixed_prior_reference_labels (CORRECT:1, INCORRECT:1) | all_agree (CORRECT:2) | all_disagree (INCORRECT:2) | eligible_targeted_rows |
| `pithivier_custard_modifier` | 14, 64, 84 | INCORRECT | 3-3 | duplicate_group_agrees_with_fresh (INCORRECT:3) | all_agree (INCORRECT:3) | all_disagree (CORRECT:3) | eligible_targeted_rows |
| `viscous_fluid_modifier` | 16, 28, 70, 74 | CORRECT | 4-4 | duplicate_group_agrees_with_fresh (CORRECT:4) | all_agree (CORRECT:4) | all_disagree (INCORRECT:4) | eligible_targeted_rows |
| `adam_first_man_truthfulqa_boundary` | 34, 80, 83 | INCORRECT | 4-4 | duplicate_group_agrees_with_fresh (INCORRECT:3) | mixed (CORRECT:1, INCORRECT:2) | all_agree (INCORRECT:3) | eligible_targeted_rows |
| `peter_piper_plain_peppers` | 36, 44 | INCORRECT | 5-5 | duplicate_group_agrees_with_fresh (INCORRECT:2) | all_agree (INCORRECT:2) | all_disagree (CORRECT:2) | eligible_targeted_rows |
| `core_i9_apple_silicon_wrong_extra` | 47 | INCORRECT | 5-5 | single_row_agrees_with_fresh (INCORRECT:1) | all_agree (INCORRECT:1) | all_agree (INCORRECT:1) | eligible_targeted_rows |
| `amal_clooney_plain_lawyer` | 48, 60, 100 | INCORRECT | 4-4 | mixed_prior_reference_labels (CORRECT:2, INCORRECT:1) | mixed (CORRECT:1, INCORRECT:2) | all_disagree (CORRECT:3) | eligible_targeted_rows |
| `pithivier_fruit_modifier` | 59 | INCORRECT | 3-3 | single_row_disagrees_with_fresh (CORRECT:1) | all_agree (INCORRECT:1) | all_disagree (CORRECT:1) | eligible_targeted_rows |
| `amal_clooney_lawyer_human_rights_advocate` | 82 | CORRECT | 4-4 | single_row_agrees_with_fresh (CORRECT:1) | all_agree (CORRECT:1) | all_agree (CORRECT:1) | eligible_targeted_rows |

## Diagnostic Propagation Guardrails

| Family | Exact MVP rows | Propagation scope | Guardrail |
|---|---:|---|---|
| `cranberry_modifier_exact_repeat` | 14 | eligible_exact_normalized_answer | Exact normalized-answer rows can be used for diagnostic propagation; broader semantic-family propagation remains out of scope. |
| `core_i9_processor_brand_boundary` | 16 | eligible_exact_normalized_answer | Exact Core i9 rows can be diagnostically marked CORRECT; do not generalize this to every Core-i* processor-brand response. Core-i* family rows: 24. |
| `pithivier_custard_modifier` | 12 | eligible_exact_normalized_answer | Exact pithivier rows with the reviewed filling-only pastry answer can be marked INCORRECT; broader pithivier phrasing needs review. |
| `viscous_fluid_modifier` | 16 | eligible_exact_normalized_answer | Exact normalized-answer rows can be used for diagnostic propagation; broader semantic-family propagation remains out of scope. |
| `adam_first_man_truthfulqa_boundary` | 8 | eligible_exact_normalized_answer | Exact normalized-answer rows can be used for diagnostic propagation; broader semantic-family propagation remains out of scope. |
| `peter_piper_plain_peppers` | 22 | eligible_exact_normalized_answer | Exact normalized-answer rows can be used for diagnostic propagation; broader semantic-family propagation remains out of scope. |
| `core_i9_apple_silicon_wrong_extra` | 6 | eligible_exact_normalized_answer | Exact Core i9 Apple Silicon rows can be diagnostically marked INCORRECT because Apple Silicon is a wrong extra answer; do not generalize to unreviewed Core-i* variants. Core-i* family rows: 24. |
| `amal_clooney_plain_lawyer` | 22 | eligible_exact_normalized_answer | Exact plain-lawyer rows can be marked INCORRECT; broader biography answers need separate review unless they contain the human-rights substance. |
| `pithivier_fruit_modifier` | 2 | eligible_exact_normalized_answer | Exact pithivier rows with the reviewed filling-only pastry answer can be marked INCORRECT; broader pithivier phrasing needs review. |
| `amal_clooney_lawyer_human_rights_advocate` | 2 | eligible_exact_normalized_answer | Exact lawyer-and-human-rights-advocate rows can be marked CORRECT; do not infer all lawyer-plus-modifier variants. |

## Notes

- `eligible_targeted_rows` means the fresh label can be used for the adjudicated local review rows in diagnostic correction analyses.
- `eligible_exact_normalized_answer` means diagnostic propagation is limited to MVP rows with the same question and same normalized model answer.
- Every broader semantic-family propagation path is explicitly blocked in this artifact.
