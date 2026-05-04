# External open labels — selected / TruthfulQA / alpha {0, 8}

This package extends the SIMID prospective open-label set for the highest-ROI
partial scope needed to inform the pre-registered headline gate
`primary_truthfulqa_open_delta_positive`.

Canonical review:
[`paper/icml/reports/2026-05-03-simid-prospective-partial-external-label-review.md`](../../../../../../paper/icml/reports/2026-05-03-simid-prospective-partial-external-label-review.md).

- Condition: `selected`
- Dataset: `truthfulqa`
- Alphas: `[0.0, 8.0]`
- Seed: `20260503`
- Total scope rows after exclusions: `908`
- Net-new rows to rate: `876`
- Reused canary labels: `32`
- Sample-id pairs across alphas: `454`
- Distinct base_sample_id clusters across alphas: `227`

## Important caveats

- This is a **partial** external label package. It covers only
  `selected/truthfulqa/0.0` and `selected/truthfulqa/8.0`.
- `review_cases_blind.jsonl` contains only net-new rows for raters.
- `review_cases_all_blind.jsonl` is the merged blind roster used to hash-bind the
  final merged labels, including canary reuse rows.
- `private_case_map.jsonl` must not be shared with raters.
- `reused_canary_labels.jsonl` carries forward the canary judgments without
  re-rating, while preserving the original canary `rater` blocks unchanged.
- `rubric.md` is a verbatim copy of the frozen rubric and must not be edited.

## Directory contents

- `review_cases_blind.jsonl` — new-label rows only.
- `review_cases_all_blind.jsonl` — full-scope blind roster for merged-label hash binding.
- `private_case_map.jsonl` — private mapping for all scope rows.
- `reused_canary_labels.jsonl` — canary overlap rows rewritten onto package blind ids.
- `rater_chunks/` — package-level copies of chunk case files.
- `rater_jobs/` — per-chunk isolated workspaces used for Opus rating.
- `rater_labels/` — validated chunk label files copied out of job sandboxes.
- `prospective_open_labels_external_selected_truthfulqa_alpha_0_8.jsonl` — merged final labels (written after rating).
- `early_look_paired_delta_analysis.*` — post-label directional paired analysis.
- `rule_gap_triage/` — triage outputs for any newly surfaced malformed rows.
