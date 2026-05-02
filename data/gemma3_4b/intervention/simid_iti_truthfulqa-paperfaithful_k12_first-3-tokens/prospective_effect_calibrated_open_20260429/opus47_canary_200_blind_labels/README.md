# Opus 4.7 Canary — 200-row blinded SIMID prospective effect labels

This is a **pipeline / rubric / cost-risk canary** for Opus 4.7 max adjudication on
the SIMID prospective effect run. It is **not** a claim-bearing label set and it
does **not** upgrade the prospective open grading authority by itself.

- Seed: `20260502`
- Total rows: 200
- Alphas sampled: [0.0, 8.0]
- Datasets sampled: ['truthfulqa', 'triviaqa_bridge']
- Condition families sampled: ['selected', 'random_head', 'random_direction']
- Source run dir: `data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/prospective_effect_calibrated_open_20260429`
- Authority manifest sha256: `6a7f7ef92ef2eeabef09f0c633f5ae1cc54c38eea9ec4d137e5a939388074d8a`
- Frozen rubric sha256: `80f9baf09d757a9296c931b6973e633571974411e3ec421e6b512dba53df6f6f`
- review_cases_blind.jsonl sha256: `0592dfbc63d31e7fbd5c9d88c1809d876136292aa1f327107f37ec040e44b734`

## What is in this directory

- `review_cases_blind.jsonl` — 200 blinded rows. Whitelist-only fields:
  `blind_case_id`, `review_order`, `question`, `gold_aliases`, `predicted_answer`.
- `private_case_map.jsonl` — private mapping back to `sample_id`, `condition`, `alpha`,
  etc. **MUST NOT** be shared with any rater.
- `rubric.md` — verbatim copy of the frozen rubric, hash-bound to the source.
- `label_schema.json` — JSONSchema for returned labels.
- `review_manifest.json` — manifest with all hash bindings and design.
- `prospective_open_labels_opus47_canary.jsonl` — merged labels (written after rating).
- `canary_validation_summary.md` — pipeline-only summary; no outcome content.

## Rules for the rating phase

Raters see only their assigned blind rows + `rubric.md` + `label_schema.json`.
Raters never see condition, alpha, dataset, sample_id, base_sample_id, the private
map, or any pre-existing adjudication labels.

Each label row must include all required fields from `label_schema.json`, with
`blind_cases_file_sha256 = 0592dfbc63d31e7fbd5c9d88c1809d876136292aa1f327107f37ec040e44b734` and
`rubric_sha256 = 80f9baf09d757a9296c931b6973e633571974411e3ec421e6b512dba53df6f6f`.

## What this canary will NOT be used for

- It is not used to compute selected-vs-control deltas.
- It is not used to declare any prospective open-correctness claim.
- It is not used to revise the rubric.
- It does not act as the external open label package referenced by the authority
  manifest.
