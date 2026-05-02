# Opus 4.7 Canary — Validation Summary (pipeline-only)

This summary intentionally contains **no outcome content**: no per-condition or
per-alpha label distributions, no selected-vs-control comparisons, no example
rows that could leak outcome patterns. It documents only that the canary
pipeline executed end-to-end and that validation invariants held.

## Package

- Path: `data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/prospective_effect_calibrated_open_20260429/opus47_canary_200_blind_labels/`
- Sampling seed: `20260502`
- Total rows: **200**
- Stratification: `dataset` (`truthfulqa`, `triviaqa_bridge`) × `condition_family` (`selected`, `random_head`, `random_direction`) × `alpha` (`0.0`, `8.0`)
  — 12 strata, 8 strata × 17 + 4 strata × 16.
- Subagent rater chunks: **15** (5 chunks of 14 rows + 10 chunks of 13 rows).

## Hash bindings

- `review_cases_blind.jsonl` sha256: `0592dfbc63d31e7fbd5c9d88c1809d876136292aa1f327107f37ec040e44b734`
- `rubric.md` sha256: `80f9baf09d757a9296c931b6973e633571974411e3ec421e6b512dba53df6f6f`
  (matches the frozen `prospective_open_calibration_gate_20260429/rubric.md`)
- Authority manifest sha256: `6a7f7ef92ef2eeabef09f0c633f5ae1cc54c38eea9ec4d137e5a939388074d8a`
  (`prospective_effect_run_gate_20260429_r2_external_labels/effect_run_manifest.json`)
- Merged labels file: `prospective_open_labels_opus47_canary.jsonl`
  sha256: `9a6406dafdf68a0e605edb6ccf78130a5fd4c0c2a9bb6fbb4a5af3ec9413b588`

## Rater configuration

- Model: `claude-opus-4.7`, effort: `max`.
- Each subagent received only its chunk file, the frozen rubric, and the label schema.
- Subagents did NOT see: `condition`, `alpha`, `dataset`, `sample_id`,
  `base_sample_id`, the private case map, deterministic open-grade fields, or
  any pre-existing adjudication labels.

## Validation result

**PASS**. All of the following invariants held:

- exactly 200 labels in the merged file
- no duplicate `blind_case_id` across labels
- every `blind_case_id` in `review_cases_blind.jsonl` has exactly one label
- every label maps to a known `blind_case_id`
- `review_order` covers `1..200` exactly across the merged set
- no label row contains private fields (`sample_id`, `base_sample_id`,
  `dataset`, `condition`, `alpha`, `run_row_sha256`,
  `authority_manifest_sha256`)
- every label has the constant `blind_cases_file_sha256` and `rubric_sha256`
  pinned to the package's hash invariants
- every label's `schema_version` is `simid_prospective_effect_open_label/v1`
- every `label` is one of `CORRECT` / `INCORRECT` / `NOT_ATTEMPTED`
- every `confidence` is an integer in `1..5`
- every `rule_gap` is a boolean
- every `flags` array uses only allowed enum values, with no duplicates within
  a row
- every `rater` block names model `claude-opus-4.7`, type `llm`, and the
  expected `opus47_canary_subagent_NN` id for the chunk

No schema or rubric issues were encountered during rating; no rater reported a
parse error or had to retry.

## Wall time

- Build phase (sample, blind/private split, manifest, label schema, README,
  chunking): a few seconds.
- Rating phase (15 subagent raters in two parallel waves of 8 and 7): ~100 s
  bounded by the slowest single chunk in each wave (max single-chunk duration
  ≈ 104 s).
- Merge + validation: a few seconds.
- End-to-end: well under 5 minutes wall time, ~15 subagent invocations.

## Status

The canary executed cleanly: blinding held, the rubric was hash-bound to the
frozen gate rubric, the label schema validated, and full coverage was achieved
without rater retries. The pipeline is **ready to scale** to a full API-batch
adjudication run for the prospective effect dataset.

This canary is **not** claim-bearing. It does not stand in for the external
open label package referenced by `effect_run_manifest.json`, and it must not be
used to compute any selected-vs-control deltas, rubric revisions, or
pass/fail decisions on the prospective effect gates.
