# SIMID Fresh Blind Boundary Adjudication 2026-04-29

This is a small, append-only measurement-cleanup package derived from the existing SIMID open independent review package. It focuses on exact-repeat conflicts and high-value Bridge modifier or partial-entity boundary cases. It does not mutate historical run outputs.

Rater-facing files live in `llm_blind_batches/batch_001/` and intentionally omit calibration IDs and all prior labels. The private reconciliation map is `llm_blind_case_map.jsonl` and should be used only after fresh labels are finalized.

Cases: 23
Blind cases SHA-256: `36b47414710562ddaa17002c4f1e204ecd71b1479bf307bc73653582f9df0347`

Fresh outputs:

- `llm_blind_batches/batch_001/fresh_blind_boundary_labels.jsonl` - blinded rater labels keyed by `blind_case_id`.
- `fresh_blind_boundary_labels_resolved.jsonl` - post-label labels resolved to calibration IDs.
- `validation_summary.json` - coverage, schema, duplicate-consistency, and unresolved-case checks.
- `duplicate_collapsed_boundary_status.jsonl` - duplicate-collapsed boundary rows for correction-evidence tables.
- `fresh_blind_adjudication.provenance.20260429_115543.json` - source and output hashes for this append-only package.

Validation result: 23 labels for 23 blind cases, 0 rule gaps, 0 unresolved cases, and 0 inconsistent duplicate groups across 7 duplicate groups.

Validate labels with:

```bash
uv run python scripts/validate_simid_open_review_labels.py --package-dir data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/mvp_20260427_calibration/human_review_package/fresh_blind_adjudication_20260429 --output data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/mvp_20260427_calibration/human_review_package/fresh_blind_adjudication_20260429/fresh_blind_boundary_labels_resolved.jsonl
```

Pass `--overwrite` if intentionally regenerating the resolved-label file.
