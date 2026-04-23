# Ground-Truth Evidence Pack

## Purpose

This pack keeps the JSONL ledgers exact and turns the Markdown into deterministic briefings that surface the main evidence, the dominant caveats, and the remaining provenance boundaries without duplicating raw payloads.

## How To Read This Pack

- Start with `metric_ledger.jsonl`, `example_ledger.jsonl`, and `surface_crosswalk.jsonl` if you need machine-stable evidence.
- `exact_metric` means the claim lands on a ledger metric; `exact_example` means example-only coverage; `structured_surface_only` means a structured scalar exists but is not promoted; `markdown_fallback_only` means only prose/audit context exists; `historical_only` means provenance is preserved but intentionally not promoted as live evidence.
- This build contains 201 `exact_metric` crosswalk rows and 15 ledger metrics sourced directly from markdown audits.

## Family Map

- `readout_surfaces`: `01_readout_surfaces.md`; 27 metrics; 8 examples.
- `intervention_surfaces`: `02_intervention_surfaces.md`; 45 metrics; 16 examples.
- `measurement_surfaces`: `03_measurement_surfaces.md`; 44 metrics; 9 examples.
- `transfer_externality_surfaces`: `04_transfer_externality_surfaces.md`; 44 metrics; 16 examples.
- `mechanism_diagnostic_surfaces`: `05_mechanism_diagnostic_surfaces.md`; 29 metrics; 8 examples.

## Coverage Summary

- Crosswalk rows: 4049 total.
- `exact_metric`: 201.
- `exact_example`: 27.
- `structured_surface_only`: 3817.
- `markdown_fallback_only`: 3.
- `historical_only`: 1.
- `unresolved`: 0.
- Fallback-only sources:
  - `bioasq_pipeline_audit` -> `data/gemma3_4b/intervention/bioasq/bioasq_pipeline_audit.md`.
  - `faitheval_sae_utility_selector_audit` -> `data/gemma3_4b/intervention/faitheval_sae_utility_selector/report/audit_note.md`.
  - `faitheval_sae_utility_selector_augment_audit` -> `data/gemma3_4b/intervention/faitheval_sae_utility_selector/report_augment/augment_audit_note.md`.

## Known Gaps / Claim Hygiene

- Unresolved provenance claims: none.
- Historical-only provenance rows:
  - `historical-april-8-legacy-ruler-causal-full-500-effect` remains historical provenance only.
