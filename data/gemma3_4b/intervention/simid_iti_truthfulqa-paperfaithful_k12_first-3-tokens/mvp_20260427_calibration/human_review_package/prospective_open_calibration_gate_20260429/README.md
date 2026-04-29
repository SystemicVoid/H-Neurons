# Prospective SIMID Open Calibration Gate

This append-only package freezes a revised SIMID open-response grading rubric
and exports a fresh blind sample for future calibration. It is a measurement
gate for future claimability, not evidence that the existing SIMID MVP improves
truthfulness.

Reviewer-facing files:

- `rubric.md`
- `review_cases_blind.jsonl`
- `label_schema.json`
- `prompt.md`

Private analysis files:

- `private_case_map.jsonl`
- `review_manifest.json`

Analyze returned labels with:

```bash
uv run python scripts/analyze_simid_prospective_open_calibration_gate.py \
  --package-dir <this-directory> \
  --labels <returned-labels.jsonl> \
  --output <append-only-analysis.json>
```
