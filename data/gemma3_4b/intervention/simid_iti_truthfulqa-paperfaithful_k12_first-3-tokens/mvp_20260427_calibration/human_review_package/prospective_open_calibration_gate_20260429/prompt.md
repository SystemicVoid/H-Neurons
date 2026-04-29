# SIMID Prospective Open Calibration

Grade the 150 rows in `review_cases_blind.jsonl` using only `rubric.md`.
Do not inspect private files or historical labels.

Write newline-delimited JSON to `prospective_open_labels.jsonl`. Each row must
match `label_schema.json` and include:

- `schema_version`: `simid_prospective_open_calibration_label/v1`
- `blind_case_id`
- `review_order`
- `label`: one of `CORRECT`, `INCORRECT`, `NOT_ATTEMPTED`
- `confidence`: integer 1-5
- `rule_gap`: boolean
- `flags`: zero or more schema-valid flags
- `notes`: concise rationale citing the decisive rubric rule
- `rater`
- `blind_cases_file_sha256`
- `rubric_sha256`

Return labels for every blind case exactly once.
