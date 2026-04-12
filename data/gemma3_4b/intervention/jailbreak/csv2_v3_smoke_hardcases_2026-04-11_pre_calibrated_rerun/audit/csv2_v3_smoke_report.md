# CSV v3 Hard-Case Smoke Test

## Automated Summary

- Cohort size: 30 rows
- Gold-backed rows: 24
- Directional gold agreement (HARMFUL->yes, SAFE->no, borderline skipped): 9/24
- Rows with refusal spans: 27
- Rows with harmful spans: 7
- Rows with pivot span: 0
- Rows with span errors: 1

## Attention Rows

- Legacy pivot without v3 pivot: 20 rows
- Gold HARMFUL but v3 not `yes`: 13 rows
- Gold SAFE but v3 `yes`: 2 rows

## Manual Audit Status

Pass/fail remains pending human review of the CSV workbook columns:
`expected_refusal`, `expected_pivot`, `spans_human_sane`, `csv_human_sane`, and `note`.

This report is the instrument panel, not the verdict. The actual trust gate is whether the span-localized outputs look sane on the hard cases once a human reads them.
