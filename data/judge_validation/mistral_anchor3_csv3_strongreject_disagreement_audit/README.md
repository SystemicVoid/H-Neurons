# Mistral Anchor-3 Disagreement Human Audit Package

This package supports blinded review of selected Mistral anchor-3 full-output
responses. Raters should judge only the prompt, goal, response, and
`adjudication_rule.md`; machine-judge metadata is reserved for post-label
analysis.

## Files

Rater-facing files:

- `index.html` - static local review UI.
- `review_cases_blind.jsonl` - blinded review surface with prompt, goal, and
  response text.
- `adjudication_rule.md` - independent harmfulness rule for human labels.
- `label_schema.json` - machine-readable schema for exported labels.

Analyst-only files:

- `private_analysis_key.jsonl` - private post-label reconciliation metadata.
  Do not provide this file to blinded raters until labeling is complete.
- `audit_summary.md` - concise non-raw package summary and verification.
- `review_manifest.json` - counts, hashes, source paths, and verification.
- `export_mistral_jailbreak_disagreement_audit_package.provenance.*.json` -
  append-only exporter provenance sidecars.

For blinded labeling, share only the rater-facing files.

## Selection

- Total blinded review cases: 164
- Blind cases file SHA-256: `e3502f2764b8e78843d22a19b7bedc41424997707c472d200fdab34078d334bc`

## How To Use

1. Open `data/judge_validation/mistral_anchor3_csv3_strongreject_disagreement_audit/index.html` locally in a browser.
2. Enter a rater ID.
3. Assign exactly one label for each response: `HARMFUL`, `SAFE`, or
   `AMBIGUOUS_OR_MALFORMED`.
4. Use confidence 1-5, optional flags, and notes where useful.
5. Export JSONL from the UI after labeling.
6. Treat case position and `review_order` as opaque queue identifiers.
7. Join exported labels to `private_analysis_key.jsonl` only after blinded
   labeling is complete.

This package does not promote CSV-v3 or StrongREJECT to ground truth. It exists
so later judge-validity claims can be grounded in reviewed disagreement
evidence.
