# Mistral Anchor-3 CSV-v3/StrongREJECT Disagreement Audit Package

Status: exported for blinded human review. No human labels are included in this
package.

## Counts

- Total disagreement rows: 164
- CSV-v3 safe / StrongREJECT harmful: 139
- CSV-v3 harmful / StrongREJECT safe: 25

| Alpha | CSV-v3 safe / StrongREJECT harmful | CSV-v3 harmful / StrongREJECT safe |
| --- | --- | --- |
| 0.0 | 42 | 1 |
| 1.0 | 29 | 5 |
| 1.5 | 24 | 11 |
| 3.0 | 44 | 8 |

| CSV-v3 primary outcome | Main stratum | Reverse stratum |
| --- | --- | --- |
| benign_answer | 114 | 0 |
| deflection | 18 | 0 |
| partial_compliance | 0 | 21 |
| refusal | 7 | 0 |
| substantive_compliance | 0 | 4 |

## Prioritization

Review order is deterministic and prioritizes:

1. CSV-v3 safe / StrongREJECT harmful rows before reverse disagreements.
2. CSV-v3 `benign_answer` rows before other CSV-v3-safe outcomes.
3. Higher StrongREJECT score, then convincingness, specificity, alpha, and
   source ID for stable tie-breaking.

High-priority subset:

- Main-stratum `benign_answer` rows: 114
- Main-stratum `benign_answer` rows with StrongREJECT score >= 0.75: 103
- Main-stratum `benign_answer` rows with StrongREJECT score = 1.0: 66

## Blinding

- `review_cases_blind.jsonl` and `index.html` include only the reconstructed
  prompt, goal, response, source ID, alpha, category, template index, and stable
  review ID. They do not include CSV-v3 labels, StrongREJECT labels, scores, or
  rationales.
- `private_analysis_key.jsonl` contains the machine judge metadata and stratum
  labels for post-label reconciliation. Do not provide it to a blinded rater.

## Files

- Package directory: `data/judge_validation/mistral_anchor3_csv3_strongreject_disagreement_audit`
- `index.html`: static local review UI styled after `/site`, with keyboard
  shortcuts for label, confidence, rule-gap toggle, save, and navigation.
- `review_cases_blind.jsonl`: 164 blind review cases.
- `private_analysis_key.jsonl`: private join key and machine judge metadata.
- `adjudication_rule.md`: frozen independent harmfulness rule.
- `label_schema.json`: JSON schema for exported human labels.
- `review_manifest.json`: package hashes, source hashes, counts, and
  verification.

File hashes:

- Blind cases SHA-256: `e3502f2764b8e78843d22a19b7bedc41424997707c472d200fdab34078d334bc`
- Private key SHA-256: `45a2716969710c581bb23e2ec44802737d9c9794df11638d691bc6a47f3d355d`

## Verification

The exporter validated row counts against
`mistral_anchor3_jailbreak_evaluator_summary.json`, matched `(id, alpha)` keys
across generation, CSV-v3, and StrongREJECT files for all four alpha files, and
confirmed response text identity across the three sources before extraction.

| Alpha | Experiment | CSV-v3 | StrongREJECT | Matched keys | Matched responses |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 500 | 500 | 500 | 500 | 500 |
| 1.0 | 500 | 500 | 500 | 500 | 500 |
| 1.5 | 500 | 500 | 500 | 500 | 500 |
| 3.0 | 500 | 500 | 500 | 500 | 500 |
