# SIMID Fresh Blind Boundary Adjudication Batch 001

Use only `review_cases_blind.jsonl`, `adjudication_rule.md`, and `label_schema.json` in this directory. Do not inspect prior primary, secondary, Opus, human, reference, or adjudication labels.

For each row, apply the frozen SIMID open correctness rule and write one JSONL label per case to `fresh_blind_boundary_labels.jsonl`. Use exactly one of `CORRECT`, `INCORRECT`, or `NOT_ATTEMPTED`. Set `rule_gap` only when the frozen rule cannot be applied because the case is malformed or contains a genuine unresolved rubric conflict.

Use this rater block:

```json
{"type": "llm", "model": "gpt-5-codex", "prompt_version": "simid_fresh_blind_boundary_adjudication_20260429_v1"}
```
