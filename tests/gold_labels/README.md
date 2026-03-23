# Evaluator Gold-Label Audit Sets

Hand-verified reference cases for evaluator regression detection. Each
benchmark has a JSONL fixture with human-confirmed ground-truth labels.

## Fixtures

### `faitheval_standard.jsonl` — deterministic parser + text remap

Tests `extract_mc_answer` and `strict_remap_label`. No API calls required.

Schema:
```json
{
  "id": "sample_id",
  "response": "model response text",
  "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "counterfactual_key": "D",
  "expected_mc_extract": null,
  "expected_remap_label": "D",
  "expected_remap_method": "exact_normalized_text",
  "expected_compliance": true,
  "category": "strict_remap | manual_review",
  "notes": null
}
```

### `falseqa_candidates.jsonl` / `jailbreak_candidates.jsonl` — LLM judge

Cases selected for human review of GPT-4o judge verdicts. Records with
`human_label: null` have not yet been verified and are skipped by the
validation script.

Schema:
```json
{
  "id": "sample_id",
  "question": "prompt text",
  "response": "model response text",
  "human_label": "ACCEPTED | REFUSED | null",
  "judge_label_at_creation": "ACCEPTED",
  "source_alpha": 0.0,
  "notes": "Why this case was selected"
}
```

## Usage

Deterministic tests (FaithEval):
```
uv run pytest tests/test_evaluator_gold.py
```

LLM judge audit (FalseQA / jailbreak):
```
uv run python scripts/validate_evaluator_gold.py --benchmark falseqa
uv run python scripts/validate_evaluator_gold.py --benchmark jailbreak
```
