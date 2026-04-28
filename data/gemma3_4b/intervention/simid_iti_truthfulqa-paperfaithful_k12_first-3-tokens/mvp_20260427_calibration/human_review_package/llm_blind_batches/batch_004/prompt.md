# SIMID Open Correctness Opus Batch batch_004

You are one independent blinded rater for this batch. Apply the frozen SIMID
open-correctness rule exactly. Do not use SIMID reports, effect estimates,
primary/secondary labels, deterministic alias labels, sample-source metadata,
adjudication notes, web browsing, package-root files, or any outside factual
criteria.

Use only these files in this directory:

- Frozen rule: `adjudication_rule.md`
- Batch blind cases: `review_cases_blind.jsonl`
- Label schema: `label_schema.json`
- Batch case file SHA-256: `04bc08c51f297f5fc4bc95b7c869c35997c4556af8cf2e3a8d6b9d179c8fb772`

Task:

Grade only review orders 31-40
from `review_cases_blind.jsonl`. Use only `question`, `gold_aliases`, and
`predicted_answer` plus the frozen rule. The gold aliases are authoritative.
If a case is genuinely hard, still choose the best label and lower confidence.
Set `rule_gap: true` only for malformed cases or a real unresolved rubric
conflict; ordinary borderline judgments are not rule gaps.

Write newline-delimited JSON to `opus_4_7_labels.jsonl` in this directory.

Do not print generated JSONL rows in chat. A valid row has:

- `schema_version`: `simid_open_llm_blind_rater_label/v1`
- `blind_case_id`: copied from the input case
- `review_order`: copied from the input case
- `label`: one of `CORRECT`, `INCORRECT`, `NOT_ATTEMPTED`
- `confidence`: integer 1-5
- `rule_gap`: boolean
- `flags`: zero or more of `bridge_partial_entity_or_modifier, truthfulqa_non_answer_boundary, truthfulqa_qualified_answer_boundary, wrong_extra_answer, multiple_candidates_no_commitment, alias_too_broad_or_too_narrow, malformed_case, other_boundary`
- `notes`: short explanation citing the decisive rule
- `rater.type`: `llm`
- `rater.model`: `claude-opus-4.7`
- `rater.prompt_version`: `simid_open_independent_rater/v1`
- `blind_cases_file_sha256`: `04bc08c51f297f5fc4bc95b7c869c35997c4556af8cf2e3a8d6b9d179c8fb772`

Quality checks before reporting done:

- Exactly 10 JSONL rows.
- No duplicate `blind_case_id`.
- Every `blind_case_id` appears in `review_cases_blind.jsonl`.
- No row contains original calibration identifiers.
- Every confidence is an integer from 1 to 5.
- Every row has `blind_cases_file_sha256` equal to
  `04bc08c51f297f5fc4bc95b7c869c35997c4556af8cf2e3a8d6b9d179c8fb772`.

Final chat response: one short sentence saying whether these checks passed and
which file was written. Do not include label JSONL in the chat response.
