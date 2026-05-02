# Bridge Question/Gold Token-Overlap Audit (heuristic)

Heuristic only. A row appears in the no-overlap list when the question and all gold aliases share zero lowercased content tokens of length >= 4 (stopwords removed). Many legitimate rows have zero token overlap (e.g., "capital of France" vs "Paris"), so the no-overlap set is a screening pool, not an exclusion list.

- Total Bridge rows scanned: **400**
- Rows with zero question/gold token overlap: **342** (85.5%)
- Canary's known-malformed `simid_bridge_odql_6602` present in pool: **True**

## Token-overlap-count histogram (questions vs gold aliases)

| shared_tokens | rows |
|---|---|
| 0 | 342 |
| 1 | 48 |
| 2 | 10 |

## Recommendation

Treat `bridge_question_gold_overlap_audit.jsonl` as a screening list to review with an LLM judge or human auditor before the full-batch labeling pass. Promote rows confirmed as malformed to `excluded_sample_ids_recommended_for_full_batch.jsonl` and bind that file by sha256 in a fresh effect-run-gate manifest. Do not modify the manifest-bound exclusions file at `prospective_effect_run_gate_20260429_r2_external_labels/excluded_effect_sample_ids.jsonl`.
