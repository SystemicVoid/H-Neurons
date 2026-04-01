
## 2026-04-01T10:42:00+00:00 | OpenAI Batch API tier-3 upgrade

What: account upgraded from Tier 2 → Tier 3; update local batch-queue-limit table in `scripts/openai_batch.py` to reflect new ceilings.
Key files: `scripts/openai_batch.py`, `scripts/evaluate_intervention.py`, `scripts/evaluate_csv2.py`, `tests/test_openai_batch.py`
Status: done: TIER3_BATCH_QUEUE_LIMITS table updated, all tier-string refs patched, regression tests updated

Verified Tier-3 queue limits checked on 2026-04-01 (live API headers confirmed tier via gpt-4o headers: 5,000 RPM / 800,000 TPM, up from Tier-2's 450,000 TPM):

| Model      | Tier 2        | Tier 3          | Source URL                                          |
|------------|---------------|-----------------|-----------------------------------------------------|
| gpt-4o     | 1,350,000     | 50,000,000      | developers.openai.com/api/docs/models/gpt-4o       |
| gpt-4o-mini| 20,000,000    | 40,000,000      | developers.openai.com/api/docs/models/gpt-4o-mini  |
| gpt-4.1    | 1,350,000     | 50,000,000      | developers.openai.com/api/docs/models/gpt-4.1      |
| gpt-5      | 3,000,000     | 100,000,000     | developers.openai.com/api/docs/models/gpt-5        |
| gpt-5-mini | 20,000,000    | 40,000,000      | developers.openai.com/api/docs/models/gpt-5-mini   |
| o3         | 1,350,000     | 50,000,000      | developers.openai.com/api/docs/models/o3           |
| o4-mini    | 2,000,000     | 40,000,000      | developers.openai.com/api/docs/models/o4-mini      |

Changes made:
- `scripts/openai_batch.py`: renamed `TIER2_BATCH_QUEUE_LIMITS` → `TIER3_BATCH_QUEUE_LIMITS`, updated all 7 values, updated comment date and tier label, updated log string "Tier-2 queue" → "Tier-3 queue".
- `scripts/evaluate_intervention.py`, `scripts/evaluate_csv2.py`: updated `--batch-max-enqueued-tokens` help text to reference Tier-3.
- `tests/test_openai_batch.py`: renamed `test_known_model_uses_tier2_limit_with_default_margin` → `..._tier3_...`, updated `queue_limit` assertions (1,350,000 → 50,000,000); renamed `test_gpt4o_tier2_cap_reduces_old_17_chunk_case_to_two_chunks` → `..._tier3_..._to_one_chunk` (1,284,222 tokens now fits in 1 batch, not 2), updated `resolution.value` assertion.

Practical impact: the previously-17-chunk gpt-4o workload (~1.28M tokens) now submits as a single batch. gpt-4o-mini and gpt-5-mini see a more modest 2× increase (20M → 40M) since they were already near-unlimited at Tier 2.

## 2026-03-27T22:24:00+00:00 | OpenAI Batch API tier-2 follow-up
What: verify whether local batch chunking is still pinned to the old `80_000` token ceiling after the OpenAI Tier 2 upgrade, and record the exact code/doc refs needed to adjust it tomorrow after the live `eval` tmux job completes.
Key files: `scripts/openai_batch.py`, `scripts/evaluate_intervention.py`, `scripts/evaluate_csv2.py`, `logs/jailbreak_alpha1_binary_judge_20260327_204212.log`
Status: done: model-aware Tier-2 batch caps implemented with override hooks and regression tests

Findings:
- Local hard cap: `scripts/openai_batch.py` hard-codes `DEFAULT_MAX_ENQUEUED_TOKENS = 80_000`.
- Chunking path: `scripts/openai_batch.py` uses that cap in `resume_or_submit(..., max_enqueued_tokens=...)` and auto-chunks when `total_est > max_enqueued_tokens`.
- Inheritance path: `scripts/evaluate_intervention.py`, `scripts/evaluate_csv2.py`, `scripts/validate_evaluator_gold.py`, and `scripts/characterize_swing.py` call `resume_or_submit(...)` without overriding `max_enqueued_tokens`, so they all inherit the `80_000` default.
- Live-run evidence, binary judge: `logs/jailbreak_alpha1_binary_judge_20260327_204212.log` shows `Auto-chunking: 500 requests into 17 batches (~1284222 est. tokens, limit 80000)`.
- Live-run evidence, CSV-v2: log line showed `Auto-chunking: 500 requests into 20 batches (~1559221 est. tokens, limit 80000)`.
- Verified against current Tier 2 `gpt-4o` docs on 2026-03-28: the published batch queue limit is `1,350,000` tokens. That means `~1.284M` would fit only if we target the raw ceiling; with a safer `0.8x` to `0.95x` cap it still lands in `2` batches. `~1.559M` requires `2` batches even at the raw ceiling. The real regression is therefore `17/20` batches vs an expected `2/2`, not `17/20` vs `1/2`.

Verified OpenAI refs checked on 2026-03-28:
- Batch guide: <https://platform.openai.com/docs/guides/batch>
  Batch limits are separate from normal per-model rate limits; guide documents up to `50,000` requests per batch, `200 MB` input files, and a per-model enqueued prompt-token cap.
- GPT-4o: <https://developers.openai.com/api/docs/models/gpt-4o>
  Tier 2: `450,000 TPM`, `1,350,000` batch queue limit.
- GPT-4.1: <https://developers.openai.com/api/docs/models/gpt-4.1>
  Tier 2: `450,000 TPM`, `1,350,000` batch queue limit.
- GPT-5: <https://developers.openai.com/api/docs/models/gpt-5>
  Tier 2: `1,000,000 TPM`, `3,000,000` batch queue limit.
- GPT-5 mini: <https://developers.openai.com/api/docs/models/gpt-5-mini>
  Tier 2: `2,000,000 TPM`, `20,000,000` batch queue limit.
- o3: <https://developers.openai.com/api/docs/models/o3>
  Tier 2: `450,000 TPM`, `1,350,000` batch queue limit.
- o4-mini: <https://developers.openai.com/api/docs/models/o4-mini>
  Tier 2: `2,000 RPM`, `2,000,000 TPM`, `2,000,000` batch queue limit.

Tomorrow's follow-up:
- Replace the fixed `80_000` default in `scripts/openai_batch.py` with a model-aware default derived from the request model, while keeping an explicit `max_enqueued_tokens` override for callers that need to pin behavior.
- Add a small in-repo limit registry for the models we actually use, keyed by model alias, and keep the Tier-2 queue values local rather than scraping docs at runtime. Treat the docs as the update source, not the live dependency.
- Apply a configurable safety margin to the published queue limit instead of aiming at the ceiling exactly. First pass should be `0.9x` by default, with CLI or env override if we want to ratchet down to `0.85x` or `0.8x`.
- Thread the override through the batch entrypoints in `scripts/evaluate_intervention.py` and `scripts/evaluate_csv2.py` so a run can force a lower cap without editing code. The shared helper change should also cover `scripts/validate_evaluator_gold.py` and `scripts/characterize_swing.py`.
- Add regression tests in `tests/test_openai_batch.py` for model-aware cap resolution and for chunk-count behavior around the `gpt-4o` Tier-2 boundary.
- Validate with one cheap post-job run after the current eval completes: confirm the binary-judge request set now lands in `2` batches on `gpt-4o`, and that the CSV-v2 set also lands in `2` rather than `20`.
- Do not reuse one cap across all models; the current docs already diverge materially across `gpt-4o`, `gpt-5`, `gpt-5-mini`, `o3`, and `o4-mini`.

Implemented on 2026-03-28:
- `scripts/openai_batch.py` now resolves `max_enqueued_tokens` in this order: explicit argument, `OPENAI_BATCH_MAX_ENQUEUED_TOKENS`, model-aware Tier-2 default with `OPENAI_BATCH_QUEUE_SAFETY_MARGIN` / default `0.9`, then legacy fallback `80_000` for unknown models.
- Local Tier-2 queue registry covers `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-5`, `gpt-5-mini`, `o3`, and `o4-mini`, with snapshot aliases normalized onto those keys.
- `scripts/evaluate_intervention.py` and `scripts/evaluate_csv2.py` now expose `--batch-max-enqueued-tokens` / `--batch_max_enqueued_tokens`.
- Regression tests added in `tests/test_openai_batch.py`, including a guard that the old `~1,284,222`-token `gpt-4o` case resolves to `2` chunks under the new default.
- `scripts/infra/check_openai_batch_limits_via_codex.sh` now launches `codex exec --search` as a preflight verifier against official OpenAI docs and only patches the local hardcoded Tier-2 table if the docs changed.
- `scripts/infra/jailbreak_alpha1_pipeline.sh`, `scripts/infra/jailbreak_alpha1_eval_only.sh`, and `scripts/infra/csv2_pipeline.sh` now run that Codex preflight by default before queuing batch-eval work. Set `CODEX_VERIFY_OPENAI_LIMITS=0` to skip.
- Live preflight check on 2026-03-28 completed successfully and found no drift: the local Tier-2 table in `scripts/openai_batch.py` already matched the current official docs for `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-5`, `gpt-5-mini`, `o3`, and `o4-mini`.
