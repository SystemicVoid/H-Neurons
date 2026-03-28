
## 2026-03-27T22:24:00+00:00 | OpenAI Batch API tier-2 follow-up
What: verify whether local batch chunking is still pinned to the old `80_000` token ceiling after the OpenAI Tier 2 upgrade, and record the exact code/doc refs needed to adjust it tomorrow after the live `eval` tmux job completes.
Key files: `scripts/openai_batch.py`, `scripts/evaluate_intervention.py`, `scripts/evaluate_csv2.py`, `logs/jailbreak_alpha1_binary_judge_20260327_204212.log`
Status: awaiting analysis/action

Findings:
- Local hard cap: `scripts/openai_batch.py` hard-codes `DEFAULT_MAX_ENQUEUED_TOKENS = 80_000`.
- Chunking path: `scripts/openai_batch.py` uses that cap in `resume_or_submit(..., max_enqueued_tokens=...)` and auto-chunks when `total_est > max_enqueued_tokens`.
- Inheritance path: `scripts/evaluate_intervention.py` and `scripts/evaluate_csv2.py` call `resume_or_submit(...)` without overriding `max_enqueued_tokens`, so both inherit the `80_000` default.
- Live-run evidence, binary judge: log line showed `Auto-chunking: 500 requests into 17 batches (~1048861 est. tokens, limit 80000)`.
- Live-run evidence, CSV-v2: log line showed `Auto-chunking: 500 requests into 20 batches (~1559221 est. tokens, limit 80000)`.
- Inference from current Tier 2 `gpt-4o` docs: `~1.05M` estimated prompt tokens should fit in one batch queue, and `~1.56M` should require about two batches, not `17` and `20`.

Verified OpenAI refs checked on 2026-03-27:
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
- Replace the fixed `80_000` default with a model-aware cap, ideally configurable by CLI or env.
- Keep a safety margin below the published queue limit instead of aiming at the ceiling exactly; first pass should be about `0.8x` to `0.9x` of the published limit.
- Validate with one cheap dry run after the current eval completes: confirm a request set that previously split into many `~80k` chunks now lands in `1` to `2` batches on `gpt-4o`.
- Do not reuse one cap across all models; the pasted table was already partly stale when checked, especially for `GPT-5` and `o4-mini`.