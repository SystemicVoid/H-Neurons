# GLM-5 as Jailbreak Judge: Evaluation Against GPT-4o

**Date:** 2026-03-26
**Test set:** `tests/gold_labels/jailbreak_cross_alpha_gold.jsonl` (75 records, 25 prompts x 3 alphas)
**Results:** `results/judge_validation/deepinfra-glm5-test/`

## TL;DR

GLM-5 (DeepInfra, fp4) matches or beats GPT-4o on binary jailbreak judging
accuracy (89.2% vs 85.1% agreement with human labels) and costs ~40% less
without batching. However, reasoning tokens inflate completion costs, there's
no way to disable them on DeepInfra, and throughput is ~4 req/min due to rate
limits and per-request latency (~10-17s).

## Accuracy (vs human gold labels)

| Alpha | GPT-4o | GLM-5 |
|-------|--------|-------|
| 0.0   | 20/25 (80.0%) | 21/25 (84.0%) |
| 1.5   | 22/25 (88.0%) | 23/25 (92.0%) |
| 3.0   | 21/24 (87.5%) | 22/24 (91.7%) |
| **All** | **63/74 (85.1%)** | **66/74 (89.2%)** |

Inter-judge agreement: 69/74 (93.2%).

## Error direction

| Judge  | False SAFE (missed harm) | False HARMFUL (false alarm) |
|--------|-------------------------|-----------------------------|
| GPT-4o | 10                      | 1                           |
| GLM-5  | 8                       | 0                           |

Both judges lean permissive (more false-SAFE than false-HARMFUL). GLM-5 has
fewer errors in both directions. The 3 cases both judges miss
(jbb_harmful_14_t3, jbb_harmful_36_t4, jbb_harmful_48_t3) are heavy
disclaimer-then-comply patterns -- likely hard for any automated judge.

## Cost comparison (1500-request production run)

Estimated token volumes per request: ~3,000 input, ~400 output (GPT-4o) or
~500 output (GLM-5, includes ~350 reasoning + ~150 content).

| Model   | Mode      | In $/M | Out $/M | Cache $/M | Est. Total | vs Baseline |
|---------|-----------|--------|---------|-----------|------------|-------------|
| GPT-4o  | Batch     | $1.25  | $5.00   | --        | **~$8.63** | baseline    |
| GPT-4o  | Real-time | $2.50  | $10.00  | --        | ~$17.25    | +100%       |
| GLM-5   | Real-time | $0.80  | $2.56   | $0.16     | **~$5.52** | **-36%**    |
| GLM-5   | + 50% cache | $0.48 avg | $2.56 | $0.16  | **~$4.07** | **-53%**    |

GLM-5 completion tokens include reasoning overhead (~2.5x more tokens than
GPT-4o per request), but at $2.56/M vs $5.00/M (batch) the per-token cost
advantage more than compensates.

## Reasoning token problem

GLM-5 is a reasoning model. Every request burns ~350 reasoning tokens before
producing ~150 content tokens. DeepInfra bills both as completion tokens.
There is no `thinking_budget` or reasoning-disable parameter on their API.
This means:

- **Output cost is ~2.5x what you'd expect** from content length alone
- **Latency is ~10-17s/request** (reasoning chain before answer)
- Even so, the low per-token price still makes it cheaper than GPT-4o batch

A non-reasoning variant (GLM-4.6 exists on DeepInfra) might be cheaper still
but would need a separate accuracy evaluation.

## Throughput

| Model   | Mode     | Throughput       | 1500 requests |
|---------|----------|------------------|---------------|
| GPT-4o  | Batch    | Fire-and-forget  | ~1-2 hours    |
| GLM-5   | Serial   | ~4 req/min       | ~6 hours      |
| GLM-5   | Async (est.) | ~10-15 req/min | ~2 hours    |

DeepInfra returns 429 ("Model busy") under even modest concurrency for GLM-5.
A 3s inter-request delay was needed for reliable serial execution. Async with
limited concurrency (~3-5 workers) might improve this but wasn't tested.

## Verdict

GLM-5 is a viable GPT-4o replacement for binary jailbreak judging:
- **Better accuracy** on the gold set (89.2% vs 85.1%)
- **Cheaper** even with reasoning overhead (~$5.52 vs $8.63)
- **Worse throughput** (~6h serial vs ~1-2h batch)

The throughput gap is the main blocker for production use. Options to address:
1. Test async concurrency with backoff to find the throughput ceiling
2. Evaluate GLM-4.6 (non-reasoning) -- same architecture, no reasoning tax
3. Accept the latency if running overnight / non-time-critical
