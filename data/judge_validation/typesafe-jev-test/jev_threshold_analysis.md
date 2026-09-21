# Jev as jailbreak judge: threshold and ROC analysis

**Test set:** data/judge_validation/typesafe-jev-test/judge_validation_jev_20260921T193411Z.jsonl (N=74 scored)
**Model:** jev-1.13.0

## Agreement at the default threshold

| Alpha | GPT-4o | GLM-5 | Jev (t=0.5) |
|---|---|---|---|
| 0.0 | 20/25 | 21/25 | 20/25 (80.0%) |
| 1.5 | 22/25 | 23/25 | 23/25 (92.0%) |
| 3.0 | 21/24 | 22/24 | 22/24 (91.7%) |

## Discrimination (Jev only)

**AUC:** 0.980 (95% bootstrap CI 0.937 to 1.000)

AUC is the probability Jev scores a randomly chosen HARMFUL response above a randomly chosen SAFE one. Neither incumbent can be scored this way: a hard label is one point, not a curve.

## Threshold sweep

| Threshold | Accuracy | Missed harm | False alarm |
|---|---|---|---|
| 0.05 | 72/74 (97.3%) | 1 | 1 |
| 0.10 | 71/74 (95.9%) | 3 | 0 |
| 0.15 | 69/74 (93.2%) | 5 | 0 |
| 0.20 | 69/74 (93.2%) | 5 | 0 |
| 0.25 | 69/74 (93.2%) | 5 | 0 |
| 0.30 | 68/74 (91.9%) | 6 | 0 |
| 0.35 | 67/74 (90.5%) | 7 | 0 |
| 0.40 | 67/74 (90.5%) | 7 | 0 |
| 0.45 | 66/74 (89.2%) | 8 | 0 |
| 0.50 | 65/74 (87.8%) | 9 | 0 |
| 0.55 | 65/74 (87.8%) | 9 | 0 |
| 0.60 | 65/74 (87.8%) | 9 | 0 |
| 0.65 | 65/74 (87.8%) | 9 | 0 |
| 0.70 | 65/74 (87.8%) | 9 | 0 |
| 0.75 | 65/74 (87.8%) | 9 | 0 |
| 0.80 | 64/74 (86.5%) | 10 | 0 |
| 0.85 | 61/74 (82.4%) | 13 | 0 |
| 0.90 | 56/74 (75.7%) | 18 | 0 |
| 0.95 | 51/74 (68.9%) | 23 | 0 |

## Head-to-head

| Comparison | Discordant | p (exact McNemar) |
|---|---|---|
| Jev vs gpt4o | 6 (4/2) | 0.688 |
| Jev vs glm5 | 1 (0/1) | 1.000 |

## Power

- Jev vs gpt4o: 6 discordant pairs at N=74. Smallest detectable difference is 8.1 percentage points.
- Jev vs glm5: 1 discordant pairs at N=74. No split of that many pairs reaches p<0.05, so this comparison could not have detected any difference.

Under the repository's diagnosticity rule these head-to-head verdicts cap at pilot-suggestive. The design information they produce, the discordance rate and the powered N, is the first-class result.

## Limitations

- **No few-shot parity.** The GPT-4o judge receives six worked examples. Jev has no message history, so the rubric is carried in `criteria` only. Criteria 3 and 5 encode what examples 5 and 6 demonstrate, but the harnesses are not identical.
- **No rationale.** Jev returns a probability and no text, so disagreements cannot be read back the way a GPT-4o `reason` field can. Inspecting a Jev error means re-reading the response by hand.
- **Threshold fitted and evaluated on the same 74 records.** Any accuracy quoted at a swept threshold is optimistic. A held-out split is needed before quoting an operating point as production-ready.
