# Jev threshold: out-of-sample re-fit

**Records:** 74 scored, in 22 question groups across 25 record ids  
**Primary rule (pre-registered):** maximise MCC on the training fold  
**Primary estimator:** leave-one-group-out over question text

## Headline

| Threshold policy | Accuracy | 95% CI (clustered) | MCC | Missed harm | False alarm |
|---|---|---|---|---|---|
| Fixed default 0.5 | 65/74 (87.8%) | 79.2% to 95.3% | 0.781 | 9 | 0 |
| Fitted, out of sample | 70/74 (94.6%) | 89.4% to 98.7% | 0.889 | 3 | 1 |
| Fitted, in sample | 72/74 (97.3%) | not applicable | 0.943 | 1 | 1 |

Optimism gap: 97.3% in sample against 94.6% out of sample, a difference of 2.7 points.

## Where the fitted threshold lands

In-sample fit: 0.040, tied across the band 0.040 to 0.040.

Across the 22 folds the fitted cut runs from 0.040 to 0.090, median 0.040. 0 folds land inside the empty band 0.48 to 0.76, where the cut makes no difference to any verdict, and 22 land below it.

Jev's probabilities on this set take 26 distinct values, none between 0.48 and 0.76. Every cut inside that band is the same classifier, so a fold that fits there has effectively chosen the default.

## Spread under repeated grouped 5-fold

200 repeats, mean 95.0%, 5th to 95th percentile 94.6% to 95.9%, full range 93.2% to 95.9%. Reported as spread, not as the headline.

## Negative control

Labels permuted between same-sized question groups, 200 repeats through the identical pipeline: mean accuracy 50.5%, worst case 75.7%, against a majority-class rate of 60.8%. The real out-of-sample 94.6% beats the permuted runs at p = 0.0050. The cross-validation is not manufacturing accuracy from noise.

## Calibration

**Brier score: 0.0848** against 0.2383 for always predicting the base rate. Binned expected calibration error 0.1231.

| Bin | n | Mean probability | Observed harmful rate |
|---|---|---|---|
| 0.00 to 0.05 | 29 | 0.018 | 0.034 |
| 0.05 to 0.10 | 3 | 0.053 | 0.667 |
| 0.10 to 0.25 | 2 | 0.115 | 1.000 |
| 0.25 to 0.50 | 4 | 0.372 | 1.000 |
| 0.50 to 0.75 | 0 | | |
| 0.75 to 0.90 | 9 | 0.837 | 1.000 |
| 0.90 to 0.95 | 5 | 0.924 | 1.000 |
| 0.95 to 1.00 | 22 | 0.970 | 1.000 |

Read the counts before the rates. Most bins hold a handful of records, so this table shows the shape of the score distribution, not a calibration estimate fit to quote.

## Discrimination

AUC 0.980, 95% clustered bootstrap 0.934 to 1.000. Resampling question groups rather than rows, which is wider than the row-level interval in `jev_judge_evaluation.md` and is the correct one for this design.

## Head-to-head, with and without the independence assumption

| Comparison | Discordant | Exact McNemar p | Clustered permutation p |
|---|---|---|---|
| Jev vs gpt4o | 6 (4/2) | 0.688 | 0.752 |
| Jev vs glm5 | 1 (0/1) | 1.000 | 1.000 |

The exact test treats each discordant record as independent. The permutation flips a whole question group together. Neither reaches significance, so the independence assumption was not carrying the conclusion.

## Sensitivity to the fitting rule

These are not the headline. The primary rule was fixed before the script ran; these show whether the answer depends on that choice.

| Rule | In-sample cut | Out-of-sample accuracy | MCC | Missed harm | False alarm |
|---|---|---|---|---|---|
| mcc (primary) | 0.040 | 70/74 (94.6%) | 0.889 | 3 | 1 |
| balanced_accuracy | 0.040 | 70/74 (94.6%) | 0.889 | 3 | 1 |
| accuracy | 0.040 | 70/74 (94.6%) | 0.889 | 3 | 1 |
| zero_false_alarm | 0.085 | 70/74 (94.6%) | 0.889 | 3 | 1 |

## What this settles

It settles whether the in-sample 97.3% survives honest re-fitting. It does not establish a production operating point: every fold draws from the same 74 records and the same 22 questions, so the estimate corrects optimism without being independent evidence. A cut chosen here still needs confirming on records this set has never seen.

