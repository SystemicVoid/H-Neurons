# Measurement Summary

This file is a reviewer-facing derivative of the paper’s measurement evidence. It combines the seed-0 jailbreak specificity analysis with the contamination-clean evaluator holdout summary.

## Canonical Repo Origins Used For This Summary

- `notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md`
- `notes/act3-reports/2026-04-12-4way-evaluator-holdout-validation.md`
- `data/judge_validation/holdout_comparison.json`

## §5.1 Seed-0 Jailbreak Negative Control

| Claim | Value | CI / exact test |
|---|---|---|
| Binary harmful endpoint shift (`α = 0.0 → 3.0`) | 152 / 500 → 167 / 500 | delta +3.0 pp; CI includes zero |
| H-neuron graded harmfulness slope | +2.30 pp/α | [+0.99, +3.58] |
| H-neuron graded endpoint shift (`α = 0.0 → 3.0`) | +7.6 pp | [+2.6, +12.8] |
| Random-neuron graded slope (seed-0 control) | -0.47 pp/α | [-1.42, +0.47] |
| H-minus-random slope difference | +2.77 pp/α | [+1.17, +4.42] |
| H-minus-random permutation test | `p = 0.013` | one-sided |

Interpretation: the graded CSV2 surface detects a neuron-specific harmfulness trend that the binary endpoint does not resolve cleanly at this sample size.

## §5.2 Holdout Evaluator Validation

### Contamination boundary

| Item | Value |
|---|---|
| Bundled holdout JSON | [`../data/judge_validation/holdout_comparison.json`](../data/judge_validation/holdout_comparison.json) |
| Clean holdout size | 50 records |
| Bootstrap method | prompt-clustered bootstrap over 17 prompt IDs |
| Calibration contamination rule | exclude the 8 prompt IDs used in CSV2 v3 calibration |

### Holdout accuracies

| Evaluator | Accuracy | 95% CI |
|---|---|---|
| CSV2 v3 (GPT-4o) | 96.0% | [90.0, 100.0] |
| StrongREJECT (GPT-4o) | 96.0% | [90.0, 100.0] |
| CSV2 v2 (GPT-4o) | 92.0% | [84.3, 98.0] |
| Binary judge (GPT-4o) | 90.0% | [80.0, 98.0] |

### Paired comparison summary

| Claim | Value |
|---|---|
| CSV2 v3 vs StrongREJECT discordant records | 0 |
| CSV2 v3 vs StrongREJECT McNemar exact `p` | 1.0 |
| Pairwise holdout significance summary | no pairwise holdout difference is statistically confirmable |
| All six exact McNemar `p` values | `>= 0.25` |

Interpretation: CSV2 v3 and StrongREJECT-GPT-4o tie on the contamination-clean holdout. The paper uses CSV2 v3 as the structured measurement surface, not because the clean holdout establishes binary superiority over StrongREJECT.

## Reviewer Notes

- The holdout JSON is bundled because it is safe, compact, and machine-readable.
- The raw gold fixture and raw scored JSONLs are not bundled because they contain harmful prompts and full model responses.
