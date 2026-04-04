# E2-B TriviaQA Family-Default Selectors (Diagnostic) -- 2026-04-04

> **Status: data authority for E2-B (family-default selectors).**
>
> E2-A (val_accuracy + last_answer_token) was near-inert. E2-B tests whether AUROC + all_answer_positions unlocks a signal in the same TriviaQA data.
>
> **Analytical synthesis:** [2026-04-04-e2-triviaqa-transfer-synthesis.md](./2026-04-04-e2-triviaqa-transfer-synthesis.md) interprets E2-A and E2-B together.

## 1. Extraction Metadata Diff (E2-B vs E2-A)

| Property | E2-A | E2-B | Changed? |
|---|---|---|---|
| head_ranking_metric | `val_accuracy` | `auroc` | YES |
| position_policy | `last_answer_token` | `all_answer_positions` | YES |
| position_summaries | `['last_answer_token']` | `['first_answer_token', 'mean_answer_span', 'last_answer_token']` | YES |
| selector_overrides | `{'position_policy_override': 'last_answer_token', 'ranking_metric_override': 'val_accuracy'}` | `{'position_policy_override': None, 'ranking_metric_override': None}` | YES |

## 2. Probe Quality Under Family-Default Selectors

| Metric | E2-B | E2-A | E0 (paper) |
|---|---|---|---|
| mean_auroc | 0.7536 | 0.7334 | 0.8198 |
| max_auroc | 0.7685 | 0.7610 | 0.8364 |
| min_auroc | 0.7469 | 0.7112 | 0.7992 |
| mean_val_accuracy | 0.6907 | 0.6794 | 0.7461 |

**E2-B top-5 heads:**

| Rank | Layer | Head | Position | AUROC |
|---|---|---|---|---|
| 1 | 11 | 2 | mean_answer_span | 0.7685 |
| 2 | 11 | 3 | mean_answer_span | 0.7677 |
| 3 | 11 | 0 | last_answer_token | 0.7513 |
| 4 | 9 | 3 | last_answer_token | 0.7500 |
| 5 | 22 | 4 | mean_answer_span | 0.7493 |

**March 30 artifact sanity check:** PASS (5/5 positional matches)

## 3. Calibration Sweep Summary

- Best MC1: 0.283951
- Locked config: K=8, alpha=6.0
- Shortlist (4 candidates):
  - K=8, alpha=6.0, MC1=0.2840, MC2=0.4190
  - K=24, alpha=6.0, MC1=0.2716, MC2=0.4190
  - K=8, alpha=8.0, MC1=0.2716, MC2=0.4145
  - K=8, alpha=12.0, MC1=0.2716, MC2=0.4046

## 4. Held-Out TruthfulQA (Pooled n=655)

### 4.1 Within E2-B (optimized lock vs alpha=0)
- MC1: +0.00 pp (95% CI [-1.98, +1.98])
- MC2: +1.46 pp (95% CI [-0.11, +3.02])

### 4.2 E2-B Optimized vs E2-A (KEY DIAGNOSTIC)
- MC1: +0.92 pp (95% CI [-1.37, +3.21])
- MC2: +0.73 pp (95% CI [-1.01, +2.46])

### 4.3 Sidecar A: E2-B @ K=40 alpha=6.0 vs E2-A (artifact-only isolation)
- MC1: -0.92 pp (95% CI [-2.75, +0.92])
- MC2: -1.10 pp (95% CI [-2.47, +0.25])

### 4.4 Sidecar B: E2-B @ K=16 alpha=6.0 vs E2-B @ K=40 alpha=6.0 (compact vs broad)
- MC1: +0.76 pp (95% CI [-1.22, +2.75])
- MC2: +0.95 pp (95% CI [-0.45, +2.36])

### 4.5 E2-B @ K=16 alpha=6.0 vs E2-A (combined effect)
- MC1: -0.15 pp (95% CI [-2.29, +1.98])
- MC2: -0.14 pp (95% CI [-1.84, +1.58])

### 4.6 E2-B Optimized vs E0 (paper-faithful TruthfulQA)
- MC1: -6.26 pp (95% CI [-9.16, -3.51])
- MC2: -6.04 pp (95% CI [-8.39, -3.70])

### 4.7 E2-B Optimized vs E1 (modernized TruthfulQA)
- MC1: -2.75 pp (95% CI [-5.19, -0.31])
- MC2: -3.03 pp (95% CI [-5.12, -0.95])

## 5. SimpleQA-200 (first_3_tokens)

### 5.1 Within E2-B
- Compliance: +0.00 pp (95% CI [-3.00, +3.00])
- Attempt: -0.50 pp (95% CI [-2.50, +1.00])
- Precision: +0.03 pp (95% CI [-3.03, +3.10])

### 5.2 E2-B vs E2-A
- Compliance: +0.50 pp (95% CI [-2.00, +3.00])
- Attempt: -0.50 pp (95% CI [-2.50, +1.00])
- Precision: +0.53 pp (95% CI [-2.04, +3.12])

### 5.3 E2-B vs E0 (paper-faithful)
- Compliance: +1.50 pp (95% CI [-1.50, +5.00])
- Attempt: +9.00 pp (95% CI [+5.00, +13.50])

### 5.4 E2-B vs E1 (modernized)
- Compliance: -0.50 pp (95% CI [-4.00, +3.00])
- Attempt: +1.00 pp (95% CI [-1.50, +3.50])

## 6. Head Overlap Diagnostics

### E2-B vs E2-A (same data, different selectors)
- Jaccard: 0.043 (intersection=2, union=46)
- Spearman rho (full rank): 0.914 (n=114, p=0.0000)
- Direction cosines (shared selected): mean=1.000, median=1.000, range=[1.000, 1.000]

### E2-B vs E0 (different data, different selectors)
- Jaccard: 0.053 (intersection=1, union=19)
- Spearman rho (full rank): 0.527 (n=114, p=0.0000)
- Direction cosines (shared selected): mean=-0.351, median=-0.351, range=[-0.351, -0.351]

### E2-B vs E1 (different data, different selectors)
- Jaccard: 0.000 (intersection=0, union=16)
- Spearman rho (full rank): 0.366 (n=145, p=0.0000)

## 7. Diagnostic Classification

**Classification: `wrong_source_still_likely`**

Both selector policies produce null results. The issue is likely TriviaQA source quality or model-level direction weakness, not selector choice.

## 8. Decision

Both selector policies failed. Accept the three-variant evidence: TriviaQA transfer under mass-mean ITI does not engage TruthfulQA-relevant representations in this model. Shift priority to D5 (externality audit) or D7 (causal head selection).

