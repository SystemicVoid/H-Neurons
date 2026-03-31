# ITI K×α Calibration Sweep Analysis — Gate 1

**Date:** 2026-03-31
**Model:** Gemma-3-4B-IT
**Benchmark:** TruthfulQA (MC1 + MC2)
**Calibration set:** 81 cal-val samples (seed=42)
**Sweep grid:** K ∈ {8, 12, 16, 24, 32, 40} × α ∈ {0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0} → 54 combos

## MC1 Heatmap

```
         α→   0.0   0.5   1.0   2.0   4.0   6.0   8.0  12.0  16.0
K= 8        22.2  23.5  23.5  25.9  25.9  30.9  30.9  33.3  32.1
K=12        22.2  23.5  25.9  27.2  34.6  35.8  37.0 *38.3* 34.6
K=16        22.2  23.5  24.7  25.9  29.6  34.6  35.8  35.8  35.8
K=24        22.2  23.5  24.7  24.7  24.7  23.5  23.5  23.5  24.7
K=32        22.2  23.5  24.7  23.5  18.5  18.5  18.5  21.0  17.3
K=40        22.2  25.9  27.2  25.9  22.2  22.2  22.2  17.3  14.8
```

Baseline (α=0): MC1=22.22%, MC2=40.79%.

## Surface Structure

**Hot zone.** MC1 >30% is confined to K ∈ {8, 12, 16} × α ∈ {6, 8, 12}. The peak (38.27%, K=12 α=12) is a **lone spike** — only 1 combo within 1pp. This is a fragile optimum.

**K rollover.** At K≥24, MC1 collapses to baseline or below, even at the same α values that work for K=12-16. Why: ITI selects the top-K heads by truthfulness-direction magnitude. At K=24+, the marginal heads carry little signal but inject substantial noise, diluting the intervention. At K=32/40, high α actively *hurts* — pushing the model's representations along noisy directions degrades below the no-intervention baseline.

**K=16 plateau.** The K=16 row shows α=8/12/16 all yielding exactly 35.80% MC1 — a flat plateau where additional α no longer changes the discrete MC answer. This suggests the intervention saturates at K=16; it's shifting probability mass around but not flipping enough answers to move MC1.

**MC1–MC2 agreement.** Pearson r=0.937 across all 54 combos. Both metrics point to the same K={12,16} × α={6,8,12} region. MC2 peak is at K=16 α=12 (52.44%), slightly favoring more heads — MC2 rewards partial probability shifts that MC1 ignores.

## Confidence Interval Analysis

With n=81, binomial proportion 95% CIs are wide (~±10pp). Computed via Wilson interval:

| Config | MC1 | 95% CI | MC2 | 95% CI |
|--------|-----|--------|-----|--------|
| K=12, α=12 (auto) | 38.27% | [27.7, 48.9] | 50.59% | [39.7, 61.5] |
| K=12, α=8 (locked) | 37.04% | [26.5, 47.6] | 49.92% | [39.0, 60.8] |
| K=16, α=12 | 35.80% | [25.4, 46.2] | 52.44% | [41.6, 63.3] |
| K=16, α=8 | 35.80% | [25.4, 46.2] | 50.44% | [39.6, 61.3] |
| K=12, α=6 | 35.80% | [25.4, 46.2] | 49.19% | [38.3, 60.1] |
| Baseline (α=0) | 22.22% | [13.2, 31.3] | 40.79% | [30.1, 51.5] |

**Key finding:** The top 5 combos are all **statistically indistinguishable** from each other. The 1.23pp gap between K=12 α=12 and K=12 α=8 is noise on 81 samples (z=0.16, p=0.87). Even the best combo vs. baseline is only marginally significant (z=2.26, p=0.024) — the 2-fold held-out evaluation in Gate 2 (n≈327) will be the real test.

## Recommendation: Override to K=12, α=8

**Decision.** Override the auto-lock (K=12, α=12) to **K=12, α=8.0**.

**Rationale:**

1. **Statistical indistinguishability.** 1.23pp delta on n=81 is well within noise. The auto-lock's preference is an artifact of the selection rule applied to a single draw from a wide distribution.

2. **Lower α is safer downstream.** α scales the intervention vector added to hidden states. α=8 is 33% less perturbation than α=12. Since we evaluate on SimpleQA, FalseQA, and FaithEval next, minimizing side-effects is more valuable than chasing a phantom 1pp MC1 gain.

3. **Rising slope vs. spike.** K=12 MC1 climbs monotonically from α=4 (34.6%) → α=6 (35.8%) → α=8 (37.0%) → α=12 (38.3%) then drops at α=16 (34.6%). Sitting at α=8 (on the slope) rather than α=12 (at the peak-then-drop) gives margin against cal-set overfitting.

4. **K=12 over K=16.** K=12 consistently equals or exceeds K=16 in MC1 at matched α. Fewer heads = simpler intervention with less capacity for noise.

## What Was Locked

```json
{
  "K_locked": 12,
  "alpha_locked": 8.0,
  "calibration_mc1": 0.37037,
  "calibration_mc2": 0.499169,
  "artifact_fingerprint": "461f3ca7e0a21876",
  "human_override": {
    "K": 12,
    "alpha": 8.0,
    "reason": "Statistically indistinguishable from auto-lock; lower alpha reduces downstream risk"
  }
}
```

**Files written:**
- `data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_calibration/locked_iti_config.json`
- `data/contrastive/truthfulness/iti_truthfulqa_paperfaithful_calibration/pipeline_state.json`

## Gate 2 Readiness

`pipeline_state.json` exists with `gate_1_sweep.locked.K=12`, `gate_1_sweep.locked.alpha=8.0`. The Gate 2 script reads these values and will use them for fold extraction, evaluation, and controls.

```bash
./scripts/infra/iti_pipeline_evaluate.sh
```
