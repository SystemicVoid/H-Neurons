# Quantitative Reporting Standards

Use this file when adding, changing, auditing, or publishing quantitative claims in the site, notes, reports, manuscripts, or claim-bearing data exports.

## Core Rule

Use evidence that matches the claim. A public number is not automatically well-supported because it has a 95% CI, and a CI is not automatically the right object for every claim.

Before reporting a quantitative claim, identify:

- the estimand or decision the number is meant to support;
- the evaluation unit: prompt, item, response, seed, model, evaluator, run, or neuron;
- the dependence structure: paired items, repeated generations, shared judges, clustered prompts, shared random seeds, or sweep reuse;
- the primary metric and why it is appropriate for the task;
- the comparison or control condition;
- the evidence type needed: interval, hypothesis test, calibration analysis, stability analysis, ablation, sensitivity analysis, or exact provenance.

If those are unclear, downgrade the claim to exploratory/descriptive until the analysis is better specified.

## Metric Choice

Prefer metrics that answer the scientific question rather than metrics that are easiest to compute.

- Balanced binary classification: accuracy can be fine when class balance and error costs make it meaningful.
- Imbalanced thresholded classification: prefer precision, recall, F1, balanced accuracy, or MCC, and report the thresholding rule.
- Ranking or detector quality: prefer AUROC for rank separation; prefer average precision when positives are rare or top-ranked recovery matters.
- Probabilistic prediction: use log loss, Brier score, calibration curves, or ECE. Accuracy alone does not support calibration claims.
- Safety or refusal outcomes: choose the metric that maps to the claim, such as harmful compliance, substantive compliance, refusal rate, task utility, or adjudicated correctness. Avoid changing the primary outcome after seeing the sweep.
- Continuous scores or margins: report a meaningful location/effect summary with scale and direction defined. Consider robust summaries if tails dominate.
- Curves and sweeps: predefine curve-level summaries such as slope, area under curve, max delta, or selected-alpha effect. Do not cherry-pick a single point after inspection.
- Feature, neuron, or model-selection claims: downstream score intervals are insufficient by themselves. Add selection stability, rank stability, held-out replication, ablations, or sensitivity checks as appropriate.
- Descriptive constants, exact counts, configured hyperparameters, hashes, paths, and provenance facts do not need uncertainty, but must not be framed inferentially.

## Uncertainty And Tests

Match uncertainty to the sampling design.

- Independent proportions: binomial or bootstrap intervals are acceptable.
- Paired binary outcomes: prefer paired bootstrap, McNemar tests, or permutation tests over independent-binomial intervals.
- Paired continuous or ordinal outcomes: prefer paired bootstrap, signed-rank/permutation tests, or justified model-based intervals.
- Clustered data: resample at the cluster level when prompts, users, datasets, judges, seeds, or runs induce dependence.
- Multi-seed or multi-run results: distinguish within-item uncertainty from between-seed/run stability. Do not hide seed instability inside a per-example CI.
- Evaluator-judged outputs: report judge identity/version and include agreement, calibration, sentinel, or audit evidence when judge behavior is itself part of the claim.
- Model comparison: prefer paired comparisons on the same items when possible. Report the paired estimand directly instead of comparing two unrelated CIs by eye.
- Multiple comparisons or broad sweeps: state what was pre-specified. Treat post-hoc discoveries as exploratory unless replicated or adjusted.

Use 95% intervals by default for public-facing claims, but the method name matters. Store enough metadata for a reader to understand how the interval or test was computed.

## Selection And Leakage

Keep model-selection evidence separate from final evaluation evidence.

- Choose primary metrics, thresholds, alpha grids, and stopping rules before inspecting final-test outcomes whenever practical.
- Tune thresholds, hyperparameters, judge prompts, feature sets, and neuron selections on train/dev data, then report locked final-test performance.
- If the reported result is the best of many runs, seeds, neurons, alphas, judge prompts, or metrics, say so and add held-out replication, correction, or stability evidence.
- Do not use the public/site/manuscript number as a debugging oracle. If final-test feedback shaped the method, downgrade the claim or rerun on a fresh holdout.
- For feature or neuron discovery, distinguish "selected because it performed well" from "validated after selection." The latter needs a separate evaluation path.

## Claim Strength

Route claims by evidence strength:

- Descriptive: exact values, counts, configuration, or observed sample summaries. No population inference implied.
- Exploratory: useful signal found after inspection, weak controls, missing uncertainty, or not yet replicated.
- Supported: metric, split, comparison, and uncertainty match the claim; provenance is registered.
- Causal or mechanism-level: requires controls, ablations, specificity checks, and failure-mode analysis. A performance delta alone is not enough.

Do not use broad language such as "improves", "causes", "robust", "specific", or "best" unless the design actually supports it.

## Manifest And Audit

`docs/ci_manifest.json` is the claim-provenance registry for public/reporting surfaces. It should track the source artifact, JSON path, claim kind, and surfaces that display the claim.

Extend the manifest or `scripts/audit_ci_coverage.py` when a claim needs a new evidence type. Do not force all claims into `estimate_with_ci` when the right evidence is a paired test, calibration artifact, stability table, selection audit, ablation, or descriptive provenance value.

Before finishing a change that touches claim-bearing reporting surfaces, run:

```bash
uv run python scripts/audit_ci_coverage.py
```

For site-facing quantitative claims, also follow `site/AGENTS.md`.

## Reporting Checklist

Before publishing or committing a claim-bearing surface, confirm:

- the metric answers the claim;
- the uncertainty/test matches the design;
- paired or clustered structure was handled correctly;
- the comparison/control is explicit;
- post-hoc choices are labelled exploratory or replicated;
- exact descriptive values are not framed as inference;
- the source artifact and surface are registered in `docs/ci_manifest.json`;
- stale or superseded numbers were removed from public surfaces.
