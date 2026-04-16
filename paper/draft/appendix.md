# Appendix

## Appendix A. Detector Interpretation Audits

This appendix collects the two detector-interpretation cautions referenced in §4.1.

**N4288 / L1 instability.** The top-weight CETT neuron at layer 20, neuron 4288, is not a stable "most important neuron" finding across regularization settings. In the pipeline audit, it is absent at $C <= 0.3$, appears at $C = 1.0$, and drops to rank 5 at $C = 3.0$ while a wider detector reaches 80.5% test accuracy with 219 positive neurons. Across six follow-up analyses, the verdict is that the extreme weight is better explained by L1 concentration among correlated features than by unique causal importance.^[Source: `data/gemma3_4b/pipeline/pipeline_report.md`, detector audit section; `scripts/investigate_neuron_4288.py` generated the underlying analyses.]

**Verbosity confound.** The verbosity audit shows that under full-response aggregation, response length dominates truthfulness signal by roughly $3.7\times$ (mean aggregation) to $16\times$ (max aggregation), and 36 of 38 H-neurons are more length-dominant than truth-dominant in that setting. The mitigation is scope: the classifier result is still a held-out answer-token discrimination result, but it should not be interpreted as a pure hallucination detector without this qualification.^[Source: `data/gemma3_4b/intervention/verbosity_confound/verbosity_confound_audit.md`.]

These appendix notes are not separate headline claims. Their role is to justify the narrower wording used in the main text: the readouts are real, but detector interpretation is fragile.

## Appendix B. Benchmark Power Summary

**Appendix Table B1 -- Benchmark Power and MDE Summary**

| Benchmark | $n$ | Primary Metric | Observed H-neuron Effect (no-op to max) | Slope | MDE (paired, 80% power) | Status |
|---|---:|---|---|---|---|---|
| FaithEval | 1,000 | Compliance rate | +4.5 pp [2.9, 6.1] | +2.09 pp/$\alpha$ [1.38, 2.83] | ${\sim}3$ pp | Well-powered |
| FalseQA | 687 | Compliance rate | +2.5 pp [$-0.6$, 5.5] | +1.62 pp/$\alpha$ [0.52, 2.74] | ${\sim}4$ pp | Slope significant; endpoint borderline |
| JailbreakBench | 500 | Strict harmfulness rate | +7.6 pp [2.6, 12.8] ($\alpha = 0 \rightarrow 3$ full sweep) | +2.30 pp/$\alpha$ [0.99, 3.58] | ${\sim}5$ pp | Graded well-powered; binary underpowered |
| BioASQ | 1,600 | Accuracy | $-0.06$ pp [$-1.5$, 1.4] | -- | ${\sim}2$ pp | Well-powered flat endpoint |

## Appendix C. FaithEval Detailed Rates

Appendix Table C1 preserves the full per-alpha FaithEval rate table summarized in §4.2.

**Appendix Table C1 -- FaithEval Compliance by Intervention Method and Scaling Factor**

| $\alpha$ | Neurons (38) | SAE H-features (266) | SAE random (mean $\pm$ SD, 3 seeds) |
|---|---|---|---|
| 0.0 | 64.2% [61.2, 67.1] | 72.3% [69.4, 75.0] | 74.9% $\pm$ 0.4 |
| 0.5 | 65.4% [62.4, 68.3] | 74.7% [71.9, 77.3] | 74.8% $\pm$ 0.4 |
| 1.0 | 66.0% [63.0, 68.9] | **66.0%** [63.0, 68.9] | **66.0%** $\pm$ 0.0 |
| 1.5 | 67.0% [64.0, 69.8] | 75.0% [72.2, 77.6] | 75.0% $\pm$ 0.2 |
| 2.0 | 68.2% [65.2, 71.0] | 75.1% [72.3, 77.7] | 74.9% $\pm$ 0.1 |
| 2.5 | 69.5% [66.6, 72.3] | 74.9% [72.1, 77.5] | 74.9% $\pm$ 0.1 |
| 3.0 | 70.5% [67.6, 73.2] | 69.9% [67.0, 72.7] | 74.6% $\pm$ 0.5 |

Wilson 95% CIs shown for neurons and SAE H-features ($n = 1{,}000$). $\alpha = 1.0$ is the no-op baseline for both intervention modes.

## Appendix D. Supporting Jailbreak Selector Comparator

The current full-500 selector panel is supporting evidence rather than a main-text figure because it is not fully like-for-like. The table is preserved here so the detailed current-state comparator is still reader-visible.

**Appendix Table D1 -- Current Full-500 Selector Comparator (April 16 Audit)**

| Condition | Strict harmfulness | 95% CI | Paired delta vs. baseline |
|---|---:|---:|---:|
| baseline | 51.6% | [47.2, 55.9] | -- |
| probe | 34.8% | [30.8, 39.1] | -16.8 pp [-20.8, -12.8] |
| random seed 1 | 37.2% | [33.1, 41.5] | -14.4 pp [-19.0, -9.8] |
| random seed 2 | 38.8% | [34.6, 43.1] | -12.8 pp [-17.4, -8.4] |
| causal | 24.8% | [21.2, 28.8] | -26.8 pp [-31.0, -22.6] |

Three caveats remain attached to this table. First, the panel combines branches scored under different evaluation pipelines rather than providing a fully like-for-like comparison. Second, the probe and both random branches are error-bearing. Third, the gradient-ranked branch still shows visible generation-cap artifacts, with 112 of 500 responses hitting the cap at $\alpha = 4.0$. The correct paper-facing sentence is therefore the narrow one used in §4.4: on the current full-500 comparator panel, the gradient-ranked branch is the strongest completed D7 branch.

## Appendix E. Measurement Support

### Appendix Table E1 -- Development-Set Evaluator Comparison

| Evaluator | Judge model | Accuracy | FP | FN |
|---|---|---:|---:|---:|
| CSV2 v3 | GPT-4o | 86.5% | 0 | 10 |
| Binary judge | GPT-4o | 77.0% | 4 | 13 |
| StrongREJECT | GPT-4o-mini | 74.3% | 0 | 19 |
| CSV2 v2 | GPT-4o | 73.0% | 0 | 19 |

This development-set comparison overstates the eventual separation because 24 of the 74 records overlap with CSV-v3 calibration rows. It is preserved here because it still reveals where evaluator disagreement concentrates: outputs with refusal framing followed by harmful substance.

### Worked Evaluator Disagreement Regime

The characteristic disagreement case is a refuse-then-comply output. In the held-out audit, one response began with a refusal preamble and then went on to provide substantive harmful content. StrongREJECT zeroed the score because `refused = 1`, while the CSV2 evaluators and the binary judge all marked the output harmful. This is why §6 frames evaluator disagreement as construct mismatch rather than simple noise.^[Sources: `notes/act3-reports/2026-04-12-4way-evaluator-holdout-validation.md`, §3; `notes/act3-reports/2026-04-12-4way-evaluator-comparison.md`, §3.]

### Schema-Drift Detail

The schema-drift bug discussed in §6.4 applied CSV-v3 normalization logic to CSV-v2 rows, silently reclassifying 167 of 171 borderline records (97.7%) as strict compliance and inflating the strict harmfulness rate from 18.8% to 52.2%. The debugging mechanics are not themselves the scientific claim, but the episode is worth preserving because it shows how easily a plausible-looking evaluator pipeline can erase a real specificity contrast if schema boundaries are not enforced.^[Source: `notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md`, §6.1.]
