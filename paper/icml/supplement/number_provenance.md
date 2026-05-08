# Number Provenance Ledger

Reviewer-facing provenance ledger for the ICML submission in [`reference/main.tex`](reference/main.tex). Section references follow the anonymous TeX manuscript bundled in this supplement.

Each listed row resolves in one hop to a bundled reviewer-facing support file or bundled safe JSON artifact.

## Main-Body Quantitative Claims

### Section 3. Similar Readout Quality Does Not Guarantee Control (`\S\ref{sec:localization}`)

#### §3.1 The Readouts Are Real

| Claim | Value | CI / exact test | Bundled source |
|---|---|---|---|
| H-neuron AUROC | 0.843 | [0.815, 0.870] | `support/localization_summary.md` |
| H-neuron disjoint accuracy | 76.5% | [73.6, 79.5] | `support/localization_summary.md` |
| H-neuron detector test size | 780 | — | `support/localization_summary.md` |
| H-neuron count | 38 / 348,160 | — | `support/localization_summary.md` |
| SAE feature AUROC | 0.848 | [0.820, 0.874] | `support/localization_summary.md` |
| SAE feature disjoint accuracy | 77.2% | [74.3, 80.2] | `support/localization_summary.md` |
| SAE detector test size | 782 | — | `support/localization_summary.md` |
| SAE positive features selected | 266 across 10 layers | — | `support/localization_summary.md` |

#### §3.2 Comparable Readouts, Divergent Control

| Claim | Value | CI / exact test | Bundled source |
|---|---|---|---|
| H-neuron FaithEval slope | +2.09 pp/α | [1.38, 2.83] | `support/localization_summary.md` |
| H-neuron monotonicity | Spearman ρ = 1.0 | — | `support/localization_summary.md` |
| H-neuron FaithEval no-op→max (`α = 1.0 → 3.0`) | +4.5 pp | [2.9, 6.1] | `support/localization_summary.md` |
| H-neuron FaithEval full sweep (`α = 0.0 → 3.0`) | +6.3 pp | [4.2, 8.5] | `support/localization_summary.md` |
| Random-neuron null summary (five unconstrained seeds) | mean slope +0.02 pp/α | descriptive seed mean | `support/localization_summary.md` |
| Random-neuron null summary (all eight seeds) | max slope +0.21 pp/α | descriptive seed max | `support/localization_summary.md` |
| Random-neuron specificity | 8/8 paired neuron-minus-random differences positive | every seed-specific 95% CI excludes zero | `support/localization_summary.md` |
| SAE H-feature slope | +0.16 pp/α | [-0.51, 0.84] | `support/localization_summary.md` |
| SAE monotonicity | Spearman ρ = 0.18 | — | `support/localization_summary.md` |
| Delta-only SAE H-feature slope | +0.12 pp/α | audit summary; no slope CI in package | `support/localization_summary.md` |
| Delta-only random-feature slope | -0.09 pp/α | audit summary; no slope CI in package | `support/localization_summary.md` |
| Neuron baseline on same three-alpha subset | +2.12 pp/α | audit summary; 3-point OLS on `α ∈ {0.0, 1.0, 3.0}`; no slope CI in package | `support/localization_summary.md` |
| SAE random slope | +0.59 pp/α | descriptive seed mean | `support/localization_summary.md` |
| Neuron-minus-SAE slope difference | +1.93 pp/α | [+0.94, +2.92] | `support/localization_summary.md` |
| Neuron-minus-SAE directional permutation test | one-sided `p < 0.001` | saved value `9.9998e-05` | `support/localization_summary.md` |
| FaithEval evaluation size | 1,000 items | — | `support/localization_summary.md` |

### Section 4. Control Is Surface-Local and Can Externalize (`\S\ref{sec:externality}`)

#### §4.1 Positive Results Are Task-Local

| Claim | Value | CI / exact test | Bundled source |
|---|---|---|---|
| FalseQA slope | +1.62 pp/α | [0.52, 2.74] | `support/externality_summary.md` |
| FalseQA no-op→max (`α = 1.0 → 3.0`) | +2.5 pp | [-0.6, 5.5] | `support/externality_summary.md` |
| FalseQA full sweep (`α = 0.0 → 3.0`) | +4.8 pp | [1.3, 8.3] | `support/externality_summary.md` |
| FalseQA sample size | 687 | — | `support/externality_summary.md` |
| BioASQ net accuracy effect | -0.06 pp | [-1.5, 1.4] | `support/externality_summary.md` |
| BioASQ sample size | 1,600 | — | `support/externality_summary.md` |
| BioASQ responses behaviorally changed | 1,339 / 1,600 | — | `support/externality_summary.md` |

#### §4.2 ITI: Answer Selection vs. Open-Ended Generation

| Claim | Value | CI / exact test | Bundled source |
|---|---|---|---|
| ITI MC1 effect (held-out TruthfulQA folds) | +6.3 pp | [+3.7, +8.9] | `support/externality_summary.md` |
| ITI MC2 effect (held-out TruthfulQA folds) | +7.49 pp | [+5.28, +9.82] | `support/externality_summary.md` |
| TruthfulQA flip counts | 61 wrong→right, 20 right→wrong | — | `support/externality_summary.md` |
| SimpleQA evaluation size | 1,000 items | — | `support/externality_summary.md` |
| SimpleQA correct-answer rate at `α = 0.0` | 4.6% | [3.5, 6.1] | `support/externality_summary.md` |
| SimpleQA correct-answer rate at `α = 8.0` | 2.8% | [1.9, 4.0] | `support/externality_summary.md` |
| SimpleQA paired compliance delta (`α = 0.0 → 8.0`) | -1.8 pp | [-3.1, -0.6] | `support/externality_summary.md` |
| SimpleQA attempt rate at `α = 0.0` | 99.7% | — | `support/externality_summary.md` |
| SimpleQA attempt rate at `α = 8.0` | 67.0% | — | `support/externality_summary.md` |
| SimpleQA paired attempt-rate delta (`α = 0.0 → 8.0`) | -32.7 pp | [-35.6, -29.9] | `support/externality_summary.md` |
| SimpleQA precision among attempted answers at `α = 0.0` | 4.6% | — | `support/externality_summary.md` |
| SimpleQA precision among attempted answers at `α = 8.0` | 4.2% | — | `support/externality_summary.md` |
| SimpleQA paired precision delta (`α = 0.0 → 8.0`) | -0.4 pp | [-1.9, +1.1] | `support/externality_summary.md` |

#### §4.3 Bridge: Wrong-Entity Substitution

| Claim | Value | CI / exact test | Bundled source |
|---|---|---|---|
| Bridge baseline adjudicated accuracy | 45.0% | [40.7, 49.4] | `support/externality_summary.md` |
| E0 `α = 8.0` adjudicated delta | -5.8 pp | [-8.8, -3.0] | `support/externality_summary.md` |
| E0 `α = 8.0` deterministic delta | -4.6 pp | [-7.6, -1.6] | `support/externality_summary.md` |
| Bridge paired significance | McNemar `p = 0.0002` | — | `support/externality_summary.md` |
| Bridge flip counts | 43 right→wrong, 14 wrong→right | net -29 | `support/externality_summary.md` |
| Wrong-entity substitution | 31 / 43 (72.1%) | [57.3, 83.3] (Wilson) | `support/externality_summary.md` |
| Evasion / factual denial | 9 / 43 (20.9%) | [11.4, 35.2] (Wilson) | `support/externality_summary.md` |
| Answer dilution / verbosity | 3 / 43 (7.0%) | [2.4, 18.6] (Wilson) | `support/externality_summary.md` |
| Formal refusal among right→wrong flips | 0 / 43 (0.0%) | [0.0, 8.2] (Wilson) | `support/externality_summary.md` |
| Dual-rater raw agreement (all 57 discordant) | 55 / 57 (96.5%) | [88.1, 99.0] (Wilson) | `support/externality_summary.md` |
| Cohen's κ (A vs B, 4-category) | 0.90 | — | `support/externality_summary.md` |
| Gwet's AC1 (A vs B, 4-category) | 0.96 | — | `support/externality_summary.md` |
| NOT_ATTEMPTED counts (baseline / ITI) | 2 / 8 | — | `support/externality_summary.md` |
| Bridge test set size | 500 | — | `support/externality_summary.md` |

### Section 5. Measurement Choices Changed the Conclusion (`\S\ref{sec:measurement}`)

| Claim | Value | CI / exact test | Bundled source |
|---|---|---|---|
| Binary harmful endpoint shift (`α = 0.0 → 3.0`) | 152 / 500 → 167 / 500 | delta +3.0 pp; CI includes zero | `support/measurement_summary.md` |
| H-neuron CSV-v2 strict-harmfulness slope | +2.30 pp/α | [+0.99, +3.58] | `support/measurement_summary.md` |
| H-neuron graded endpoint shift (`α = 0.0 → 3.0`) | +7.6 pp | [+2.6, +12.8] | `support/measurement_summary.md` |
| Random-neuron CSV-v2 strict-harmfulness slope (seed-0 control) | -0.47 pp/α | [-1.42, +0.47] | `support/measurement_summary.md` |
| CSV-v2 H-minus-random slope difference | +2.77 pp/α | [+1.17, +4.42] | `support/measurement_summary.md` |
| H-minus-random permutation test | `p = 0.013` | one-sided | `support/measurement_summary.md` |
| H-neuron CSV-v3 `harmful_binary` slope | +0.46 pp/α | point-estimate audit summary | `support/measurement_summary.md` |
| Random-neuron CSV-v3 `harmful_binary` slope | -0.34 pp/α | point-estimate audit summary | `support/measurement_summary.md` |
| CSV-v3 H-minus-random slope difference | +0.80 pp/α | point-estimate audit summary; no packaged gap CI or permutation test | `support/measurement_summary.md` |
| Holdout evaluator accuracy: CSV-v3 (GPT-4o) | 96.0% | [90.0, 100.0] | `support/measurement_summary.md`; `data/judge_validation/holdout_comparison.json` |
| Holdout evaluator accuracy: StrongREJECT (GPT-4o) | 96.0% | [90.0, 100.0] | `support/measurement_summary.md`; `data/judge_validation/holdout_comparison.json` |
| Holdout evaluator accuracy: CSV-v2 (GPT-4o) | 92.0% | [84.3, 98.0] | `support/measurement_summary.md`; `data/judge_validation/holdout_comparison.json` |
| Holdout evaluator accuracy: Binary judge (GPT-4o) | 90.0% | [80.0, 98.0] | `support/measurement_summary.md`; `data/judge_validation/holdout_comparison.json` |
| Holdout evaluator sample size | 50 | contamination-clean holdout | `support/measurement_summary.md`; `data/judge_validation/holdout_comparison.json` |
| Evaluator tie statement kept in main text | CSV-v3 and StrongREJECT have identical error sets | 0 discordant records; McNemar `p = 1.0` | `support/measurement_summary.md`; `data/judge_validation/holdout_comparison.json` |
| Pairwise holdout significance summary | no pairwise holdout difference statistically confirmable | all six McNemar exact `p` values `>= 0.25` | `support/measurement_summary.md`; `data/judge_validation/holdout_comparison.json` |

## Detailed Claim-Defense Ledger

This is the full claim-defense ledger moved out of the manuscript appendix for readability. Brace notation expands metric-ID prefixes. Internal artifact IDs name the source-repository artifacts used to produce the reviewer-facing summaries; bundled support files provide the one-hop supplement surface.

### Comparable Readouts

| Field | Entry |
|---|---|
| Model / surface | Gemma-3-4B-IT / FaithEval readout |
| Exact metric IDs / internal artifacts | `readout.disjoint.{auroc,accuracy}` / `classifier_disjoint_summary`; `readout.sae.{auroc,accuracy}` / `classifier_sae_summary` |
| Estimator / CI | Unpaired; stratified bootstrap for H-neuron readout; SAE point and CI summary. |
| Gate status | Localization / monitoring passed. |
| Caveat | Individual-weight interpretation remains fragile. |
| Bundled support | `support/localization_summary.md` |

### Gemma FaithEval H-Neuron Control

| Field | Entry |
|---|---|
| Model / surface | Gemma-3-4B-IT / FaithEval anti-compliance, neuron scaling |
| Exact metric IDs / internal artifacts | `intervention.faitheval.anti.{slope_pp_per_alpha,delta_noop_to_max_pp,delta_0_to_max_pp}` / `intervention_sweep_site`; `intervention.faitheval.{random_mean_slope_pp_per_alpha,random_max_slope_pp_per_alpha}` / `faitheval_control_comparison` |
| Estimator / CI | Paired bootstrap for slope / deltas; random-control summaries. |
| Gate status | FaithEval control and specificity passed. |
| Caveat | Surface-local compliance claim; no general factuality claim. |
| Bundled support | `support/localization_summary.md` |

### SAE H-Feature Null

| Field | Entry |
|---|---|
| Model / surface | Gemma-3-4B-IT / FaithEval, tested Gemma Scope SAE operators |
| Exact metric IDs / internal artifacts | `intervention.faitheval_sae.{h_slope_pp_per_alpha,neuron_minus_sae_slope_pp_per_alpha,permutation_p}` / `faitheval_sae_control_comparison`, `faitheval_sae_slope_difference` |
| Estimator / CI | SAE slope summary; paired bootstrap gap; permutation test. |
| Gate status | SAE control gate did not pass; localization-to-control dissociation passed. |
| Caveat | Operator-form and layer-coverage limits remain. |
| Bundled support | `support/localization_summary.md` |

### Within-SAE Selector Defense

| Field | Entry |
|---|---|
| Model / surface | Gemma-3-4B-IT / FaithEval, 509-feature SAE pool |
| Exact metric IDs / internal artifacts | `intervention.faitheval_sae_utility.{utility_minus_noop_compliance_pp,utility_minus_noop_margin,utility_minus_readout_margin}` / `faitheval_sae_utility_selector_heldout`; `heldout_summary.json::{heldout_answer_span_margin,heldout_compliance_answer_span_selected}` |
| Estimator / CI | Paired bootstrap for compliance; continuous bootstrap for margins and answer-span locators. |
| Gate status | No selector reaches the compliance endpoint. |
| Caveat | Margin-only, metric-specific, cross-metric mixed. |
| Bundled support | `support/localization_summary.md` |

### Surface-Local H-Neuron Behavior

| Field | Entry |
|---|---|
| Model / surface | Gemma-3-4B-IT / FalseQA and BioASQ, neuron scaling |
| Exact metric IDs / internal artifacts | `intervention.falseqa.{slope_pp_per_alpha,delta_0_to_max_pp}` / `falseqa_results`; `intervention.bioasq.{delta_0_to_max_pp,changed_response_count}` / `bioasq_results`, `bioasq_alpha_0_rows`, `bioasq_alpha_3_rows` |
| Estimator / CI | Paired bootstrap for effects; paired response-join count for text change. |
| Gate status | FalseQA passes; BioASQ factual-accuracy endpoint fails / null. |
| Caveat | Alias and domain surface; no general factuality gain. |
| Bundled support | `support/externality_summary.md` |

### ITI MC vs. Generation

| Field | Entry |
|---|---|
| Model / surface | Gemma-3-4B-IT / TruthfulQA MC and SimpleQA open generation |
| Exact metric IDs / internal artifacts | `transfer.truthfulqa_mc.{mc1.iti.delta_pp,mc2.iti.delta_pp}` / TruthfulQA ITI fold artifacts; `transfer.simpleqa.{compliance_delta_0_to_8_pp,attempt_delta_0_to_8_pp}` / `simpleqa_ranked_results`, `simpleqa_alpha_0_rows`, `simpleqa_alpha_8_rows` |
| Estimator / CI | Paired bootstrap on matched items / folds. |
| Gate status | MC answer-selection passed; open-generation transfer failed. |
| Caveat | MC / generation mismatch. |
| Bundled support | `support/externality_summary.md` |

### Bridge Externality

| Field | Entry |
|---|---|
| Model / surface | Gemma-3-4B-IT / TriviaQA bridge open generation, ITI |
| Exact metric IDs / internal artifacts | `transfer.bridge.{adjudicated_accuracy_delta_pp,mcnemar_p,base_correct_iti_wrong,base_wrong_iti_correct}` / `bridge_phase3_site`; `measurement.bridge_irr.{right_to_wrong.wrong_entity_substitution,cohen_kappa}` / `bridge_irr_summary` |
| Estimator / CI | Report percentile CI; McNemar / flips; Wilson share; kappa. |
| Gate status | Externality failed and failure taxonomy accepted. |
| Caveat | LLM second rater; wide Wilson interval. |
| Bundled support | `support/externality_summary.md`; `failure_coding_manifest.md`; `data/judge_validation/bridge_irr/bridge_irr_summary.json` |

### Bridge Margin Boundary

| Field | Entry |
|---|---|
| Model / surface | Gemma-3-4B-IT / TriviaQA bridge teacher-forced margins |
| Exact metric IDs / internal artifacts | `transfer.bridge_margins.{a_rw_substitution.first3_shift_nats,b_rw_nonsubstitution.first3_shift_nats,c_wr_rescue.first3_shift_nats,a_vs_d.first3_shift_nats_gap,a_vs_d.first3_shift_nats_p}` / `bridge_margins_test_results` |
| Estimator / CI | Paired within-cohort bootstrap; unpaired gap; permutation test. |
| Gate status | Broad margin sanity passed; substitution-specific mechanism failed. |
| Caveat | Behavioral diagnosis only. |
| Bundled support | This ledger section and the bridge margin analysis in the manuscript appendix. |

### Jailbreak Measurement Granularity

| Field | Entry |
|---|---|
| Model / surface | Gemma-3-4B-IT / JailbreakBench, neuron scaling |
| Exact metric IDs / internal artifacts | `intervention.jailbreak.binary.delta_0_to_max_pp` / `jailbreak_results`; `measurement.jailbreak.v2.{h_slope_csv2_yes_pp_per_alpha,random_mean_slope_csv2_yes_pp_per_alpha,gap_h_minus_random_mean_pp_per_alpha,h_minus_random_permutation_p}` / `jailbreak_control_v2`, `jailbreak_seed0_control_audit` |
| Estimator / CI | Binary paired bootstrap; graded v2 summaries; seed-0 permutation. |
| Gate status | Graded measurement detects signal; binary endpoint inconclusive. |
| Caveat | Single-seed specificity. |
| Bundled support | `support/measurement_summary.md` |

### Evaluator Holdout

| Field | Entry |
|---|---|
| Model / surface | Gemma-3-4B-IT / JailbreakBench evaluator holdout |
| Exact metric IDs / internal artifacts | `measurement.holdout.{csv2v3.accuracy,sr.accuracy,csv2v3_vs_sr_discordant_count,csv2v3_vs_sr_mcnemar_p,min_pairwise_mcnemar_p}` / `holdout_comparison` |
| Estimator / CI | Cluster bootstrap by prompt ID; McNemar / discordance counts. |
| Gate status | CSV-v3 superiority claim withheld; construct-pluralism caveat retained. |
| Caveat | `n = 50` holdout. |
| Bundled support | `support/measurement_summary.md`; `data/judge_validation/holdout_comparison.json` |

## Scope Note

This supplement ledger is intentionally limited to main-body paper claims. Historical-only and appendix-only quantitative rows from the broader repository are omitted here unless they are needed to support a main-body statement.
