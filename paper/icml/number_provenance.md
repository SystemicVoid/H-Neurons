# Number Provenance Ledger

Reviewer-facing provenance ledger for the current ICML submission in `paper/icml/main.tex`.
Section references below follow the TeX manuscript, not the older long-form markdown draft.
Abstract headline numbers are covered by the section-level rows below rather than duplicated.

## Main-Body Quantitative Claims

### Section 3. Similar Readout Quality Does Not Guarantee Control (`\S\ref{sec:localization}`)

#### §3.1 The Readouts Are Real

| Claim | Value | CI / exact test | Canonical source |
|---|---|---|---|
| H-neuron AUROC | 0.843 | [0.815, 0.870] | `data/gemma3_4b/pipeline/classifier_disjoint_summary.json` (`evaluation.metrics.auroc`) |
| H-neuron disjoint accuracy | 76.5% | [73.6, 79.5] | same file (`evaluation.metrics.accuracy`) |
| H-neuron detector test size | 780 | — | same file (`evaluation.n_examples`) |
| H-neuron count | 38 / 348,160 | — | `data/gemma3_4b/pipeline/classifier_disjoint_summary.json` (`selected_h_neurons`); `data/gemma3_4b/pipeline/classifier_structure_summary.json` (`total_ffn_neurons`) |
| SAE feature AUROC | 0.848 | [0.820, 0.874] | `data/gemma3_4b/pipeline/classifier_sae_summary.json` (`best.test_metrics.auroc`) |
| SAE feature disjoint accuracy | 77.2% | [74.3, 80.2] | same file (`best.test_metrics.accuracy`) |
| SAE detector test size | 782 | — | same file (`best.test_metrics.n_examples`) |
| SAE positive features selected | 266 across 10 layers | — | same file (`best.n_positive_features`; `extraction_metadata.layer_indices`) |

#### §3.2 Comparable Readouts, Divergent Control

| Claim | Value | CI / exact test | Canonical source |
|---|---|---|---|
| H-neuron FaithEval slope | +2.09 pp/α | [1.38, 2.83] | `data/gemma3_4b/intervention/faitheval/control/comparison_summary.json` (`h_neuron_baseline.slope_per_alpha`); canonical reported CI in `data/gemma3_4b/intervention/faitheval/experiment/results.json` (`effects.compliance_curve.slope_pp_per_alpha`) |
| H-neuron monotonicity | Spearman ρ = 1.0 | — | `data/gemma3_4b/intervention/faitheval/control/comparison_summary.json` (`h_neuron_baseline.spearman_rho`) |
| H-neuron FaithEval no-op→max (`α = 1.0 → 3.0`) | +4.5 pp | [2.9, 6.1] | `data/gemma3_4b/intervention/faitheval/experiment/results.json` (`delta_noop_to_max_pp`) |
| H-neuron FaithEval full sweep (`α = 0.0 → 3.0`) | +6.3 pp | [4.2, 8.5] | same file (`delta_0_to_max_pp`) |
| Random-neuron null summary (five unconstrained seeds) | mean slope +0.02 pp/α | descriptive seed mean | `data/gemma3_4b/intervention/faitheval/control/comparison_summary.json` (`unconstrained_random.per_seed.*.slope_per_alpha` mean); verified in `notes/act3-reports/2026-04-13-faitheval-slope-difference-reporting-audit.md` §1.2 |
| Random-neuron null summary (all eight seeds) | max slope +0.21 pp/α | descriptive seed max | same file (`unconstrained_random.per_seed.*.slope_per_alpha`, `layer_matched_random.per_seed.*.slope_per_alpha`); verified in same audit |
| Random-neuron specificity | 8/8 paired neuron-minus-random differences positive | every seed-specific 95% CI excludes zero | `data/gemma3_4b/intervention/faitheval/control/slope_difference_summary.json` (`aggregate`, `per_seed`); supporting audit: `notes/act3-reports/2026-04-13-faitheval-slope-difference-reporting-audit.md` §1.3 |
| SAE H-feature slope | +0.16 pp/α | [-0.51, 0.84] | `data/gemma3_4b/intervention/faitheval_sae/control/comparison_summary.json` (`h_sae_features.slope_per_alpha`); canonical reported CI in `data/gemma3_4b/intervention/faitheval_sae/sae_pipeline_audit.md` |
| SAE monotonicity | Spearman ρ = 0.18 | — | `data/gemma3_4b/intervention/faitheval_sae/control/comparison_summary.json` (`h_sae_features.spearman_rho`) |
| Delta-only SAE H-feature slope | +0.12 pp/α | audit summary; no slope CI in package | `data/gemma3_4b/intervention/faitheval_sae_delta/control/comparison_summary.json` (`h_sae_features.slope_per_alpha`); audit discussion in `data/gemma3_4b/intervention/faitheval_sae/sae_pipeline_audit.md` §5 |
| Delta-only random-feature slope | -0.09 pp/α | audit summary; no slope CI in package | `data/gemma3_4b/intervention/faitheval_sae_delta/control/comparison_summary.json` (`random_sae_features.mean_slope_per_alpha`); audit discussion in `data/gemma3_4b/intervention/faitheval_sae/sae_pipeline_audit.md` §5 |
| Neuron baseline on same three-alpha subset | +2.12 pp/α | 3-point OLS; no slope CI in package | `data/gemma3_4b/intervention/faitheval_sae_delta/control/comparison_summary.json` (`neuron_baseline.slope_per_alpha`) |
| SAE random slope | +0.59 pp/α | descriptive seed mean | `data/gemma3_4b/intervention/faitheval_sae/control/comparison_summary.json` (`random_sae_features.per_seed.*.slope_per_alpha` mean) |
| Neuron-minus-SAE slope difference | +1.93 pp/α | [+0.94, +2.92] | `data/gemma3_4b/intervention/faitheval_sae/control/slope_difference_summary.json` (`slope_difference_pp_per_alpha`) |
| Neuron-minus-SAE directional permutation test | one-sided p < 0.001 | saved value `9.9998e-05` | same file (`permutation_test`); supporting audit: `notes/act3-reports/2026-04-13-faitheval-slope-difference-reporting-audit.md` §1.3 |
| FaithEval evaluation size | 1,000 items | — | `notes/measurement-blueprint.md` |
| Within-SAE selector split | 160 validation / 840 test | stratified, seed 42 | `data/gemma3_4b/intervention/faitheval_sae_utility_selector/report/heldout_summary.json` (`selector_design.validation_policy`); audit report: `paper/icml/reports/2026-04-25-faitheval-answer-span-extension.md` |
| Within-SAE candidate pool | 509 probe-nonzero SAE features | k = 266 selector size | same file (`selector_diagnostics.candidate_pool_n`, `selected_k`, `answer_span_selected_k`) |
| Readout-selected prompt-end margin effect | +0.92 nats | [+0.61, +1.23] | `paper/icml/reports/2026-04-25-faitheval-answer-span-extension.md` §3.4; canonical means in `heldout_summary.json` (`heldout_anti_compliance_margin.families`) |
| Prompt-end utility-selected prompt-end margin effect | -0.76 nats | [-1.08, -0.42] | `data/gemma3_4b/intervention/faitheval_sae_utility_selector/report/heldout_summary.json` (`heldout_anti_compliance_margin.paired_deltas.utility_minus_noop`) |
| Prompt-end utility minus readout prompt-end margin | -1.68 nats | [-2.17, -1.19] | same file (`heldout_anti_compliance_margin.paired_deltas.utility_minus_readout`) |
| Path-drift prompt-end margin effect | +8.2e-8 nats | [-2.4e-3, +2.1e-3] | same file (`heldout_anti_compliance_margin.path_drift.paired_delta`) |
| Answer-span selector compliance effect | +0.95 pp | [-0.83, +2.74] | same file (`heldout_compliance_answer_span_selected.paired_deltas_pp.answer_span_selected_minus_noop`) |
| Answer-span selector native answer-text margin effect | -0.33 nats | [-0.61, -0.06] | same file (`heldout_answer_span_margin.paired_deltas.answer_span_selected_minus_noop`) |
| Answer-span selector native random-null contrast | -0.53 nats | nested bootstrap [-0.81, -0.26] | same file (`heldout_answer_span_margin.across_seeds.seed_mean_summary`) |
| Answer-span selector vs prompt-end utility on prompt-end margin | +0.46 nats | [+0.16, +0.75] | same file (`heldout_anti_compliance_margin_answer_span_selected.paired_deltas.answer_span_selected_minus_utility_selected`) |
| Prompt-end utility / answer-span selector overlap | 167 / 266 shared features | Jaccard 0.46 | same file (`selector_diagnostics.answer_span_overlap_with_utility`) |

### Section 4. Control Is Surface-Local and Can Externalize (`\S\ref{sec:externality}`)

#### §4.1 Positive Results Are Task-Local

| Claim | Value | CI / exact test | Canonical source |
|---|---|---|---|
| FalseQA slope | +1.62 pp/α | [0.52, 2.74] | `data/gemma3_4b/intervention/falseqa/experiment/results.json` (`effects.compliance_curve.slope_pp_per_alpha`) |
| FalseQA no-op→max (`α = 1.0 → 3.0`) | +2.5 pp | [-0.6, 5.5] | same file (`effects.compliance_curve.delta_noop_to_max_pp`) |
| FalseQA full sweep (`α = 0.0 → 3.0`) | +4.8 pp | [1.3, 8.3] | same file (`effects.compliance_curve.delta_0_to_max_pp`) |
| FalseQA sample size | 687 | — | same file (`results."0.0".compliance.n_total`) |
| BioASQ net accuracy effect | -0.06 pp | [-1.5, 1.4] | `data/gemma3_4b/intervention/bioasq/bioasq_pipeline_audit.md` |
| BioASQ sample size | 1,600 | — | same file |
| BioASQ responses behaviorally changed | 1,339 / 1,600 | — | same file |

#### §4.2 ITI: Answer Selection vs. Open-Ended Generation

| Claim | Value | CI / exact test | Canonical source |
|---|---|---|---|
| ITI MC1 effect (held-out TruthfulQA folds) | +6.3 pp | [+3.7, +8.9] | `notes/act3-reports/2026-04-01-priority-reruns-audit.md` §1 |
| ITI MC2 effect (held-out TruthfulQA folds) | +7.49 pp | [+5.28, +9.82] | same file |
| TruthfulQA flip counts | 61 wrong→right, 20 right→wrong | — | same file §1 (`Pooled D4 ITI, α=0.0 → α=8.0`) |
| SimpleQA correct-answer rate at `α = 0.0` | 4.6% | [3.5, 6.1] | same file §3a |
| SimpleQA correct-answer rate at `α = 8.0` | 2.8% | [1.9, 4.0] | same file §3a |
| SimpleQA paired compliance delta (`α = 0.0 → 8.0`) | -1.8 pp | [-3.1, -0.6] | same file §3b |
| SimpleQA attempt rate at `α = 0.0` | 99.7% | — | same file §3a |
| SimpleQA attempt rate at `α = 8.0` | 67.0% | — | same file §3a |
| SimpleQA paired attempt-rate delta (`α = 0.0 → 8.0`) | -32.7 pp | [-35.6, -29.9] | same file §3b |

#### §4.3 Bridge: Wrong-Entity Substitution

| Claim | Value | CI / exact test | Canonical source |
|---|---|---|---|
| Bridge baseline adjudicated accuracy | 45.0% | [40.7, 49.4] | `notes/act3-reports/2026-04-13-bridge-phase3-test-results.md` §2.1 |
| E0 `α = 8.0` adjudicated delta | -5.8 pp | [-8.8, -3.0] | same file §2.2 |
| E0 `α = 8.0` deterministic delta | -4.6 pp | [-7.6, -1.6] | same file §2.2 |
| Bridge paired significance | McNemar p = 0.0002 | — | same file §2.2 |
| Bridge flip counts | 43 right→wrong, 14 wrong→right | net -29 | same file §2.3 |
| Wrong-entity substitution (adjudicated, R→W) | 31 / 43 (72.1%) | [57.3, 83.3] (Wilson) | `data/judge_validation/bridge_irr/bridge_irr_summary.json` (`direction_summaries.right_to_wrong.categories.wrong_entity_substitution`) |
| Evasion / factual denial (adjudicated, R→W) | 9 / 43 (20.9%) | [11.4, 35.2] (Wilson) | same file (`...evasion_or_factual_denial`) |
| Answer dilution / verbosity (adjudicated, R→W) | 3 / 43 (7.0%) | [2.4, 18.6] (Wilson) | same file (`...answer_dilution`) |
| Formal refusal among right→wrong flips (adjudicated) | 0 / 43 (0.0%) | [0.0, 8.2] (Wilson) | same file (`...formal_refusal`) |
| Dual-rater raw agreement (all 57 discordant) | 55 / 57 (96.5%) | [88.1, 99.0] (Wilson) | same file (`irr.raw_agreement`) |
| Cohen's κ (A vs B, 4-category) | 0.90 | — | same file (`irr.cohen_kappa`) |
| Gwet's AC1 (A vs B, 4-category) | 0.96 | — | same file (`irr.gwet_ac1`) |
| Adjudicated disagreements | 2 / 57 | — | same file (`irr.n_disagreements`, `adjudication`) |
| NOT_ATTEMPTED counts (baseline / ITI) | 2 / 8 | — | `notes/act3-reports/2026-04-13-bridge-phase3-test-results.md` §7 |
| Bridge test set size | 500 | — | same file §1 |

### Section 5. Measurement Choices Changed the Conclusion (`\S\ref{sec:measurement}`)

| Claim | Value | CI / exact test | Canonical source |
|---|---|---|---|
| Binary harmful endpoint shift (`α = 0.0 → 3.0`) | 152 / 500 → 167 / 500 | delta +3.0 pp; CI includes zero | `notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md` §1.2 |
| H-neuron graded harmfulness slope | +2.30 pp/α | [+0.99, +3.58] | same file §2 |
| H-neuron graded endpoint shift (`α = 0.0 → 3.0`) | +7.6 pp | [+2.6, +12.8] | same file §0 / §2 |
| Random-neuron graded slope (seed-0 control) | -0.47 pp/α | [-1.42, +0.47] | same file §2 |
| H-minus-random slope difference | +2.77 pp/α | [+1.17, +4.42] | same file §2 |
| H-minus-random permutation test | p = 0.013 | one-sided | same file §2 |
| Holdout evaluator accuracy: CSV-v3 (GPT-4o) | 96.0% | [90.0, 100.0] | `data/judge_validation/holdout_comparison.json` (`bootstrap_cis.evaluators.csv2v3.accuracy`) |
| Holdout evaluator accuracy: StrongREJECT (GPT-4o) | 96.0% | [90.0, 100.0] | same file (`bootstrap_cis.evaluators.sr.accuracy`) |
| Holdout evaluator accuracy: CSV-v2 (GPT-4o) | 92.0% | [84.3, 98.0] | same file (`bootstrap_cis.evaluators.csv2v2.accuracy`) |
| Holdout evaluator accuracy: Binary judge (GPT-4o) | 90.0% | [80.0, 98.0] | same file (`bootstrap_cis.evaluators.binary.accuracy`) |
| Holdout evaluator sample size | 50 | contamination-clean holdout | same file (`cross_alpha_holdout` totals); supporting report: `notes/act3-reports/2026-04-12-4way-evaluator-holdout-validation.md` |
| Evaluator tie statement kept in main text | CSV-v3 and StrongREJECT have identical error sets | 0 discordant records; McNemar p = 1.0 | `data/judge_validation/holdout_comparison.json` (`mcnemar_tests.csv2v3_vs_sr`); supporting report: `notes/act3-reports/2026-04-12-4way-evaluator-holdout-validation.md` |
| Pairwise holdout significance summary | no pairwise holdout difference statistically confirmable | all six McNemar exact p-values ≥ 0.25 | `data/judge_validation/holdout_comparison.json` (`mcnemar_tests`) |

## Supporting Rows Not Currently Quoted as Main-Body ICML Headlines

These rows remain useful for supplement or rebuttal support, but they are not
currently front-line numeric claims in `paper/icml/main.tex`.

| Claim | Value | CI / exact test | Canonical source |
|---|---|---|---|
| Probe selector pilot: best effect | ~-2 pp strict harmfulness rate | [-10, +6] | `notes/act3-reports/2026-04-07-d7-causal-pilot-audit.md` |
| Gradient-ranked pilot effect | -13 pp strict harmfulness rate | [-21, -6] | same file |
| Probe-causal Jaccard | 0.11 | — | same file |
| Historical April 8 legacy-ruler causal full-500 effect | -9.0 pp strict harmfulness rate | [-12.2, -5.8] | `notes/act3-reports/2026-04-08-d7-full500-audit.md` (historical provenance only) |
| Current full-500 normalized baseline strict harmfulness rate | 51.6% | Wilson 95% CI [47.2, 56.0] | `data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/d7_full500_current_state_summary.json` (`current_panel.conditions.baseline.strict_harmfulness_normalized`) |
| Current full-500 normalized probe strict harmfulness rate | 34.8% | Wilson 95% CI [30.8, 39.1] | same file (`current_panel.conditions.probe.strict_harmfulness_normalized`) |
| Current full-500 normalized random seed 1 strict harmfulness rate | 37.2% | Wilson 95% CI [33.1, 41.5] | same file (`current_panel.conditions.random_layer_seed1.strict_harmfulness_normalized`) |
| Current full-500 normalized random seed 2 strict harmfulness rate | 38.8% | Wilson 95% CI [34.6, 43.1] | same file (`current_panel.conditions.random_layer_seed2.strict_harmfulness_normalized`) |
| Current full-500 normalized causal strict harmfulness rate | 24.8% | Wilson 95% CI [21.2, 28.8] | same file (`current_panel.conditions.causal.strict_harmfulness_normalized`) |
| Current full-500 normalized causal vs probe gap | -10.0 pp | [-14.0, -6.2] | same file (`current_panel.direct_comparisons.probe_vs_causal.strict_harmfulness_normalized`, sign-flipped for causal-minus-probe phrasing) |
| Current full-500 normalized causal vs random seed 1 gap | -12.4 pp | [-16.8, -8.0] | same file (`current_panel.direct_comparisons.random_layer_seed1_vs_causal.strict_harmfulness_normalized`, sign-flipped for causal-minus-random phrasing) |
| Current full-500 normalized causal vs random seed 2 gap | -14.0 pp | [-18.2, -10.0] | same file (`current_panel.direct_comparisons.random_layer_seed2_vs_causal.strict_harmfulness_normalized`, sign-flipped for causal-minus-random phrasing) |
| Current full-500 token-cap hits kept for truncation discussion | 112 / 500 (22.4%) | — | same file (`artifact_status.causal_locked.token_cap`) |
| StrongREJECT GPT-4o dev accuracy after rerun | 78.4% | [67.7, 86.2] | `notes/act3-reports/2026-04-13-jailbreak-measurement-cleanup.md` §3.2 |
| StrongREJECT GPT-4o-mini dev accuracy (prior) | 74.3% | [63.3, 82.9] | `notes/act3-reports/2026-04-12-4way-evaluator-comparison.md` §0 |
| Holdout gap after StrongREJECT model upgrade | 0.0 pp | — | `notes/act3-reports/2026-04-13-jailbreak-measurement-cleanup.md` §3.4 |
| False negatives recovered by StrongREJECT model upgrade | 3 of 19 | — | same file §3.2 |
| Persistent StrongREJECT false negatives after upgrade | 16 (all `refused = 1`) | — | same file §3.3 |
| Contamination bug: borderline records reclassified | 167 / 171 (97.7%) | — | `notes/act3-reports/2026-04-12-seed0-jailbreak-control-audit.md` §6 |
| Contamination bug: strict harmfulness inflation | 18.8% → 52.2% | — | same file §6 |
