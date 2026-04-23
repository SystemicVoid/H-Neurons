# Readout Surfaces

## What This Surface Measures

These surfaces track whether the H-neuron and SAE readouts are real detectors, how sparse they are, and where the main classification errors land. Primary artifacts in this family: `classifier_disjoint_summary`, `classifier_overlap_summary`, `classifier_sae_summary`, `classifier_structure_summary`, `pipeline_answer_tokens`, `pipeline_consistency_samples`, `pipeline_summary_site`.

## High-Signal Patterns

- H-neuron detector quality is real on the disjoint split: AUROC 84.3%, accuracy 76.5% [`readout.disjoint.auroc`, `readout.disjoint.accuracy`, `classifier_disjoint_summary`].
- Localization is sparse rather than diffuse: 38 selected neurons out of 348160, with layer counts skewed early (18) relative to middle/late (10, 10) [`readout.pipeline.selected_h_neurons`, `readout.pipeline.total_ffn_neurons`, `readout.structure.band.early`, `readout.structure.band.middle`, `readout.structure.band.late`, `pipeline_summary_site`, `classifier_disjoint_summary`, `classifier_structure_summary`].
- SAE readout parity is close on detection quality but much less sparse: SAE AUROC 84.8% on 782 examples, with 266 positive features across 10 layers [`readout.sae.auroc`, `readout.sae.test_size`, `readout.sae.n_positive_features`, `readout.sae.layer_count`, `classifier_sae_summary`].
- False positives and false negatives are nearly balanced on the disjoint split (90 FP vs 93 FN), so the dominant issue is not a one-sided collapse but boundary cases around answer normalization and consistency support [`readout.disjoint.confusion.fp`, `readout.disjoint.confusion.fn`, `classifier_disjoint_summary`].

## Measurement / Interpretation Risks

- The overlap split is slightly easier than the disjoint split, so readout quality should be read off the disjoint numbers first [`readout.overlap.accuracy`, `readout.disjoint.accuracy`, `readout.overlap.auroc`, `readout.disjoint.auroc`, `classifier_overlap_summary`, `classifier_disjoint_summary`].
- Readout quality does not transport to control utility by itself; the pack keeps detector metrics and intervention metrics separate on purpose [`readout.disjoint.auroc`, `intervention.faitheval.anti.slope_pp_per_alpha`, `intervention.faitheval_sae.h_slope_pp_per_alpha`, `classifier_disjoint_summary`, `intervention_sweep_site`, `faitheval_sae_control_comparison`].

## Grouped Metric Tables

### Detector Quality

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `readout.disjoint.accuracy` | 76.5% | 780 | `[73.6, 79.5]%` | `classifier_disjoint_summary` |
| `readout.disjoint.auroc` | 84.3% | 780 | `[81.5, 87.0]%` | `classifier_disjoint_summary` |
| `readout.disjoint.confusion.fn` | 93 | 780 | `—` | `classifier_disjoint_summary` |
| `readout.disjoint.confusion.fp` | 90 | 780 | `—` | `classifier_disjoint_summary` |
| `readout.disjoint.confusion.tn` | 301 | 780 | `—` | `classifier_disjoint_summary` |
| `readout.disjoint.confusion.tp` | 296 | 780 | `—` | `classifier_disjoint_summary` |
| `readout.disjoint.f1` | 76.4% | 780 | `[73.2, 79.4]%` | `classifier_disjoint_summary` |
| `readout.disjoint.precision` | 76.7% | 780 | `[73.4, 80.1]%` | `classifier_disjoint_summary` |
| `readout.disjoint.recall` | 76.1% | 780 | `[71.7, 80.5]%` | `classifier_disjoint_summary` |
| `readout.disjoint.test_size` | 780 | 780 | `—` | `classifier_disjoint_summary` |
| `readout.overlap.accuracy` | 77.7% | 1993 | `[75.9, 79.5]%` | `classifier_overlap_summary` |
| `readout.overlap.auroc` | 86.3% | 1993 | `[84.7, 87.9]%` | `classifier_overlap_summary` |

### SAE Readout

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `readout.sae.accuracy` | 77.2% | 782 | `—` | `classifier_sae_summary` |
| `readout.sae.auroc` | 84.8% | 782 | `—` | `classifier_sae_summary` |
| `readout.sae.layer_count` | 10 | — | `—` | `classifier_sae_summary` |
| `readout.sae.n_nonzero_features` | 509 | — | `—` | `classifier_sae_summary` |
| `readout.sae.n_positive_features` | 266 | — | `—` | `classifier_sae_summary` |
| `readout.sae.test_size` | 782 | 782 | `—` | `classifier_sae_summary` |

### Localization / Sparsity

| metric_id | estimate | n | ci | source_ids |
|---|---:|---:|---|---|
| `readout.pipeline.consistent_total` | 3115 | — | `—` | `pipeline_summary_site` |
| `readout.pipeline.disjoint_evaluated_total` | 780 | — | `—` | `pipeline_summary_site` |
| `readout.pipeline.extracted_answer_tokens` | 2997 | — | `—` | `pipeline_summary_site` |
| `readout.pipeline.sampled_questions` | 3500 | — | `—` | `pipeline_summary_site` |
| `readout.pipeline.selected_h_neurons` | 38 | — | `—` | `pipeline_summary_site, classifier_disjoint_summary` |
| `readout.pipeline.total_ffn_neurons` | 348160 | — | `—` | `pipeline_summary_site, classifier_structure_summary` |
| `readout.structure.band.early` | 18 | — | `—` | `classifier_structure_summary` |
| `readout.structure.band.late` | 10 | — | `—` | `classifier_structure_summary` |
| `readout.structure.band.middle` | 10 | — | `—` | `classifier_structure_summary` |

## Representative Examples

### fn #1

Trace: `example.readout.fn.1`.
Source ids: `classifier_disjoint_summary, pipeline_answer_tokens, pipeline_consistency_samples`.
Label summary: classifier_prediction=0; gold_label=1; answer_token_judge=false. target=bad (+9 aliases).
Question summary (118 chars): Man In The Mirror first featured on which Michel Jackson album? Respond with the answer only, without any explanation.
Response summary (8 chars): Thriller
Why selected: quantiles_25_75_of_classifier_probability_with_answer_and_consistency_support; score=0.20365914056198825; tiebreak=tc_81.

### fn #2

Trace: `example.readout.fn.2`.
Source ids: `classifier_disjoint_summary, pipeline_answer_tokens, pipeline_consistency_samples`.
Label summary: classifier_prediction=0; gold_label=1; answer_token_judge=false. target=Boxers (+86 aliases).
Question summary (106 chars): What is the most popular sport to feature in films? Respond with the answer only, without any explanation.
Response summary (19 chars): Football (American)
Why selected: quantiles_25_75_of_classifier_probability_with_answer_and_consistency_support; score=0.3823446179003473; tiebreak=qz_437.

### fp #1

Trace: `example.readout.fp.1`.
Source ids: `classifier_disjoint_summary, pipeline_answer_tokens, pipeline_consistency_samples`.
Label summary: classifier_prediction=1; gold_label=0; answer_token_judge=true. target=Nancy sinatra (+8 aliases).
Question summary (117 chars): Who sang the title song for the Bond film You Only Live Twice? Respond with the answer only, without any explanation.
Response summary (13 chars): Nancy Sinatra
Why selected: quantiles_25_75_of_classifier_probability_with_answer_and_consistency_support; score=0.5524287973382225; tiebreak=tc_522.

### fp #2

Trace: `example.readout.fp.2`.
Source ids: `classifier_disjoint_summary, pipeline_answer_tokens, pipeline_consistency_samples`.
Label summary: classifier_prediction=1; gold_label=0; answer_token_judge=true. target=Newton Heath, Greater Manchester (+3 aliases).
Question summary (112 chars): What was Manchester United football club's original name? Respond with the answer only, without any explanation.
Response summary (16 chars): Newton Heath LYR
Why selected: quantiles_25_75_of_classifier_probability_with_answer_and_consistency_support; score=0.6763307883191508; tiebreak=qz_909.

### tn #1

Trace: `example.readout.tn.1`.
Source ids: `classifier_disjoint_summary, pipeline_answer_tokens, pipeline_consistency_samples`.
Label summary: classifier_prediction=0; gold_label=0; answer_token_judge=true. target=Gunnar Gaasland (+11 aliases).
Question summary (118 chars): Who was West German Chancellor from 1969 to 1974? Willy Brandt. Respond with the answer only, without any explanation.
Response summary (12 chars): Willy Brandt
Why selected: quantiles_25_75_of_classifier_probability_with_answer_and_consistency_support; score=0.12642180516318072; tiebreak=tc_1091.

### tn #2

Trace: `example.readout.tn.2`.
Source ids: `classifier_disjoint_summary, pipeline_answer_tokens, pipeline_consistency_samples`.
Label summary: classifier_prediction=0; gold_label=0; answer_token_judge=true. target=fathership (+26 aliases).
Question summary (115 chars): What relation is author Kingsley Amis to author Martin Amis? Respond with the answer only, without any explanation.
Response summary (14 chars): Father and son
Why selected: quantiles_25_75_of_classifier_probability_with_answer_and_consistency_support; score=0.3456628994564059; tiebreak=qz_830.

### tp #1

Trace: `example.readout.tp.1`.
Source ids: `classifier_disjoint_summary, pipeline_answer_tokens, pipeline_consistency_samples`.
Label summary: classifier_prediction=1; gold_label=1; answer_token_judge=false. target=franz hals (+3 aliases).
Question summary (89 chars): Who painted The Laughing Cavalier? Respond with the answer only, without any explanation.
Response summary (15 chars): Nicolas Poussin
Why selected: quantiles_25_75_of_classifier_probability_with_answer_and_consistency_support; score=0.6591172719441899; tiebreak=qz_1297.

### tp #2

Trace: `example.readout.tp.2`.
Source ids: `classifier_disjoint_summary, pipeline_answer_tokens, pipeline_consistency_samples`.
Label summary: classifier_prediction=1; gold_label=1; answer_token_judge=false. target=rheum rhabarbarum (+15 aliases).
Question summary (164 chars): Which fruit shares a name with a 1969 short film by Eric Sykes, and an animated television show for children? Respond with the answer only, without any explanation.
Response summary (4 chars): Plum
Why selected: quantiles_25_75_of_classifier_probability_with_answer_and_consistency_support; score=0.8437796597939967; tiebreak=qz_2043.
