# Readout Surfaces

Family id: `readout_surfaces`
Act tags: `act1, act2`
Template: neutral structured surface summary.

## Sources

| source_id | source_path | role | status |
|---|---|---|---|
| `classifier_disjoint_summary` | `data/gemma3_4b/pipeline/classifier_disjoint_summary.json` | `primary_readout` | `current` |
| `classifier_overlap_summary` | `data/gemma3_4b/pipeline/classifier_overlap_summary.json` | `overlap_readout` | `current` |
| `classifier_sae_summary` | `data/gemma3_4b/pipeline/classifier_sae_summary.json` | `sae_readout` | `current` |
| `classifier_structure_summary` | `data/gemma3_4b/pipeline/classifier_structure_summary.json` | `readout_structure` | `current` |
| `pipeline_answer_tokens` | `data/gemma3_4b/pipeline/answer_tokens.jsonl` | `example_pool` | `current` |
| `pipeline_consistency_samples` | `data/gemma3_4b/pipeline/consistency_samples.jsonl` | `example_pool` | `current` |
| `pipeline_summary_site` | `site/data/pipeline_summary.json` | `pipeline_counts` | `current` |

## Metrics

| metric_id | benchmark | metric_name | condition_a | condition_b | estimate | unit | n | ci | source_ids |
|---|---|---|---|---|---:|---|---:|---|---|
| `readout.disjoint.accuracy` | `pipeline` | `accuracy` | `disjoint_test` | `disjoint_test` | 0.7653846153846153 | `proportion` | 780 | `[0.735897435897436, 0.7948717948717948]` | `classifier_disjoint_summary` |
| `readout.disjoint.auroc` | `pipeline` | `auroc` | `disjoint_test` | `disjoint_test` | 0.842904950065418 | `proportion` | 780 | `[0.8148766592811261, 0.8695129488030822]` | `classifier_disjoint_summary` |
| `readout.disjoint.confusion.fn` | `pipeline` | `confusion_fn` | `disjoint_test` | `disjoint_test` | 93 | `count` | 780 | `—` | `classifier_disjoint_summary` |
| `readout.disjoint.confusion.fp` | `pipeline` | `confusion_fp` | `disjoint_test` | `disjoint_test` | 90 | `count` | 780 | `—` | `classifier_disjoint_summary` |
| `readout.disjoint.confusion.tn` | `pipeline` | `confusion_tn` | `disjoint_test` | `disjoint_test` | 301 | `count` | 780 | `—` | `classifier_disjoint_summary` |
| `readout.disjoint.confusion.tp` | `pipeline` | `confusion_tp` | `disjoint_test` | `disjoint_test` | 296 | `count` | 780 | `—` | `classifier_disjoint_summary` |
| `readout.disjoint.f1` | `pipeline` | `f1` | `disjoint_test` | `disjoint_test` | 0.7638709677419355 | `proportion` | 780 | `[0.7319587628865979, 0.794344473007712]` | `classifier_disjoint_summary` |
| `readout.disjoint.precision` | `pipeline` | `precision` | `disjoint_test` | `disjoint_test` | 0.7668393782383419 | `proportion` | 780 | `[0.733502538071066, 0.8005653551218626]` | `classifier_disjoint_summary` |
| `readout.disjoint.recall` | `pipeline` | `recall` | `disjoint_test` | `disjoint_test` | 0.7609254498714653 | `proportion` | 780 | `[0.7172236503856041, 0.8046272493573264]` | `classifier_disjoint_summary` |
| `readout.disjoint.test_size` | `pipeline` | `test_size` | `disjoint_test` | `disjoint_test` | 780 | `count` | 780 | `—` | `classifier_disjoint_summary` |
| `readout.overlap.accuracy` | `pipeline` | `accuracy` | `overlap_test` | `overlap_test` | 0.7767185148018063 | `proportion` | 1993 | `[0.7586552935273457, 0.7947817360762669]` | `classifier_overlap_summary` |
| `readout.overlap.auroc` | `pipeline` | `auroc` | `overlap_test` | `overlap_test` | 0.8632678751258813 | `proportion` | 1993 | `[0.8474337613293053, 0.8786488167170191]` | `classifier_overlap_summary` |
| `readout.pipeline.consistent_total` | `pipeline` | `consistent_total` | `pipeline` | `pipeline` | 3115 | `count` | — | `—` | `pipeline_summary_site` |
| `readout.pipeline.disjoint_evaluated_total` | `pipeline` | `disjoint_evaluated_total` | `pipeline` | `pipeline` | 780 | `count` | — | `—` | `pipeline_summary_site` |
| `readout.pipeline.extracted_answer_tokens` | `pipeline` | `extracted_answer_tokens` | `pipeline` | `pipeline` | 2997 | `count` | — | `—` | `pipeline_summary_site` |
| `readout.pipeline.sampled_questions` | `pipeline` | `sampled_questions` | `pipeline` | `pipeline` | 3500 | `count` | — | `—` | `pipeline_summary_site` |
| `readout.pipeline.selected_h_neurons` | `pipeline` | `selected_h_neurons` | `pipeline` | `pipeline` | 38 | `count` | — | `—` | `pipeline_summary_site, classifier_disjoint_summary` |
| `readout.pipeline.total_ffn_neurons` | `pipeline` | `total_ffn_neurons` | `pipeline` | `pipeline` | 348160 | `count` | — | `—` | `pipeline_summary_site, classifier_structure_summary` |
| `readout.sae.accuracy` | `pipeline` | `accuracy` | `sae_test` | `sae_test` | 0.7723785166240409 | `proportion` | 782 | `—` | `classifier_sae_summary` |
| `readout.sae.auroc` | `pipeline` | `auroc` | `sae_test` | `sae_test` | 0.8476527495241397 | `proportion` | 782 | `—` | `classifier_sae_summary` |
| `readout.sae.layer_count` | `pipeline` | `layer_count` | `sae_selected` | `sae_selected` | 10 | `count` | — | `—` | `classifier_sae_summary` |
| `readout.sae.n_nonzero_features` | `pipeline` | `n_nonzero_features` | `sae_selected` | `sae_selected` | 509 | `count` | — | `—` | `classifier_sae_summary` |
| `readout.sae.n_positive_features` | `pipeline` | `n_positive_features` | `sae_selected` | `sae_selected` | 266 | `count` | — | `—` | `classifier_sae_summary` |
| `readout.sae.test_size` | `pipeline` | `test_size` | `sae_test` | `sae_test` | 782 | `count` | 782 | `—` | `classifier_sae_summary` |
| `readout.structure.band.early` | `pipeline` | `early_count` | `layers_0_10` | `layers_0_10` | 18 | `count` | — | `—` | `classifier_structure_summary` |
| `readout.structure.band.late` | `pipeline` | `late_count` | `layers_21_33` | `layers_21_33` | 10 | `count` | — | `—` | `classifier_structure_summary` |
| `readout.structure.band.middle` | `pipeline` | `middle_count` | `layers_11_20` | `layers_11_20` | 10 | `count` | — | `—` | `classifier_structure_summary` |

## Examples

### `example.readout.fn.1`

Sample id: `tc_81`
Selection stratum: `fn`
Selection rank: `1`
Primary source id: `classifier_disjoint_summary`
Source ids: `classifier_disjoint_summary, pipeline_answer_tokens, pipeline_consistency_samples`

Question
````text
Man In The Mirror first featured on which Michel Jackson album? Respond with the answer only, without any explanation.
````

Response
````text
Thriller
````

Condition metadata
````json
{
  "label": 1,
  "prediction": 0,
  "probability": 0.20365914056198825,
  "selection_rule": "quantiles_25_75_of_classifier_probability_with_answer_and_consistency_support",
  "selection_score": 0.20365914056198825,
  "selection_tiebreak": "tc_81"
}
````

Raw label fields
````json
{
  "answer_token_judge": "false",
  "consistency_judges": [
    "false",
    "false",
    "false",
    "false",
    "false",
    "false",
    "false",
    "false",
    "false",
    "false"
  ],
  "ground_truth": [
    "bad",
    "badness",
    "Bad",
    "bad song",
    "Bad (song)",
    "Bad (disambiguation)",
    "BAD",
    "BAD (disambiguation)",
    "Badness",
    "bad disambiguation"
  ]
}
````

Judge fields
````json
{
  "classifier_correct": false,
  "classifier_label": 1,
  "classifier_prediction": 0
}
````

Paired example ids: ``

### `example.readout.fn.2`

Sample id: `qz_437`
Selection stratum: `fn`
Selection rank: `2`
Primary source id: `classifier_disjoint_summary`
Source ids: `classifier_disjoint_summary, pipeline_answer_tokens, pipeline_consistency_samples`

Question
````text
What is the most popular sport to feature in films? Respond with the answer only, without any explanation.
````

Response
````text
Football (American)
````

Condition metadata
````json
{
  "label": 1,
  "prediction": 0,
  "probability": 0.3823446179003473,
  "selection_rule": "quantiles_25_75_of_classifier_probability_with_answer_and_consistency_support",
  "selection_score": 0.3823446179003473,
  "selection_tiebreak": "qz_437"
}
````

Raw label fields
````json
{
  "answer_token_judge": "false",
  "consistency_judges": [
    "false",
    "false",
    "false",
    "false",
    "false",
    "false",
    "false",
    "false",
    "false",
    "false"
  ],
  "ground_truth": [
    "Boxers",
    "corner person",
    "parrying boxing",
    "\u62f3\u95d8",
    "prizefighter",
    "ring second",
    "Boxing match",
    "boxer",
    "Prize Fighter",
    "Western boxing",
    "Pugilism",
    "Fist-fighting",
    "History of boxing",
    "Noble art",
    "history of boxing",
    "Fistfight",
    "cornerpersons",
    "prize fighter",
    "Boxing moves",
    "pugilistic",
    "Fist fight",
    "boxing match",
    "fist fight",
    "Gentleman's sport",
    "boxing styles and techniques",
    "Prizefighter",
    "sports boxing",
    "Cornerpersons",
    "Prize fighter",
    "cornerperson",
    "Low guard",
    "Corner-man",
    "cornermen",
    "Boxer",
    "fist fighting",
    "boxing sport",
    "Boxing Styles and Techniques",
    "Hit and Away",
    "pugilist",
    "fistfighting",
    "Corner men",
    "Parrying (boxing)",
    "Fist Fighting",
    "boxing punches",
    "round boxing",
    "History of professional boxing",
    "corner man",
    "Boxing",
    "Boxing punches",
    "Corner-person",
    "boxers",
    "noble art",
    "Prize fight",
    "western boxing",
    "prizefights",
    "gentleman s sport",
    "Corner person",
    "boxing moves",
    "low guard",
    "prize fight",
    "english boxing",
    "boxing",
    "history of professional boxing",
    "Ring second",
    "Pugilist",
    "corner persons",
    "Round (boxing)",
    "Boxing (sport)",
    "prize fighting",
    "Sports boxing",
    "Cornerperson",
    "Pugilistic",
    "Prize-fighter",
    "fistfight",
    "Fist-fight",
    "corner men",
    "Cornermen",
    "pugilism",
    "Fist fighting",
    "hit and away",
    "Prizefights",
    "Corner persons",
    "English boxing",
    "Corner-persons",
    "Corner-men",
    "Fistfighting",
    "Prize fighting"
  ]
}
````

Judge fields
````json
{
  "classifier_correct": false,
  "classifier_label": 1,
  "classifier_prediction": 0
}
````

Paired example ids: ``

### `example.readout.fp.1`

Sample id: `tc_522`
Selection stratum: `fp`
Selection rank: `1`
Primary source id: `classifier_disjoint_summary`
Source ids: `classifier_disjoint_summary, pipeline_answer_tokens, pipeline_consistency_samples`

Question
````text
Who sang the title song for the Bond film You Only Live Twice? Respond with the answer only, without any explanation.
````

Response
````text
Nancy Sinatra
````

Condition metadata
````json
{
  "label": 0,
  "prediction": 1,
  "probability": 0.5524287973382225,
  "selection_rule": "quantiles_25_75_of_classifier_probability_with_answer_and_consistency_support",
  "selection_score": 0.5524287973382225,
  "selection_tiebreak": "tc_522"
}
````

Raw label fields
````json
{
  "answer_token_judge": "true",
  "consistency_judges": [
    "true",
    "true",
    "true",
    "true",
    "true",
    "true",
    "true",
    "true",
    "true",
    "true"
  ],
  "ground_truth": [
    "Nancy sinatra",
    "Country, My Way",
    "Nancy Sinatra",
    "for my dad",
    "how does it feel album",
    "How Does It Feel (album)",
    "country my way",
    "For My Dad",
    "nancy sinatra"
  ]
}
````

Judge fields
````json
{
  "classifier_correct": false,
  "classifier_label": 0,
  "classifier_prediction": 1
}
````

Paired example ids: ``

### `example.readout.fp.2`

Sample id: `qz_909`
Selection stratum: `fp`
Selection rank: `2`
Primary source id: `classifier_disjoint_summary`
Source ids: `classifier_disjoint_summary, pipeline_answer_tokens, pipeline_consistency_samples`

Question
````text
What was Manchester United football club's original name? Respond with the answer only, without any explanation.
````

Response
````text
Newton Heath LYR
````

Condition metadata
````json
{
  "label": 0,
  "prediction": 1,
  "probability": 0.6763307883191508,
  "selection_rule": "quantiles_25_75_of_classifier_probability_with_answer_and_consistency_support",
  "selection_score": 0.6763307883191508,
  "selection_tiebreak": "qz_909"
}
````

Raw label fields
````json
{
  "answer_token_judge": "true",
  "consistency_judges": [
    "true",
    "true",
    "true",
    "true",
    "true",
    "true",
    "true",
    "true",
    "true",
    "true"
  ],
  "ground_truth": [
    "Newton Heath, Greater Manchester",
    "newton heath",
    "newton heath greater manchester",
    "Newton Heath"
  ]
}
````

Judge fields
````json
{
  "classifier_correct": false,
  "classifier_label": 0,
  "classifier_prediction": 1
}
````

Paired example ids: ``

### `example.readout.tn.1`

Sample id: `tc_1091`
Selection stratum: `tn`
Selection rank: `1`
Primary source id: `classifier_disjoint_summary`
Source ids: `classifier_disjoint_summary, pipeline_answer_tokens, pipeline_consistency_samples`

Question
````text
Who was West German Chancellor from 1969 to 1974? Willy Brandt. Respond with the answer only, without any explanation.
````

Response
````text
Willy Brandt
````

Condition metadata
````json
{
  "label": 0,
  "prediction": 0,
  "probability": 0.12642180516318072,
  "selection_rule": "quantiles_25_75_of_classifier_probability_with_answer_and_consistency_support",
  "selection_score": 0.12642180516318072,
  "selection_tiebreak": "tc_1091"
}
````

Raw label fields
````json
{
  "answer_token_judge": "true",
  "consistency_judges": [
    "true",
    "true",
    "true",
    "true",
    "true",
    "true",
    "true",
    "true",
    "true",
    "true"
  ],
  "ground_truth": [
    "Gunnar Gaasland",
    "Karl Herbert Frahm",
    "gunnar gaasland",
    "willy brandt",
    "Herbert Ernst Karl Frahm",
    "herbert frahm",
    "Herbert Frahm",
    "karl herbert frahm",
    "Herbert Ernst Carl Frahm",
    "herbert ernst karl frahm",
    "herbert ernst carl frahm",
    "Willy Brandt"
  ]
}
````

Judge fields
````json
{
  "classifier_correct": true,
  "classifier_label": 0,
  "classifier_prediction": 0
}
````

Paired example ids: ``

### `example.readout.tn.2`

Sample id: `qz_830`
Selection stratum: `tn`
Selection rank: `2`
Primary source id: `classifier_disjoint_summary`
Source ids: `classifier_disjoint_summary, pipeline_answer_tokens, pipeline_consistency_samples`

Question
````text
What relation is author Kingsley Amis to author Martin Amis? Respond with the answer only, without any explanation.
````

Response
````text
Father and son
````

Condition metadata
````json
{
  "label": 0,
  "prediction": 0,
  "probability": 0.3456628994564059,
  "selection_rule": "quantiles_25_75_of_classifier_probability_with_answer_and_consistency_support",
  "selection_score": 0.3456628994564059,
  "selection_tiebreak": "qz_830"
}
````

Raw label fields
````json
{
  "answer_token_judge": "true",
  "consistency_judges": [
    "true",
    "true",
    "true",
    "true",
    "true",
    "true",
    "true",
    "true",
    "true",
    "true"
  ],
  "ground_truth": [
    "fathership",
    "fatherhood",
    "Paternity (Biology and Medicine)",
    "Non-human fatherhood",
    "dad",
    "Fathers",
    "Father",
    "non human fatherhood",
    "Daddie",
    "Biological father",
    "dads",
    "fathers",
    "biological father",
    "Fathership",
    "Fatherhood",
    "fatherhoods",
    "daddie",
    "Fatherhoods",
    "Natural father",
    "Dad",
    "natural father",
    "fathering",
    "Fathering",
    "Dads",
    "father",
    "paternity biology and medicine",
    "Paternity (biology and medicine)"
  ]
}
````

Judge fields
````json
{
  "classifier_correct": true,
  "classifier_label": 0,
  "classifier_prediction": 0
}
````

Paired example ids: ``

### `example.readout.tp.1`

Sample id: `qz_1297`
Selection stratum: `tp`
Selection rank: `1`
Primary source id: `classifier_disjoint_summary`
Source ids: `classifier_disjoint_summary, pipeline_answer_tokens, pipeline_consistency_samples`

Question
````text
Who painted The Laughing Cavalier? Respond with the answer only, without any explanation.
````

Response
````text
Nicolas Poussin
````

Condition metadata
````json
{
  "label": 1,
  "prediction": 1,
  "probability": 0.6591172719441899,
  "selection_rule": "quantiles_25_75_of_classifier_probability_with_answer_and_consistency_support",
  "selection_score": 0.6591172719441899,
  "selection_tiebreak": "qz_1297"
}
````

Raw label fields
````json
{
  "answer_token_judge": "false",
  "consistency_judges": [
    "false",
    "false",
    "false",
    "false",
    "false",
    "false",
    "false",
    "false",
    "false",
    "false"
  ],
  "ground_truth": [
    "franz hals",
    "Frans Hals",
    "frans hals",
    "Franz Hals"
  ]
}
````

Judge fields
````json
{
  "classifier_correct": true,
  "classifier_label": 1,
  "classifier_prediction": 1
}
````

Paired example ids: ``

### `example.readout.tp.2`

Sample id: `qz_2043`
Selection stratum: `tp`
Selection rank: `2`
Primary source id: `classifier_disjoint_summary`
Source ids: `classifier_disjoint_summary, pipeline_answer_tokens, pipeline_consistency_samples`

Question
````text
Which fruit shares a name with a 1969 short film by Eric Sykes, and an animated television show for children? Respond with the answer only, without any explanation.
````

Response
````text
Plum
````

Condition metadata
````json
{
  "label": 1,
  "prediction": 1,
  "probability": 0.8437796597939967,
  "selection_rule": "quantiles_25_75_of_classifier_probability_with_answer_and_consistency_support",
  "selection_score": 0.8437796597939967,
  "selection_tiebreak": "qz_2043"
}
````

Raw label fields
````json
{
  "answer_token_judge": "false",
  "consistency_judges": [
    "false",
    "false",
    "false",
    "false",
    "false",
    "false",
    "false",
    "false",
    "false",
    "false"
  ],
  "ground_truth": [
    "rheum rhabarbarum",
    "Rubarb",
    "medicinal rhubarb",
    "pieplant",
    "Rhubarb",
    "Pie plant",
    "pie plant",
    "Pieplant",
    "rhubarb",
    "Garden Rhubarb",
    "garden rhubarb",
    "Rheum rhabarbarum",
    "Rheum x hybridum",
    "rubarb",
    "Medicinal Rhubarb",
    "rheum x hybridum"
  ]
}
````

Judge fields
````json
{
  "classifier_correct": true,
  "classifier_label": 1,
  "classifier_prediction": 1
}
````

Paired example ids: ``
