# Evaluation Manifest

This manifest records the reviewer-facing evaluation surfaces used in the paper’s measurement claims. It is intentionally limited to prompt/rubric descriptions, judge models, bundled safe artifacts, and script entrypoints.

## Bundled Evaluation Artifacts

| Artifact | Purpose |
|---|---|
| [`support/judge_prompts.md`](support/judge_prompts.md) | Prompt and rubric summary for CSV2 v3 and StrongREJECT. |
| [`support/measurement_summary.md`](support/measurement_summary.md) | Derived summary of the seed-0 jailbreak control and holdout evaluator audit. |
| [`data/judge_validation/holdout_comparison.json`](data/judge_validation/holdout_comparison.json) | Canonical safe machine-readable output for the four-way holdout comparison. |

## Judge Surfaces

### CSV2 v3

| Field | Value |
|---|---|
| Primary script entrypoint | `scripts/evaluate_csv2.py` |
| Schema version | `csv2_v3` |
| Reported judge model | `gpt-4o` for the canonical v3 rerun and holdout comparison |
| Output fields | `primary_outcome`, `intent_match`, `harmful_binary`, `C`, `S`, `V`, `R`, `D`, `P`, `F`, `T`, localized evidence spans |
| Rubric summary | Structured harmfulness measurement with explicit refusal/deflection/compliance outcomes and typed evidence spans |
| Prompt summary | Bundled in [`support/judge_prompts.md`](support/judge_prompts.md) |

### StrongREJECT

| Field | Value |
|---|---|
| Primary script entrypoint | `scripts/evaluate_strongreject.py` |
| Schema version | `strongreject_v1` |
| Reported judge model | `gpt-4o` for the canonical rerun referenced by the holdout comparison |
| Output fields | `refused`, `convincingness`, `specificity`, derived `score`, derived `harmful_binary` |
| Rubric summary | Verbatim StrongREJECT safety judge prompt structure with refusal, convincingness, and specificity scoring |
| Prompt summary | Bundled in [`support/judge_prompts.md`](support/judge_prompts.md) |

### Binary judge and deterministic bridge grading

| Field | Value |
|---|---|
| Primary script entrypoint | `scripts/evaluate_intervention.py` |
| Surfaces covered | Jailbreak binary judge, SimpleQA batch judge, TriviaQA bridge deterministic + adjudicated grading |
| Paper-facing use here | Binary judge is part of the four-way evaluator audit; bridge grading is part of the §4.3 externality analysis |

## Holdout and Contamination Boundary

| Item | Value |
|---|---|
| Holdout artifact | [`data/judge_validation/holdout_comparison.json`](data/judge_validation/holdout_comparison.json) |
| Analysis script | `scripts/analysis_holdout_evaluator.py` |
| Gold fixture used in-repo | `tests/gold_labels/jailbreak_cross_alpha_gold.jsonl` |
| Bundling policy for gold fixture | Not bundled in this supplement because it contains harmful prompts and full response text |
| Contamination rule | Exclude the 8 prompt IDs used during CSV2 v3 calibration, leaving a 50-record contamination-clean holdout |

## Reported Models and Versions

| Surface | Model used for reported result |
|---|---|
| CSV2 v3 evaluator rerun / holdout | `gpt-4o` |
| StrongREJECT rerun / holdout | `gpt-4o` |
| Binary judge comparison | `gpt-4o` via the project evaluation stack |

## Non-Bundled Implementation Details

- Exact API wrappers, batching logic, and few-shot calibration examples live in the repo entrypoints listed above.
- Raw scored JSONL outputs and raw run provenance sidecars are intentionally omitted from this supplement for anonymization and compactness.
