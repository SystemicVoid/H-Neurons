# Prospective SIMID Effect-Run Gate

Status: frozen prospective protocol package.
Protocol version: `simid_prospective_effect_run_protocol_20260429/v1`.

This package pre-specifies the next SIMID effect run. It is not an execution
result and it is not a retrospective cleanup pass. Historical MVP open metrics
remain diagnostic-only.

## Frozen Open-Grading Authority

Future open grading is bound to the passing Opus prospective calibration run:

- Analysis: `data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/mvp_20260427_calibration/human_review_package/prospective_open_calibration_gate_20260429/prospective_open_calibration_analysis_opus_4_7_max_run_001.json`
- Analysis SHA-256: `976367fd64d38471d42732c8dc878ef17c0f102ddae2b0dde2b3c43fa90ae180`
- Rubric: `data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/mvp_20260427_calibration/human_review_package/prospective_open_calibration_gate_20260429/rubric.md`
- Rubric SHA-256: `80f9baf09d757a9296c931b6973e633571974411e3ec421e6b512dba53df6f6f`
- Labels: `data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/mvp_20260427_calibration/human_review_package/prospective_open_calibration_gate_20260429/prospective_open_labels_opus_4_7_max_run_001.jsonl`
- Label SHA-256: `017866f8ea998a36a344e49853873d1ea18f75dccc970f414c1d80dbcd6d99ae`

The authority scope is `future_simid_open_grading_under_frozen_rubric_only`. It does not upgrade the
historical MVP run or justify endpoint tuning on old rows.

## Planned Effect Run

- Selected ITI condition: `truthfulqa_paperfaithful`,
  k=12,
  decode scope `first_3_tokens`.
- Manifest: `data/manifests/simid_truthfulqa_bridge_prospective_effect_20260429.json`
- Effect run directory: `data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/prospective_effect_calibrated_open_20260429`
- No-op preflight directory: `data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/prospective_effect_noop_preflight_20260429`
- Alpha grid: `[-8.0, 0.0, 4.0, 8.0]`
- Primary alpha: `8.0`
- Baseline alpha: `0.0`

Controls are frozen as unhooked/no-op preflight, five random-direction seeds,
and five layer-matched random-head seeds. A future stronger mechanism claim can
add more controls, but it must not remove these controls after seeing outcomes.

## Primary Estimand

`truthfulqa_paired_open_correctness_delta_alpha8_vs_alpha0`:
paired TruthfulQA open-correctness delta for selected alpha=8 versus alpha=0,
graded under the frozen prospective rubric and summarized with a paired
base-item bootstrap 95% interval. The minimum practical effect is +5 pp.

## Secondary Endpoints

- MC letter-likelihood accuracy.
- Open attempted/non-attempted behavior.
- Open first-3-token margin diagnostics.
- TriviaQA Bridge open-correctness externality/specificity check.

## Claimability Gates

The effect run remains non-claimable if any blocker is present:

- `open_adjudication_not_complete_for_all_effect_rows`
- `mixed_adjudication_and_deterministic_open_grade_sources`
- `unknown_or_error_open_judge_verdicts`
- `prospective_open_authority_not_recorded_in_results`
- `prospective_effect_manifest_reuses_historical_mvp_or_calibration_rows`
- `control_pairing_or_replicate_pairing_failure`

To pass the prospective effect gate, the future analysis must satisfy:

- `primary_truthfulqa_open_delta_positive`: selected alpha=8 minus selected alpha=0 paired adjudicated-open-correctness delta has lower 95% CI > 0 and point estimate >= +5 pp
- `selected_exceeds_random_direction`: selected primary open delta exceeds every random-direction seed's primary delta and exceeds the random-direction seed-family mean delta
- `selected_exceeds_random_head`: selected primary open delta exceeds every random-head seed's primary delta and exceeds the random-head seed-family mean delta
- `mc_and_attempt_behavior_reported`: MC accuracy and open-attempt/non-answer behavior are reported with paired uncertainty; harmful MC degradation or pure attempt-rate shifts block a truthfulness-improvement claim

Even if all gates pass, the claim scope is limited to the specified fresh
prospective same-item SIMID run. It does not imply historical MVP claimability
or a broad truthfulness-improvement claim.

## Commands

Build the locked manifest:

```bash
uv run python scripts/build_simid_manifest.py --seed 42 --truthfulqa-leakage-policy heldout_only --option-order-replicates 2 --min-truthfulqa-rows 450 --min-bridge-rows 400 --exclude-sample-ids-jsonl data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/mvp_20260427_calibration/human_review_package/prospective_effect_run_gate_20260429/excluded_effect_sample_ids.jsonl --model-path google/gemma-3-4b-it --iti-artifact-path data/contrastive/truthfulness/iti_truthfulqa_paperfaithful/final_fold0/iti_heads.pt --bridge-n 200 --output data/manifests/simid_truthfulqa_bridge_prospective_effect_20260429.json
```

Run the no-op preflight:

```bash
uv run python scripts/run_simid.py --manifest data/manifests/simid_truthfulqa_bridge_prospective_effect_20260429.json --output-dir data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/prospective_effect_noop_preflight_20260429 --model-path google/gemma-3-4b-it --iti-artifact-path data/contrastive/truthfulness/iti_truthfulqa_paperfaithful/final_fold0/iti_heads.pt --iti-family truthfulqa_paperfaithful --iti-k 12 --decode-scope first_3_tokens --device-map cuda:0 --alphas 0.0 --conditions selected --include-unhooked --noop-check --top-k-first-token 10 --mc-max-new-tokens 4 --open-max-new-tokens 64
```

Analyze the no-op preflight:

```bash
uv run python scripts/analyze_simid.py --run-dir data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/prospective_effect_noop_preflight_20260429 --conditions unhooked selected --alphas 0.0 --phase0-gates --n-resamples 2000 --output-json data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/prospective_effect_noop_preflight_20260429/results_noop_preflight.json --report-md data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/prospective_effect_noop_preflight_20260429/report_noop_preflight.md
```

Run the effect grid:

```bash
uv run python scripts/run_simid.py --manifest data/manifests/simid_truthfulqa_bridge_prospective_effect_20260429.json --output-dir data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/prospective_effect_calibrated_open_20260429 --model-path google/gemma-3-4b-it --iti-artifact-path data/contrastive/truthfulness/iti_truthfulqa_paperfaithful/final_fold0/iti_heads.pt --iti-family truthfulqa_paperfaithful --iti-k 12 --decode-scope first_3_tokens --device-map cuda:0 --alphas -8.0 0.0 4.0 8.0 --conditions selected random_head_seed1 random_head_seed2 random_head_seed3 random_head_seed4 random_head_seed5 random_direction_seed1 random_direction_seed2 random_direction_seed3 random_direction_seed4 random_direction_seed5 --top-k-first-token 10 --mc-max-new-tokens 4 --open-max-new-tokens 64
```

Analyze and adjudicate future open responses:

```bash
uv run python scripts/analyze_simid.py --run-dir data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/prospective_effect_calibrated_open_20260429 --adjudicate-open --adjudication-mode batch --phase0-gates --judge-model gpt-4o --adjudication-output data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/prospective_effect_calibrated_open_20260429/open_adjudication.jsonl --output-json data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/prospective_effect_calibrated_open_20260429/results_calibrated_open.json --report-md data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/prospective_effect_calibrated_open_20260429/report_calibrated_open.md --alias-audit-output data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/prospective_effect_calibrated_open_20260429/alias_audit_queue_calibrated_open.jsonl --prospective-open-authority-manifest data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/mvp_20260427_calibration/human_review_package/prospective_effect_run_gate_20260429/effect_run_manifest.json
```

Before treating open correctness as claim-bearing, the analysis output must
record this package's prospective open-grading authority by path and hash and
the locked manifest must be disjoint from the excluded historical MVP and
prospective-calibration sample IDs.
