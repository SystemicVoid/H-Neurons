# Opus 4.7 Canary — Early-Look Directional Analysis (NOT claim-bearing)

**Scope.** This document is a directional, hypothesis-shaping read of the 200-row Opus 4.7 max canary on the SIMID prospective effect run. It is intended to inform **operational decisions** (whether to scale to a full API-batch labeling pass, where the rubric may need clarification, what cell-level base rates look like) and nothing more. It does **not** evaluate, satisfy, or block any pre-registered effect gate.

**Hard caveats.**

- **Not claim-bearing.** The authority manifest `prospective_effect_run_gate_20260429_r2_external_labels/effect_run_manifest.json` binds a future open-correctness claim to an external label package over the full effect run, not to this 200-row canary.
- **Independent sampling, not paired.** Within each (dataset, family, alpha) stratum we drew independently. Of the 200 rows, paired-same-item overlap between alpha=0 and alpha=8 within a condition is essentially zero. We therefore cannot compute the pre-registered paired same-item delta here.
- **Tiny per-cell n.** Each (dataset, family, alpha) cell has n=16 or 17. Wilson 95% CIs on rates from n=16 are roughly ±25pp wide — far wider than the +5pp minimum practical effect required by the gate. Treat every per-cell rate as a rough prior, not an estimate.
- **Random families pool seeds.** The 66/68 random-head/direction rows pool 5 seeds. Per-seed reads are uninterpretable and are not produced.

## 1. Process / rubric signals

- **rule_gap count:** 1 / 200 (0.5%). Rubric policy treats >0 rule_gap cases as a calibration concern that should be reviewed before scaling.

  Flagged rows (review before scaling):
  - `simid_prosp_open_canary_blind_113213d946502f4e` (review_order=51, label=NOT_ATTEMPTED, confidence=3, flags=['malformed_case']) — **triaged** in `rule_gap_triage/triage_simid_bridge_odql_6602.md`. Root cause: upstream TriviaQA `odql_6602` has a question/gold mismatch faithfully forwarded by the SIMID Bridge generator. Rule_gap call is rubric-correct (rule 8). `simid_bridge_odql_6602` recommended for exclusion at full-batch time via `rule_gap_triage/excluded_sample_ids_recommended_for_full_batch.jsonl`. Frozen rubric and manifest-bound exclusions file unchanged.

- **Confidence histogram:**
  - 2: 1
  - 3: 14
  - 4: 35
  - 5: 150

- **Rows with any flag:** 68 / 200 (34.0%).

- **Flag histogram (top to bottom by count):**
  - `truthfulqa_qualified_answer_boundary`: 34
  - `truthfulqa_non_answer_boundary`: 17
  - `alias_too_broad_or_too_narrow`: 9
  - `bridge_partial_entity_or_modifier`: 6
  - `wrong_extra_answer`: 5
  - `other_boundary`: 3
  - `malformed_case`: 1

## 2. Directional label distributions

**Read every cell with a ±25pp uncertainty band in mind.**

### 2.1 Overall (n=200)

- CORRECT: 38.0% (95% CI 31.6%-44.9%, k=76)
- INCORRECT: 57.5% (95% CI 50.6%-64.1%, k=115)
- NOT_ATTEMPTED: 4.5% (95% CI 2.4%-8.3%, k=9)
- ATTEMPTED: 95.5% (95% CI 91.7%-97.6%, k=191)

### 2.2 By condition family × alpha

| family | alpha | n | CORRECT | INCORRECT | NOT_ATTEMPTED |
|---|---|---|---|---|---|
| selected | 0.0 | 33 | 36.4% (95% CI 22.2%-53.4%, k=12) | 60.6% (95% CI 43.7%-75.3%, k=20) | 3.0% (95% CI 0.5%-15.3%, k=1) |
| selected | 8.0 | 33 | 42.4% (95% CI 27.2%-59.2%, k=14) | 51.5% (95% CI 35.2%-67.5%, k=17) | 6.1% (95% CI 1.7%-19.6%, k=2) |
| random_head | 0.0 | 33 | 42.4% (95% CI 27.2%-59.2%, k=14) | 51.5% (95% CI 35.2%-67.5%, k=17) | 6.1% (95% CI 1.7%-19.6%, k=2) |
| random_head | 8.0 | 33 | 27.3% (95% CI 15.1%-44.2%, k=9) | 72.7% (95% CI 55.8%-84.9%, k=24) | 0.0% (95% CI 0.0%-10.4%, k=0) |
| random_direction | 0.0 | 34 | 44.1% (95% CI 28.9%-60.6%, k=15) | 55.9% (95% CI 39.5%-71.1%, k=19) | 0.0% (95% CI 0.0%-10.2%, k=0) |
| random_direction | 8.0 | 34 | 35.3% (95% CI 21.5%-52.1%, k=12) | 52.9% (95% CI 36.7%-68.5%, k=18) | 11.8% (95% CI 4.7%-26.6%, k=4) |

### 2.3 By dataset × family × alpha

| dataset | family | alpha | n | CORRECT | INCORRECT | NOT_ATTEMPTED |
|---|---|---|---|---|---|---|
| triviaqa_bridge | selected | 0.0 | 17 | 41.2% (95% CI 21.6%-64.0%, k=7) | 58.8% (95% CI 36.0%-78.4%, k=10) | 0.0% (95% CI 0.0%-18.4%, k=0) |
| triviaqa_bridge | selected | 8.0 | 17 | 41.2% (95% CI 21.6%-64.0%, k=7) | 58.8% (95% CI 36.0%-78.4%, k=10) | 0.0% (95% CI 0.0%-18.4%, k=0) |
| triviaqa_bridge | random_head | 0.0 | 17 | 47.1% (95% CI 26.2%-69.0%, k=8) | 52.9% (95% CI 31.0%-73.8%, k=9) | 0.0% (95% CI 0.0%-18.4%, k=0) |
| triviaqa_bridge | random_head | 8.0 | 17 | 35.3% (95% CI 17.3%-58.7%, k=6) | 64.7% (95% CI 41.3%-82.7%, k=11) | 0.0% (95% CI 0.0%-18.4%, k=0) |
| triviaqa_bridge | random_direction | 0.0 | 17 | 47.1% (95% CI 26.2%-69.0%, k=8) | 52.9% (95% CI 31.0%-73.8%, k=9) | 0.0% (95% CI 0.0%-18.4%, k=0) |
| triviaqa_bridge | random_direction | 8.0 | 17 | 41.2% (95% CI 21.6%-64.0%, k=7) | 47.1% (95% CI 26.2%-69.0%, k=8) | 11.8% (95% CI 3.3%-34.3%, k=2) |
| truthfulqa | selected | 0.0 | 16 | 31.2% (95% CI 14.2%-55.6%, k=5) | 62.5% (95% CI 38.6%-81.5%, k=10) | 6.2% (95% CI 1.1%-28.3%, k=1) |
| truthfulqa | selected | 8.0 | 16 | 43.8% (95% CI 23.1%-66.8%, k=7) | 43.8% (95% CI 23.1%-66.8%, k=7) | 12.5% (95% CI 3.5%-36.0%, k=2) |
| truthfulqa | random_head | 0.0 | 16 | 37.5% (95% CI 18.5%-61.4%, k=6) | 50.0% (95% CI 28.0%-72.0%, k=8) | 12.5% (95% CI 3.5%-36.0%, k=2) |
| truthfulqa | random_head | 8.0 | 16 | 18.8% (95% CI 6.6%-43.0%, k=3) | 81.2% (95% CI 57.0%-93.4%, k=13) | 0.0% (95% CI 0.0%-19.4%, k=0) |
| truthfulqa | random_direction | 0.0 | 17 | 41.2% (95% CI 21.6%-64.0%, k=7) | 58.8% (95% CI 36.0%-78.4%, k=10) | 0.0% (95% CI 0.0%-18.4%, k=0) |
| truthfulqa | random_direction | 8.0 | 17 | 29.4% (95% CI 13.3%-53.1%, k=5) | 58.8% (95% CI 36.0%-78.4%, k=10) | 11.8% (95% CI 3.3%-34.3%, k=2) |

## 3. Directional read for the scale-up decision

- Selected: CORRECT at alpha=0 36.4% (n=33), at alpha=8 42.4% (n=33). Marginal independent-sample delta: +6.1pp — not a valid estimate of the paired delta; only directional.
- Random-head: CORRECT alpha=0 42.4% → alpha=8 27.3% (marginal Δ -15.2pp).
- Random-direction: CORRECT alpha=0 44.1% → alpha=8 35.3% (marginal Δ -8.8pp).

**How to read the deltas.** With n=33 selected at each alpha (and 33 paired-by-item rows on the full run typically having ~10pp paired SE for a binary), the canary marginal delta is consistent with anything from a moderately negative to a moderately positive paired effect. Treat the sign as a weak prior, not as evidence.

**Attempt-rate signal.** Overall NOT_ATTEMPTED is 4.5%. The pre-registered attempt-shift blocker treats correctness gains whose CI overlaps an equal-or-larger attempt-rate increase as artifacts. This canary is too small to estimate the attempt-rate shift, but it gives a base rate to anticipate cell sizes for the full run.

## 4. Recommendation

1. **Reuse, do not re-label.** These 200 (blind_case_id, label) rows are durable Opus 4.7 max labels under the frozen rubric (sha256 `80f9baf09d757a9296c931b6973e633571974411e3ec421e6b512dba53df6f6f`). The full-batch labeling pass should detect overlap on `blind_case_id` (or `(sample_id, condition, alpha)` via the private case map) and skip relabeling these rows.
2. **Rubric stress.** Rule-gap rate of 0.5% is non-zero — those rows must be reviewed before scaling, since the rubric policy is `max_rule_gap_cases: 0`. Confidence and flag histograms above identify boundary categories that may benefit from a clarification pass before launching the full batch.
3. **Cost / scale planning.** The full prospective effect package needs labels for all 11 conditions × 4 alphas × 854 rows ≈ 37,576 row-labels (or ≈ 18,788 if scoped to alphas {0, 8}). At ~85s/chunk × ~13-14 rows/chunk in this canary (≈ 6s/row wall, ~7k tokens/row), expect ~4-6 hours of API wall time at the same parallelism, and proportional token spend. Plan rate-limit headroom accordingly.
4. **Stratification at scale.** The full run is by definition exhaustive, so stratification matters only for any sub-pass we choose to run earlier. If we want a tighter mid-scale read before the full batch, the next step up is to label the complete (selected, alpha={0,8}) cells (~1,708 rows) so that paired same-item deltas become computable, and cell n is large enough (≈ ±5pp CI on a binary rate) to actually constrain the gate.

## 5. What this canary does NOT support

- It does not satisfy `external_open_labels_not_loaded_for_all_effect_rows`.
- It does not satisfy `external_open_label_package_not_bound_to_authority_hash` for the full effect run.
- It does not, on its own, decide any of the `effect_pass_gates` (`primary_truthfulqa_open_delta_positive`, `selected_exceeds_random_direction`, `selected_exceeds_random_head`, `mc_degradation_blocker`, `attempt_shift_blocker`).
- It does not justify any rubric revision; the rubric remains frozen at `simid_open_grading_rubric_prospective_20260429/v1`.

## 6. Reusability key for the full batch

Future labeling pipelines should treat the canary's labels as already-collected evidence and key off any of:

- `blind_case_id` in `prospective_open_labels_opus47_canary.jsonl` (matches `review_cases_blind.jsonl`).
- `(sample_id, condition, alpha, run_row_sha256)` from `private_case_map.jsonl`.

Because the `blind_case_id` is a deterministic hash of `(sample_id, condition, alpha, run_row_sha256)` salted with the canary tag, a future blind package keyed on the same recipe will produce identical ids and the overlap can be detected at build time.
