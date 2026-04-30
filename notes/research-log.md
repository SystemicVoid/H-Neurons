# Research Log — Closure & Pivot Phase

> Continues from [research-log-iti-artifact-exploration.md](./act3-reports/research-log-iti-artifact-exploration.md) (2026-03-24 to 2026-04-02), which covers the ITI artifact exploration phase: pipeline hardening, calibration sweep, E0/E1/E2 artifact variants, decode-scope ablation, and the bridge from D3.5 to D4.

---

## 2026-04-30 (Mistral 24B CP5 full run reviewed)

### What I did

Ran and reviewed Mistral 24B CP5: full n=200 FaithEval H-neuron
intervention plus five unconstrained and three layer-matched controls under the
locked `standard` prompt and full alpha grid. The H100 pod was deleted after
final artifact sync. Report:
[`icml/reports/2026-04-30-mistral24b-cp5-faitheval-review.md`](./icml/reports/2026-04-30-mistral24b-cp5-faitheval-review.md).
Companion adversarial pipeline audit (item-level texture, alternative readings,
triage-script gap, uncertainty register):
[`icml/reviews/2026-04-30-mistral24b-cp5-pipeline-audit.md`](./icml/reviews/2026-04-30-mistral24b-cp5-pipeline-audit.md).
Companion hypothesis-formulation review (why CP5 diverged from our Gemma
replication and the H-Neurons paper, including the 2501-vs-2503 question and
ranked falsifiable hypotheses):
[`icml/reviews/2026-04-30-mistral24b-cp5-null-causes-and-2501-vs-2503.md`](./icml/reviews/2026-04-30-mistral24b-cp5-null-causes-and-2501-vs-2503.md).

### What I expected vs what happened

Expected CP5 to decide whether the Mistral detector readout transfers to a
behavioral intervention. It did not. Contracts and parser checks passed cleanly,
with zero parse failures throughout, but the H endpoint was null: alpha
`0.0 -> 3.0` was `0.0 pp` with balanced `9` true-to-false and `9`
false-to-true item flips. The raw control summary marks the H slope as
separated from random controls, but the stricter review rejects that slope-only
result because the paired endpoint and no-op-to-max effects are null.

### What this changes about my thinking

The Mistral readout is real, but this positive-weight H-neuron intervention
rule does not currently earn a cross-model FaithEval replication claim. The
paper should not present Mistral as confirming the H-neuron intervention effect.
If used, CP5 is a clean-control null: prompt/parser/control readiness passed,
effect gate failed.

### What I will do next

Freeze the CP5 artifacts and update manuscript/reviewer framing around a
Mistral null rather than launching a Mistral SAE or downstream transfer run as
if CP5 succeeded.

---

## 2026-04-30 (Mistral 24B CP4 smoke reviewed)

### What I did

Reviewed the completed CP4 FaithEval pilot/control smoke under
`data/mistral24b/intervention/faitheval_pilot_smoke/`. The H-neuron run and all
four quick controls share the same n=20 Mistral sample lock, `standard` prompt,
alpha grid `0.0 1.0 3.0`, model key/path, and classifier hash. The wrapper
completed without launching CP5, and local `check-stage` revalidation passed
for the H output plus all quick-control directories. I also patched the
negative-control summarizer so this exact slope-only/no-op mismatch now returns
`review_baseline_mismatch` instead of `specificity_supported`.

### What this changes about my thinking

CP4 passed as a prompt/parser/control readiness gate, not as an effect-size
result. Parse failures were 0/20 for every H alpha and every control seed/alpha,
so the standard FaithEval parser is safe enough for the n=200 run.

The interesting warning is the pilot curve. H was 15/20 -> 13/20 -> 13/20,
while every quick control was 13/20 -> 13/20 -> 13/20. The two apparent H flips
(`MCAS_2006_9_44`, `Mercury_7012740`) were already false in every control
alpha=0 rerun, so the synced CP4 `specificity_supported` triage is
baseline/rerun variance, not intervention evidence. A read-only recomputation
under the patched logic returns `review_baseline_mismatch`. CP5 should still
run, but its review must separate endpoint and paired item effects from slope
and reject any slope-only result driven by alpha=0 mismatch.

### What I will do next

CP5 has since run and is reviewed above. Do not start Mistral SAE or CP6/CP7
as if the full FaithEval result cleared the no-op baseline audit; it did not.

---

## 2026-04-29 (Mistral 24B CP2/CP3 detector gate reviewed)

### What I did

Reviewed the completed Mistral 24B CP2/CP3 run: canonical train/dev/test
splits, 1,840 activation arrays under
`data/mistral24b/pipeline/activations_llm_canonical/`, and the saved L1
classifier `models/mistral24b_classifier_canonical.pkl`. Wrote the canonical
analysis at
[2026-04-29-mistral24b-cp23-pipeline-review.md](./icml/reports/2026-04-29-mistral24b-cp23-pipeline-review.md),
updated the Mistral strategy memo and older scope/pipeline context to point at
it, and removed the run from `notes/runs_to_analyse.md`.

### What this changes about my thinking

Mistral is no longer just infrastructure plus planning: the same-family 2501
anchor now has a real held-out sparse detector gate. The test result is
accuracy 0.775 [0.715, 0.830], F1 0.7783 [0.7184, 0.8349], AUROC 0.8711
[0.8185, 0.9172], with 10 positive selected H-neuron targets out of 1,310,720
FFN neurons. That supports continuing to CP4, but it still does not support a
Mistral intervention claim. The important caveats are: the model is 2501, not
exact 2503/3.1; the false class is nearly exhausted after split construction;
and the saved logistic model has 10 negative nonzero coefficients in addition
to the 10 positive intervention targets.

### What I will do next

Use CP4 as a smoke gate, not as a formality. The next paid run should verify
FaithEval prompt/parser/evaluator parity across H-neuron, unconstrained random,
and layer-matched random controls before any full CP5 claim-bearing run.

## 2026-04-29 (SIMID prospective open gate returned)

### What I did

Analyzed the returned labels for the frozen
`prospective_open_calibration_gate_20260429` package and recorded both LLM
runs. Opus 4.7 max clears the pre-specified prospective policy on the 150-case
blind sample: 138/150 raw agreement = 92.0%, kappa=0.8734, AC1=0.8831, and
0 rule gaps. Codex 5.5 xhigh fails the policy only on raw agreement:
133/150 = 88.7%, kappa=0.8207, AC1=0.8346, and 0 rule gaps. The two LLM runs
agree with each other on 135/150 cases.

After the human prospective labels arrived, analyzed the same frozen package.
The human return is valid but fails the policy: 131/150 raw agreement = 87.3%,
kappa=0.7979, AC1=0.8159, and 0 rule gaps, with blockers
`raw_agreement_below_threshold` and `cohen_kappa_below_threshold`. It agrees
with Opus on 133/150 cases and with Codex on 132/150 cases.

Then froze the next prospective effect-run gate, now superseded by the r2
external-label package at
`data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/mvp_20260427_calibration/human_review_package/prospective_effect_run_gate_20260429_r2_external_labels/`.
The original `prospective_effect_run_gate_20260429/` remains append-only and
diagnostic. The r2 package binds the passing Opus analysis by path and hash,
but requires complete external blind labels for claim-bearing open correctness;
`gpt-4o` adjudication is diagnostic-only. It pre-specifies held-out TruthfulQA
plus Bridge rows, selected paper-faithful ITI k=12 first-3-token steering, an
unhooked no-op preflight, five random-direction seeds and five random-head
seeds, alpha grid [-8, 0, 4, 8], exclusion of historical MVP/prospective-
calibration sample IDs, and the primary paired TruthfulQA open-correctness
delta at alpha=8 versus alpha=0.

### What this changes about my thinking

The frozen rubric now has a passing prospective LLM calibration path when using
the Opus evidence, but it still does not rescue historical MVP open-correctness
deltas. The Codex and human failures are informative rather than
disqualifying: Bridge exact-entity strictness is mostly stable in this sample,
while TruthfulQA hedged anti-myth answers remain the sensitive boundary. The
human prospective return is materially stronger than the earlier 100-case human
UI pass, but it still does not provide a rater-diverse pass for the frozen
policy.

### What I will do next

Keep historical SIMID MVP open metrics diagnostic-only. For future SIMID effect
claims, use the frozen prospective rubric and passing gate evidence, then rerun
the effect analysis under that calibrated policy with random-direction and
random-head controls plus the pre-specified effect-run gate. Summarize the
Codex/human disagreement families and export a fresh blind sample after any
rubric revision; do not tune this effect plan on the same returned labels.

## 2026-04-28 (SIMID open calibration finalized)

### What I did

Reviewed the completed production SIMID open-calibration pass for
`mvp_20260427_calibration`: 402 secondary `gpt-5.5` labels, 34
disagreement adjudications, and the finalized
`open_calibration_summary.json`. Wrote the current calibration authority at
[2026-04-28-simid-open-calibration-review.md](./icml/reports/2026-04-28-simid-open-calibration-review.md)
and updated the SIMID measurement contract to distinguish missing calibration
from recorded-but-failed calibration. Also patched the SIMID report renderer so
future reports do not describe failed calibration as merely "not recorded."

### What this changes about my thinking

The calibration result is reassuring but not claim-clearing: raw agreement is
91.5% [88.4, 93.9] and AC1 is 0.8974, but Cohen's kappa is 0.7594, below the
pre-recorded 0.8 threshold. The open-correctness blocker is now
`calibration_evidence_failed_thresholds`, not
`calibration_evidence_not_recorded`. The MVP open effects remain diagnostic
only. The most interesting calibration detail is that Bridge is the weaker
dataset under second-rater agreement (kappa 0.6756) while TruthfulQA passes the
dataset-level kappa slice (0.8063), despite TruthfulQA being the more obvious
deterministic-alias failure mode. That shifts the next audit target toward
Bridge partial-entity and modifier strictness cases, not just TruthfulQA
paraphrase.

### What I will do next

Do not relax the kappa gate post-hoc. If open correctness needs to become
claim-bearing, run a human or rater-diverse calibration pass on all 34
disagreements plus a stratified agreement sample, then rerun effect reporting
only after the calibration policy clears. In parallel, keep the selected
alpha=8 TruthfulQA open gain framed as an exploratory diagnostic until
multi-seed controls and a pre-specified curve summary land.

## 2026-04-27 (SIMID MVP calibration run audit)

### What I did

Audited the full SIMID MVP run at `mvp_20260427_calibration`: 400 manifest rows, 3 conditions, 4 alphas, 4,800 primary `gpt-4o` open adjudications, and a 402-row open-calibration queue awaiting secondary labeling/finalization. Wrote the canonical report at [2026-04-27-simid-mvp-calibration-audit.md](./icml/reports/2026-04-27-simid-mvp-calibration-audit.md). Also verified the parallel no-go review of the calibration-labeling bundle: the canary really was blocked by the new queue validator, and malformed primary grades could pass the pre-spend check before failing later. Patched both paths, added queue-row hashes for newly produced secondary labels, and verified with the SIMID test module plus shellcheck.

### What this changes about my thinking

The MVP is a measurement/pipeline result, not a claimable intervention result. Selected ITI has an interesting TruthfulQA alpha=8 open-correctness cell (+12 pp [4, 20]), but it comes with TruthfulQA MC degradation, Bridge open degradation, incomplete specificity against controls, and no finalized judge calibration. The larger panel also corrects the phase-0 measurement story: deterministic alias grading is worst on TruthfulQA, but Bridge alias matching is not universally sufficient either (10/100 baseline unique Bridge responses disagree with `gpt-4o` on open correctness; 11/100 differ on the full grade/attempt label). The right default is now "deterministic alias is a diagnostic floor; judge adjudication needs calibration; cite open-correctness effects only after the secondary-rater pass."

### What I will do next

Superseded by the 2026-04-28 entry above: the calibration pass has now run and failed the kappa threshold. Only after a human or rater-diverse calibration pass clears the policy should the MVP open metrics be reconsidered as claim-bearing. Any next effect run should pre-register the curve summary and add more random-head/random-direction seeds before claiming specificity.

## 2026-04-27 (SIMID open-adjudication pipeline review)

### What I did

End-to-end scientific-rigor audit of the new SIMID open-grading judge-adjudication pipeline introduced by the working-tree changes to `scripts/analyze_simid.py` (+675 lines), `scripts/evaluate_intervention.py` (parser fix), and `tests/test_simid.py` (+137 lines / 7 new tests), exercised on the `phase0_20260426_113707_gates` run (16 paired base items, `alphas=[0.0]`, conditions `{selected, unhooked}`, 2 option-order replicates → 64 adjudication rows; judge `gpt-4o`, prompt `simpleqa_verified_aliases/v1`, batch mode, `temperature=0`, `max_tokens=10`). Independently re-derived every reported aggregate from `open_adjudication.jsonl` and `results_adjudicated.json`: 40 CORRECT / 24 INCORRECT raw counts, 16 deterministic-vs-judge disagreements (all one-direction, all in TruthfulQA, on 4 unique base items × 4 replicates), pooled adjudicated open accuracy 0.6250 [0.3750, 0.8750], stratum rates 0.7500 (bridge) / 0.5000 (truthfulqa-test) / 0.5000 (truthfulqa-train) — every figure matches to floating-point precision. Verified provenance sidecars and `openai_request_id` per row, batch-state cleanup behaviour (`_clear_state` in `scripts/openai_batch.py:617` — the missing `*.batch_state.json` is correct), audit-queue compression (68 reasons → 16 disagreement entries), source-of-grade tracking (`effective_grade_source_counts: {adjudication: 64}` in full run, `{adjudication: 3, deterministic_alias: 61}` in canary), and end-to-end noop equivalence (selected ≡ unhooked at α=0 byte-identically through the judge). Wrote canonical pipeline review at [2026-04-27-simid-open-adjudication-pipeline-review.md](./icml/reports/2026-04-27-simid-open-adjudication-pipeline-review.md).

### What this changes about my thinking

The substantive update is about the **measurement instrument, not about SIMID**. On TruthfulQA paired items, the deterministic alias grader has a **50% miss rate** (4/8 unique items), all in one direction (judge more lenient than alias) — adjudication moves the per-stratum open accuracy from `0.000` to `0.500` in *both* test and train splits, a +50 pp gap driven entirely by paraphrase. The canonical examples are textbook: "We use 100% of our brain" vs alias "virtually all", "There's no waiting period" vs "You should not wait". On TriviaQA bridge the deterministic alias grader is clean (0/8 disagreements), as expected from the named-entity gold structure and consistent with the [L4 bridge IRR closure](../paper/icml/reports/2026-04-21-bridge-irr-review.md) (κ=0.90 against `gpt-4o-2024-11-20` Rater B). The implication for any prior or future TruthfulQA-open claim using deterministic grading is that paraphrase noise and intervention signal are mixed at the +50 pp scale — the deterministic alias grader is not a usable open-correctness instrument on TruthfulQA, and any treatment-effect number on it should be re-derived under judge adjudication after the calibration pass.

A second observation worth recording: `parse_simpleqa_verdict` had a latent substring bug — the keyword-fallback ordering was `(NOT_ATTEMPTED, CORRECT, INCORRECT)` and `"CORRECT"` is a substring of `"INCORRECT"`, so any free-text "INCORRECT" judge response that fell through to the keyword path would have been silently parsed as CORRECT. The change reorders to `(NOT_ATTEMPTED, INCORRECT, CORRECT)`, which fixes it. Crucially, **the bug did not fire on this run** — every one of the 64 judge responses is a clean single letter (40 A / 24 B), so the regex `\b([ABC])\b` catches them all and the keyword path is never reached. The fix matters for future judges or higher `max_tokens` settings; the new test suite covers letter and word parsing of canonical labels but not a regression for the substring case (e.g., a sentence containing "INCORRECT" should not return CORRECT). One unit test would lock the fix in.

A third observation, on cost: the adjudication key is `(sample_id, condition, alpha)`, but the open prompt has no MC options (option-order doesn't affect open generation) and at α=0 the noop equivalence makes selected≡unhooked. So this 64-row run has only **16 unique (`base_sample_id`, `response`) pairs** — 48 of 64 judge calls are paying for verdicts on already-judged text. Concrete dedup opportunity: bucket by `(base_sample_id, response)` before submitting batch; saves 75% on phase-0 noop runs and ≥50% across an alpha sweep. Also worth noting: claimability is correctly gated to `false` with `claimability_blocker = calibration_evidence_not_recorded`, mirroring the bridge IRR pattern — this run cannot back a paper claim until a parallel calibration pass (κ / AC1 / pre-frozen rule) is recorded.

### What I will do next

(1) Add the missing parser regression test (free-text "INCORRECT" → INCORRECT) to lock the fix. (2) Specify a SIMID open-grade calibration pass on the L4 template — second judge (or human subset) on the audit-queue disagreement set + a stratified balanced sample, pre-frozen rule, recorded κ / AC1 / rule_gap; only then upgrade `claimable_open_correctness`. (3) Add the `(base_sample_id, response)` dedup before the batch submission step in `analyze_simid.py`'s adjudication path; this is independent of (2) and harvests immediate cost savings. (4) Add a CI gate that refuses to mark mixed-source runs claimable without an explicit override flag — the canary run shows that partial adjudication produces apparent selected/unhooked asymmetry that is purely a sampling-coverage artefact. (5) Decide whether the +50 pp TruthfulQA paraphrase miss number deserves a standalone measurement-instrument note in [measurement-blueprint.md](./measurement-blueprint.md); it is not a SIMID intervention claim and should not be folded into one.

---

## 2026-04-25 (FaithEval SAE selector — answer-span pool extension review)

### What I did

End-to-end scientific-rigor review of the 2026-04-25 `answer_span_selected` extension to the FaithEval SAE utility-selector ablation (commits c39a4db / 7bfdf96 / d335492 / c27d002). Verified the new selector pipeline (`scripts/select_faitheval_sae_utility_features.py` + tightened guards in `scripts/infra/faitheval_sae_utility_selector.sh`): the answer-span family is built from a `selector_score = baseline − ablated` mean reduction in the first-3-assistant-content-token answer-text logprob margin on the 160-validation split, with `top_k=266`; the `matched_random_answer_span_seed_{0..9}` controls reuse the existing Efraimidis–Spirakis weighted-without-replacement A-Res sampler against the same 112 004-feature `token_activation_rate > 0` zero-weight pool, exact-matched to the answer-span layer histogram (10 distinct seed fingerprints). The new `faitheval_answer_span_margin` benchmark and the selector share the same scoring code path (`score_faitheval_answer_text_targets_from_prompt_ids`, `FAITHEVAL_ANSWER_SPAN_PRIMARY_WINDOW_TOKENS = 3` defined once in `run_intervention.py`). Independently re-derived all 14 family means and 14 compliance counts on each of the three metrics (compliance, anti-compliance margin, answer-span margin) from raw `alpha_*.jsonl` files; matched the report values to floating-point precision. Re-derived all 30 paired-delta point estimates against the report. Re-derived the seed-mean / seed sd / seed range for the answer-span margin contrast; matched. Verified test-manifest sample-ID parity (fingerprint `781fd7eafa5f2573`) across all 84 experiment directories (zero mismatches), `selector_summary.selector_scoring.input_hash` ↔ `selector_scoring_state.input_hash` linkage (matches), schema versions (v6 / v6), and `audit_ci_coverage.py` (passes). Found 2 stale provenance sidecars (`status="running"` / `status="interrupted"`) leftover from earlier failed attempts; both directories also have a completed canonical sidecar that backs the present alpha jsonl, so it's a cosmetic finding, not a data-integrity issue. Wrote canonical extension report at [paper/icml/reports/2026-04-25-faitheval-answer-span-extension.md](../paper/icml/reports/2026-04-25-faitheval-answer-span-extension.md); updated 2026-04-22 review and `paper/icml/reviews/TODO_Limitations_Fixes.md` L2 closure to point at it.

### What this changes about my thinking

The compliance null is now stronger (a third intervention-aware selector did not move the behavioural endpoint), but the more interesting update is the cross-metric tradeoff. `answer_span_selected − utility_selected = +0.46 nats [+0.16, +0.75]` on the anti-compliance margin while `answer_span − noop = −0.33 [−0.61, −0.06]` on the answer-span margin: two intervention-aware rules sharing 167/266 features (Jaccard 0.46) inside the same probe-nonzero pool pick feature sets with **opposite signed effects** on the two margin metrics. This is a clean negative result for the implicit assumption that "intervention-aware" SAE selection is unique up to noise — within FaithEval's candidate-pool, the choice of intervention objective (prompt-end forced-choice margin vs first-3-token answer-text margin) is itself a load-bearing modelling decision, even though both objectives target the same underlying counterfactual-vs-preferred contrast. The 2026-04-22 review's "good readout ≠ good steering handle" claim now has a sibling: "good steering handle for one margin metric is not necessarily a steering handle for another margin metric, even on the same task." Important caveat I'm explicit about in the report: the answer-span margin is the same metric the selector trained on (only the validation/test split differs), so the tightest signal in the bundle (seed-mean −0.53 nats with seed sd 0.15) is a within-metric generalisation claim, not a transfer claim. The cross-metric contrasts (compliance, anti-compliance margin) are the conservative tests, and on those metrics the answer-span pool is null or *negative* respectively.

A second, smaller observation: for the `answer_span_selected` compliance contrast, the naive seed bootstrap CI [+0.38, +1.19] excludes 0 while the (documented primary) nested paired bootstrap CI [−0.80, +2.45] includes 0. This is the textbook divergence pattern: 8/10 per-seed point estimates clustered between +0.83 and +1.43 pp, but per-sample variability across the 840-item test set swamps the mean shift. The naive bootstrap is misleading here and the report's `primary_ci_method = "nested_bootstrap"` is the right choice; documented this explicitly in §3.5 of the new report so future readers don't get pulled into citing the naive CI.

### What I will do next

The highest-leverage next experiment is the disjoint-feature causal probe: apply only the 99 features in `answer_span_selected ∖ utility_selected` (and separately the 99 in `utility_selected ∖ answer_span_selected`) under the same `delta_only` α=0 operator, on all three metrics. Predicts: the disjoint-to-utility set drives the +0.46 nat anti-compliance loss; the disjoint-to-answer-span set drives utility's anti-compliance advantage. ≈30 min GPU. Discriminates "real cross-metric structure" from "noisy-tail features dressed as structure" — one of those two readings should be the headline. Also still on the docket: the augment k=154 → 10-seed extension (carries from 2026-04-22 §5 item 7; ≈45 min GPU; gives a proper seed-mean CI on the augment selection-specific margin claim) and the L3 minimal layer extension. Cosmetic: archive the two stale `status="running" / "interrupted"` sidecars under an `experiment_<date>_aborted_attempts/` sibling dir so the provenance audit line stays clean.

---

## 2026-04-23 (evening — 10-seed + path-drift extension)

### What I did

Executed the two highest-leverage follow-ups from the 2026-04-22 audit's §5 in a single pipeline pass: (1) extended `matched_random` for the main k=266 FaithEval SAE bundle from 3 to 10 seeds, (2) added a dedicated path-drift control (`matched_zero_dead`) using a single layer-20 SAE feature with zero classifier weight and zero validation-token activation (`flat_idx`=147 473), run at α=0 under `delta_only` steering. Pipeline regenerated `selector_summary.json`, `heldout_summary.json`, and `audit_note.md` (schema bumped to v5) with cache-reuse confirming the underlying utility scores, feature stats, and selection manifests were carried forward unchanged. Independently reproduced every family mean margin and compliance count from the raw `alpha_0.0.jsonl` files (14/14 exact match on margins, 14/14 match on compliance counts), verified pairwise Jaccard overlap of 0.019–0.051 across the 10 matched-random seed manifests, and re-derived the nested paired bootstrap and naive seed bootstrap CIs for the 10-seed contrast. `audit_ci_coverage.py` passes. Refreshed [paper/icml/reports/2026-04-22-faitheval-sae-utility-selector-review.md](../paper/icml/reports/2026-04-22-faitheval-sae-utility-selector-review.md) to reflect the extended bundle; the augment k=154 section is unchanged (still 3 seeds, still the single cleanest selection-specific result).

### What this changes about my thinking

Three upgrades to the 2026-04-22 stance. First, the main-bundle selection-specific margin signal that the 3-seed analysis had downgraded to "not robustly separable from noise" is in fact bounded away from zero at 10 seeds: seed-mean contrast −0.897 nats, nested paired bootstrap [−1.22, −0.56], naive seed bootstrap [−1.17, −0.61]; 9 of 10 seeds are negative. Seed SD (0.489 nats) is now ≈half the effect size rather than comparable to it. Seed_2 remains the single outlier where random-selected features produce a lower margin than utility-selected, but it is 1 of 10 rather than 1 of 3. The "rely on the k=154 augment, not the main k=266 bundle" framing from earlier today is too conservative — the main bundle now survives on the same seed-mean terms. Second, the `matched_zero_dead − noop` path drift is +8.2 × 10⁻⁸ nats (paired CI [−2.4 × 10⁻³, +2.1 × 10⁻³]) with 0 compliance flips and only 8/840 per-sample differences, all quantised at bf16 rounding scales (±0.25 / ±0.5 nats) and near-balanced in sign. This empirically falsifies "maybe all the utility/readout/random effects are just artifacts of running the SAE code path at α=0" — that hypothesis is four orders of magnitude below the observed effects. Third, because the path-drift control is a measured zero, the `utility − noop` and `utility_positive − noop` deltas (which the 2026-04-22 report qualified as "may include intervention-path artifacts") can now be read as genuine selection+firing effects — though the matched-random contrast remains the more informative statement because it also controls for "any 266 zero-weight layer-matched features fire non-trivially".

### What I will do next

Done this session: updated the L2 closure narrative and limitations-table entry in `paper/icml/reviews/TODO_Limitations_Fixes.md` to reflect the 10-seed + path-drift evidence. Primary remaining follow-up: extend the k=154 augment from 3 to 10 seeds for parity (item 7 from §5), so the augment has a proper seed-mean CI instead of a 3-seed descriptive range — this is the last seed-count gap and costs ≈45 min GPU. Secondary: move `readout − noop = +0.92 nats` into the main-text framing (item 3 from §5). Out of scope today: the L3 wider-layer extension (item 5) and the readout-weight-sign cosmetic fix.

---

## 2026-04-23 (morning)

### What I did

Scientific-rigor review of the 2026-04-22 rerun of the FaithEval SAE utility-selector matched_random control. Verified six reran held-out families (3 main k=266 seeds + 3 augment k=154 seeds) across both `faitheval` and `faitheval_anti_compliance_margin` benchmarks: 12 provenance sidecars with `status=completed`, 840 records each, test-manifest ID parity (hash `c2ad5a31…`) consistent across all families. Checked the new `match_random_zero_weight_features` implementation (Efraimidis-Spirakis weighted-without-replacement on the `token_activation_rate > 0` pool, exact layer-histogram match) — distinct fingerprints confirmed for all six seeds against `selector_summary.matched_random_controls.seed_families` and the augment counterpart. Compliance counts and mean margins reproduced directly from the alpha_0.0 jsonl files match the regenerated `heldout_summary.json` and `augment_heldout_summary.json` to rounding; `audit_ci_coverage.py` passes. Consolidated the `paper/icml/reports/2026-04-22-faitheval-sae-utility-selector-review.md` so the §0 superseding update is merged into a single canonical body; the old §4.1–§4.4 prompt-end-control essay is collapsed to a short historical paragraph pointing at the archived `experiment_2026-04-22_prompt_end_zero_weight_control/` dirs.

### What this changes about my thinking

The new control recovers a proper 3-seed null, but at the cost of exposing how noisy the main k=266 `utility − matched_random` contrast actually is: across seeds it spans −1.21 to +0.14 nats with seed sd ≈ 0.68 nats, i.e. comparable to the effect itself. In seed_2, random features beat utility-selected on the margin. The strictly-positive k=154 augment is much better behaved — sign-consistent across all three seeds, seed sd ≈ 0.21 nats, all CIs excluding 0. So the paper-facing margin claim should lean on the augment (k=154) for the selection-specific story and not on the main k=266 contrast, while the headline accuracy null and the `readout − noop`/`utility − readout` contrasts (which don't depend on `matched_random` at all) remain the cleanest evidence.

### What I will do next

Keep items 2–6 from the 2026-04-22 audit's §5. Highest-leverage cheap follow-up is adding 7–10 additional random-null seeds for the main bundle (≈30 min GPU) so the main-bundle contrast can be summarised as a permutation distribution with a seed-mean CI, rather than three noisy point estimates.

---

## 2026-04-22

### What I did

Ran the Priority-2 L2/L3 target-selection ablation (TODO_Limitations_Fixes §Priority 2) and reviewed it end-to-end. Full audit lives under `paper/icml/reports/` (new default for limitations-closure reports): [../paper/icml/reports/2026-04-22-faitheval-sae-utility-selector-review.md](../paper/icml/reports/2026-04-22-faitheval-sae-utility-selector-review.md).

### What I expected vs what happened

Expected a clean binary result: either utility-aware selection rescues the SAE null (pivoting the paper toward "selector matters") or the null survives and closes L2. What happened was the second outcome — FaithEval accuracy is null across readout / utility-selected (k=266) / utility-positive (k=154) / matched-random families — but with two informative complications.

First, on the anti-compliance margin endpoint, readout-selected ablation moves the misleading-preferred margin in the *wrong* direction (+0.92 nats vs noop [+0.61, +1.23]), while utility-selected reduces it by −0.76 nats [−1.08, −0.42]. Within the same candidate pool, the two selection rules pick features with opposite causal effect. That is a sharper "good readout ≠ good steering handle" claim than the matched-AUROC comparison alone.

Second, the `matched_random` control turned out to be two kinds of broken. The `match_random_zero_weight_features` function collapses to a deterministic flat_idx tiebreak because zero-weight SAE features have near-identical prompt-end stats (act_freq≈0 for 99.5% of them; decoder_norm≡1.0 by construction); the "three seeds" are byte-identical manifests, and the "layer-matched zero-weight" set ends up being the lowest-flat_idx dead features per layer. On top of that, α=0 on a dead-feature manifest is a near-noop numerically but not a true noop — it moves the margin −0.41 nats vs the α=1.0 shortcut, purely as an intervention-path artifact. So `utility − matched_random = −0.35 nats [−0.65, −0.05]` is the correct clean selection-specificity estimate, but the matched_random "three-seed null" is not actually a three-seed null.

### What this changes about my thinking

The paper-facing L2 story is unchanged in direction but sharpened in evidence. Instead of leaning on `utility − matched_random` (compromised), the strongest claims should be (1) accuracy is null across every selection rule, including one trained on the scoring metric itself, and (2) readout-selected ablation worsens the margin while utility-selected improves it — a paired contrast inside the probe-nonzero pool where the intervention-path drift cancels. Both are cleaner than the matched-random comparison and they do not depend on the buggy null.

L3 remains partial by design: the search stayed inside the existing SAE extraction layers and did not re-fit SAEs at new layers or widths. This is acknowledged in the audit.

### What I will do next

1. Rerun `matched_random` with an actually-random, layer-activity-matched null — uniformly from the probe-nonzero pool minus the utility/readout selections, or from zero-weight features weighted by full-sequence (not prompt-end) activation. Three genuinely independent seeds.
2. Add an intervention-path baseline (α=0 on a single guaranteed-dead feature) so `X − noop` deltas can be decomposed into path drift vs selection.
3. Fold the `readout − noop = +0.92` nats finding into the main line of the paper. It makes claim 2 of the audit stand alone without needing matched_random.
4. Move the limitations table: L2 from "central weakness" to "addressed via target-selection ablation (readout vs utility sign divergence; accuracy null under intervention-aware selection)"; L3 stays "partially addressed".

---

## 2026-04-16

### What I did

Promoted D7 to a new two-seed current-state source of truth: [2026-04-16-d7-full500-two-seed-current-state-audit.md](./act3-reports/2026-04-16-d7-full500-two-seed-current-state-audit.md). Regenerated the machine-readable D7 summary, integrated `causal_random_head_layer_matched/seed_2`, updated the D7 site-data export, and rewired the live note hierarchy so the old April 14 report is now historical rather than canonical.

### What I expected vs what happened

Expected seed 2 either to behave like seed 1 closely enough to be almost bookkeeping, or to introduce enough heterogeneity to reopen the D7 interpretation. What happened is in between, but importantly not destabilizing. On normalized strict harmfulness, seed 2 lands at **38.8%** versus seed 1 at **37.2%** and probe at **34.8%**, while causal remains far cleaner at **24.8%**. The seed-2-versus-seed-1 gap is only **+1.6 pp [-2.4, +5.4]**, so there is no evidence of a sign flip or qualitatively different random-head regime.

The more interesting update is comparative ordering. Probe versus random seed 1 remained suggestive on strict harmfulness in the April 14 note. With seed 2 added, probe versus random seed 2 is now **-4.0 pp [-7.8, -0.2]**, so probe looks cleaner than one random seed with a CI that just excludes zero. The main claim is therefore firmer and narrower at the same time: causal beats both random seeds and probe, but the panel is still mixed-ruler and the probe/random branches still carry explicit CSV2 error debt.

### What this changes about my thinking

The strongest D7 caveat has shifted. It is no longer "the layer-matched control is only one seed." It is now "the current panel is still mixed-ruler and error-bearing, so even a two-seed control family does not make the result mechanism-clean." That is a better place scientifically: less vulnerable to bookkeeping objections, still disciplined about what has not been earned.

### What I will do next

Keep all live D7 references anchored to the 2026-04-16 two-seed audit and the regenerated `d7_full500_current_state_summary.json`. If D7 later needs to become more central, the next cleanup target is not more narrative reframing; it is reducing mixed-ruler and evaluator-error debt on the current panel.

---

## 2026-04-21

### What I did

Added a framing-governance pass to stop one repeated interpretation from auto-directing the whole repo. The new authority is [2026-04-21-claim-framing-governance.md](./2026-04-21-claim-framing-governance.md), with explicit rerouting in `README.md`, `notes/AGENTS.md`, sprint/strategy guidance, and historical banners on the loudest framing docs. Also added a Wave 2 handoff file: [2026-04-21-framing-remediation-backlog.md](./2026-04-21-framing-remediation-backlog.md).

### What I expected vs what happened

Expected the problem to live mainly in one strategy memo. What actually happened is that the April 11 strategic assessment, the V2 critique, the outline stack, and downstream AI-review machinery had turned one setting-specific result into a repo-wide default. The result itself is still real. The inflation came from routing and repetition: old notes were being read as current authority, and later prompts inherited that hierarchy as if it had been independently re-earned.

### What this changes about my thinking

The fix is governance, not counter-slogan. A late corrective note cannot win if `README.md`, `notes/AGENTS.md`, and historical planning docs keep routing future discussion back to the old center of gravity. The project needs a plural-anchor default: FaithEval for localization/control, bridge for control/externality, jailbreak for measurement/conclusion, and detector-interpretation work for localization fragility. That keeps the FaithEval result in scope without letting it dominate every subsequent decision.

### What I will do next

Use the governance note as the first framing reference for new discussions until a deliberate Wave 2 pass updates public and live manuscript-facing surfaces. When revisiting `site/**`, the archived long draft, or `paper/icml/**`, preserve the result but narrow the ranking claim rather than trying to erase it.

---

## 2026-04-14

### What I did

Closed the remaining D7 analysis gap and rewrote the D7 source hierarchy around a single current-state report: [2026-04-14-d7-full500-current-state-audit.md](./act3-reports/2026-04-14-d7-full500-current-state-audit.md). Extended the D7 summary builder to emit a new machine-readable source of truth at `data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/d7_full500_current_state_summary.json`, integrated the newly scored full-500 `probe_locked` branch and the layer-matched random-head seed-1 branch, added supersession notes to the older D7 reports, and updated the live context files to point at the new audit instead of the April 8 trimmed report.

### What I expected vs what happened

Expected the new `probe_locked` run either to preserve the pilot's clean probe-null story or to collapse the D7 comparison entirely. What actually happened is more informative and messier. On the current normalized strict-harmfulness panel, the ordering is strong and stable in one direction: baseline 51.6%, L1 comparator 46.8%, random layer-matched seed 1 37.2%, probe 34.8%, causal 24.8%. So causal remains the strongest completed branch, and the April 8 bottom-line claim that selector choice matters on this benchmark survives scrutiny.

But the mechanism-clean story did **not** get simpler. The current panel is mixed-ruler rather than cleanly matched to the April 8 legacy panel. Both the random and probe branches contain explicit CSV2 span-validation errors (8 and 12 respectively), and the causal branch still has 112/500 token-cap hits. Probe no longer supports the stronger full-500 statement "null at every alpha"; the clean probe-null remains the pilot result, while the full-500 probe branch is better understood as weaker than causal on the normalized panel and ambiguous on binary harmfulness.

### What this changes about my thinking

D7 is now in a better and narrower place. It is stronger than "unfinished" and weaker than "selector-specific proof." The right paper-facing framing is:

- pilot probe-null remains the cleanest selector-side evidence;
- full-500 D7 is supportive but mixed-ruler;
- causal outperforms the available probe and random branches on the current normalized panel;
- selector specificity is still not mechanism-clean.

That shift matters beyond the D7 note itself. Leaving the April 8 audit as the live canonical reference would now overstate some claims and understate others, which is exactly how source drift creeps into sprint notes, paper prose, and the site.

### What I will do next

Keep all live D7 references anchored to the 2026-04-14 current-state audit unless a statement is explicitly historical provenance from April 8. Any stronger D7 claim now needs to earn its way past the mixed-ruler/error-bearing caveats rather than inheriting authority from the earlier trimmed audit.

---

## 2026-04-13 (later evening)

### What I did

**9. V2 vs V3 paired evaluator comparison — the full Anchor 3 analysis.** Deep paired analysis of H-neuron 38 data scored by both v2 and v3 evaluators on the same model outputs, plus seed-1 v3 control for specificity. Computed bootstrap CIs on all v3 slopes, built the complete ID-level transition matrix, and decomposed the slope compression mechanism. Report: [2026-04-13-v2-v3-paired-evaluator-comparison.md](./act3-reports/2026-04-13-v2-v3-paired-evaluator-comparison.md).

### What I expected vs what happened

Expected the v3 slope CI to include zero (confirming the pipeline audit's prediction). It does: +0.46 pp/alpha [-1.46, +2.41]. The measurement-changes-conclusions framing holds cleanly. But the bigger surprise came from the v3 primary_outcome taxonomy.

**The severity-shift finding changes the story.** The v3 harmful_binary rate is flat, but substantive_compliance increases significantly: +2.00 pp/alpha [+0.11, +3.87], CI excludes zero. The intervention isn't crossing the harmful/safe boundary — it's intensifying severity within the harmful population. Partial compliance → substantive compliance. Deflection → refusal. The dose-response exists, but at a different measurement level than v2's binary metric captures.

The slope compression mechanism is now fully transparent: v2_yes increases by +38 records from α=0→3, but the number of borderlines that v3 absorbs into "yes" decreases by -29 (because the intervention polarizes borderlines away from the middle). Net v3_yes increase: +8 records. The 88% slope compression (2.30 → 0.46) is entirely explained by this arithmetic. The reclassification rate itself is alpha-stable (42-50%), confirming the compression is driven by polarization, not by v3 behaving differently at different alphas.

The transition matrix is remarkably clean: zero v2-no → v3-yes flips across all 1957 records. Only 2 v2-yes → v3-no flips. All 229 discordant records are v2-borderline → v3-yes — construct expansion, not evaluator error.

### What this changes about my thinking

1. **Anchor 3 gains a third level.** The original framing ("same outputs, different evaluator, different conclusion") was level 1. The mechanism transparency (borderline absorption + polarization) was level 2. The severity-shift finding adds level 3: the evaluator doesn't just change the magnitude of the conclusion — it changes the *type* of conclusion that's detectable. This is a methodological insight about measurement granularity, not just an observation about evaluator disagreement.

2. **The severity-shift claim is fragile.** The substantive_compliance slope CI lower bound is +0.11 — barely excludes zero. A different bootstrap seed could flip this. For the paper, this needs a caveat (marginally significant, single model, single control seed). It's directionally strong but not headline-safe without replication.

3. **V3 specificity still rests on v2 evidence.** The v3 binary slope gap CI includes zero (+0.80 [-2.00, +3.58]). The substantive_compliance gap CI barely excludes zero (+2.72 [+0.02, +5.44]). Scoring seed-0 with v3 would strengthen or weaken both claims and is the highest-value remaining experiment.

### What I will do next

Scoring seed-0 with v3 is the clear next step — it enables paired v2-v3 on control data *and* multi-seed specificity testing. For the paper, the three-level Anchor 3 framing is ready to draft. The severity-shift finding should appear as supporting evidence with explicit caveats, not as a headline claim.

---

## 2026-04-13 (evening)

### What I did

**8. Phase 3 jailbreak pipeline audit — v3 slopes, specificity, and concordance scripts reviewed end to end.** Deep audit of the three analysis pipelines producing Phase 3 jailbreak numbers: `analyze_csv2.py` (v3 slopes), `analyze_csv2_control.py` (v3 specificity comparison), and `analyze_concordance.py` (three-judge concordance). Verified raw data counts, traced normalization chains, checked statistical methodology. Report: [2026-04-13-phase3-jailbreak-pipeline-audit.md](./act3-reports/2026-04-13-phase3-jailbreak-pipeline-audit.md).

### What I expected vs what happened

Expected the v3 pipeline to be structurally sound (matching the v2 pipeline already audited). It is — Wilson CIs correct, OLS slopes verified, concordance joining logic clean. But the audit surfaced two high-severity issues and several moderate ones.

**Main surprise: the v3 slope has no statistical test.** The seed-0 v2 analysis computed bootstrap CIs on the slope (+2.30 [+0.99, +3.58]) and a permutation test (p=0.013). No equivalent was done for v3. The v3 slope (+0.46 pp/alpha) sits under per-point CIs of ±4pp and is almost certainly not significant. The specificity comparison uses a single control seed with no permutation test — "exceeds all random seeds" means "exceeds one seed."

**Second surprise: the concordance has zero control data.** The three-way join requires `seed_N_unconstrained_csv2_v2` directories that don't exist. All 1957 concordance records are H-neuron only. This doesn't invalidate the evaluator-agreement analysis but means we can't check whether agreement differs between conditions.

**Third finding: the delta sign disagreement at alpha=1.5 is noise.** The v3 delta is -0.0006 (0.06pp). All three judges agree the effect is near-zero; they disagree only on which side of zero a noise-level measurement falls.

### What this changes about my thinking

1. **The v3 slope flattening is real evidence but needs reframing.** The slope compression from v2 (+2.30) to v3 (+0.46) on identical model outputs is itself the strongest Anchor 3 evidence: measurement choices change conclusions. But "v3 shows a dose-response" is not supportable without a slope CI. The correct framing: "v3 slope is indistinguishable from zero while v2 slope excludes zero — the measured effect magnitude depends on how you define harmful."

2. **v3 specificity rests on v2 evidence.** The single-seed v3 comparison cannot support a standalone specificity claim. Either score seeds 0 and 2 with v3, or cite the v2 specificity result with a cross-reference noting the evaluator is different.

3. **The comparison_csv2_summary.json was overwritten by the v3 run.** The seed-0 v2 report references that file but it now contains v3 data. Need to archive the v2 version before the stale reference causes confusion.

### What I will do next

Bootstrap CI on v3 slope is the highest-value 30-minute task — it definitively frames the slope as "indistinguishable from zero," which strengthens the Anchor 3 argument. Then assess whether scoring seeds 0+2 with v3 is needed for the paper or whether v2 specificity evidence suffices.


## 2026-04-13 (afternoon)

### What I did

**7. FaithEval slope-difference reporting path — audited end to end.** Reviewed the new FaithEval paired slope-difference pipeline from raw JSONL trajectories through exported site payloads, then wrote the canonical audit: [2026-04-13-faitheval-slope-difference-reporting-audit.md](./act3-reports/2026-04-13-faitheval-slope-difference-reporting-audit.md).

### What I expected vs what happened

Expected the narrow reporting path to be mostly an export/plumbing check. It held up better than that. The pipeline did not just write plausible-looking JSON: the saved slope-difference summaries match the underlying 1000-item trajectories exactly, all 8 random-neuron controls align on the same sample IDs, and the site payload binds the new fields correctly.

The main nuance is interpretive, not numerical. The new matched-readout result is strong enough for the reporting claim — neuron-minus-SAE slope difference **+1.93 pp/alpha [0.94, 2.92]** — but it should not silently replace the older delta-only SAE closure result. The full-replacement SAE comparison shows the committed site-facing readout is much weaker than the neuron intervention; the delta-only audit is still what rules out "it was just reconstruction noise." Those are adjacent claims, not interchangeable ones.

Another useful clarification: the FaithEval random-control result is now stronger in the paired sense, not just in the old "random seeds look flat" sense. Every seed-specific paired slope difference is positive and every seed-specific CI excludes zero. That makes the specificity story cleaner to tell.

### What this changes about my thinking

1. **The FaithEval reporting path is publication-ready at the narrow claim level.** I no longer see it as "probably fine if the export wiring is right." The raw-data recheck and targeted tests make it robust enough to cite directly.

2. **The paper should cite two SAE-related documents, not one.** The new audit owns the paired slope-difference reporting claim; the older SAE audit owns the stronger causal interpretation about reconstruction confounding being eliminated by delta-only steering.

3. **Source-hierarchy discipline matters here.** Without a canonical paired-slope audit, the same numbers would drift across `intervention_findings.md`, the SAE audit, sprint notes, and site prose. This is exactly the kind of duplication that later produces paper inconsistencies.

### What I will do next

Use the new audit as the single source of truth for FaithEval paired slope-difference reporting, and keep broader SAE closure claims anchored to the older delta-only audit rather than re-explaining them from scratch in later docs.

### What I did

**6. StrongREJECT gpt-4o rerun — confound eliminated.** Re-ran the 74-record gold comparison with gpt-4o replacing gpt-4o-mini (Batch API, ~$2). Part of the jailbreak measurement cleanup pipeline. Report: [2026-04-13-jailbreak-measurement-cleanup.md §3](./act3-reports/2026-04-13-jailbreak-measurement-cleanup.md#3-strongreject-gold-rerun).

### What I expected vs what happened

Expected Branch B or C from the decision tree (see cleanup report §3.2). Got Branch C: 78.4% combined accuracy (+4.1pp), only 3 of 19 FNs recovered, all via `refused` flag flip 1→0. This confirms the construct mismatch is rubric-intrinsic, not a model capability issue.

**The surprise was the holdout result.** SR-4o on holdout: 96.0% (48/50) — ties v3 exactly, zero discordant records. The single record that was "the entire empirical basis for v3 > SR on holdout" (`jbb_harmful_43_t1` @ α=0.0, from the [holdout report](./act3-reports/2026-04-12-4way-evaluator-holdout-validation.md) §3) was one of the 3 recovered FNs. The v3-SR gap on clean data is now 0.0pp.

The persistent 16 FNs are revealing: mean convincingness 4.56/5, mean specificity 5.00/5, all `refused=1`. gpt-4o *correctly assesses* the content quality but *still codes the disclaimer as a genuine refusal*. The formula zeros out this signal. This is exactly what the 4-way report (§7) predicted: "the bottleneck is the `refused` flag in the formula, not the model's ability to detect refusal or score content quality."

### What this changes about my thinking

1. **The case for v3 shifts from accuracy to structure.** Binary accuracy advantage evaporates on clean data. v3's value is now: ordinal C/S/V axes for tracking intervention gradients, primary_outcome taxonomy for the dissociation argument, evidence spans for audit, zero FP. These are the paper-writing reasons, not "v3 is 12pp more accurate."

2. **The construct mismatch claim gets stronger.** It was a hypothesis based on formula analysis. Now it's a tested prediction: upgrading the judge changes almost nothing because the bottleneck is architectural. This is a more powerful result for the paper than if SR-4o had improved dramatically.

3. **The holdout report's open question persists.** "Does v3 genuinely outperform on new hard cases it wasn't calibrated on?" — the SR-4o tie on holdout doesn't answer this; it just confirms that both evaluators are equivalent when the cases are easy enough. The 25pp dev gap could be calibration leakage or genuine rubric superiority. We can't tell without new gold labels on new refuse-then-comply responses.

### What I will do next

Proceed with full v3 rescore (post-canary phase of cleanup pipeline). The StrongREJECT result validates v3 as primary evaluator — not because it's more accurate on clean data, but because the construct mismatch in SR means the paper needs v3's richer measurement axes regardless.

---

## 2026-04-12 (late evening)

### What I did

**5. Seed-0 jailbreak negative control — scored and analysed.** Scored the seed-0 random-neuron control (38 random neurons, 500 prompts x 4 alphas) with binary judge then CSV-v2, matching the exact d7 evaluation order of operations. Both evaluators: gpt-4o batch mode, zero failures. Ran `analyze_csv2_control.py` for slopes and comparison, then supplementary bootstrap CIs and permutation testing. Report: [2026-04-12-seed0-jailbreak-control-audit.md](./act3-reports/2026-04-12-seed0-jailbreak-control-audit.md).

### What I expected vs what happened

Expected the control to be flat and the H-neuron slope to be significantly steeper. Both confirmed: control slope -0.47 pp/alpha [-1.42, +0.47], H-neuron slope +2.30 pp/alpha [+0.99, +3.58], slope difference +2.77 [+1.17, +4.42], permutation p=0.013.

**The surprise was a pipeline contamination bug.** `analyze_csv2_control.py` imports `normalize_csv2_payload` from the current v3 evaluator, which silently reclassifies 97.7% of v2's "borderline" records as "yes" via the primary_outcome/intent_match derivation chain. This inflated csv2_yes rates from ~18-24% to ~48-52%, producing a wrong triage verdict ("review_specificity" instead of "specificity_supported"). Fixed with a 4-line schema-version check. This is the exact class of bug the mentor review warned about — "changing the ruler halfway through the argument" — but arising from a code dependency rather than a deliberate metric switch.

Second surprise: baseline discrepancy at alpha=0.0 (control 24.2% vs H-neuron 18.8%, Wilson CIs do not overlap). Both conditions should be no-op at alpha=0. The gap is ~2.1 binomial SEs — marginally significant but explained by between-run sampling noise with temperature=0.7 decoding. The slope comparison is robust to this because it's within-condition.

Third finding: severity polarization. H-neurons push borderline cases to the poles as alpha increases (borderline -73, yes +38, no +35). Random neurons show zero polarization (yes+borderline total unchanged at 245). This qualitative mechanistic signature is absent in the control and enriches the specificity finding beyond the csv2_yes rate.

### What this changes about my thinking

1. **H-neuron jailbreak specificity moves from "pending" to "single-seed confirmed."** The claim is no longer caveated as "benchmark-specific control unscored." It's now "seed-0 specificity confirmed, seeds 1-2 pending." The p-value (0.013) is strong enough to anchor a paper claim with qualification.

2. **The v3 normalization bug is a cautionary tale for the measurement section.** Even analysis code — not just evaluation prompts — can silently corrupt the measurement. This should be noted in the companion measurement note as evidence that end-to-end pipeline verification (not just evaluator validation) is necessary.

3. **Binary judge remains underpowered.** Both conditions show flat binary compliance (~30%). This confirms the MDE analysis: binary at n=500 cannot detect effects < ~6pp. The H-neuron binary Delta of +3.0pp (CI includes zero) is real but sub-MDE. This is direct evidence for the measurement discipline story.

### What I will do next

Seeds 1-2 scoring if needed for multi-seed robustness. v3 + StrongREJECT scoring if it's cheap (per mentor: "add v3 plus StrongREJECT if the pipeline makes that cheap"). Paper drafting takes priority over more scoring.

---

## 2026-04-12 (evening)

### What I did

**4. Holdout evaluator validation.** Removed the 24 calibration-contaminated records (8 prompt IDs × 3 alphas) from the 4-way joined data and recomputed all metrics on the 50-record holdout (17 prompt IDs). Added McNemar's exact test for all 6 pairwise comparisons and prompt-clustered bootstrap 95% CIs. Report: [2026-04-12-4way-evaluator-holdout-validation.md](./act3-reports/2026-04-12-4way-evaluator-holdout-validation.md). Script: `scripts/analysis_holdout_evaluator.py`.

### What I expected vs what happened

Expected the holdout to narrow the v3-SR gap but preserve a meaningful advantage. The narrowing was much more extreme than expected: 12.2pp → 2.0pp, resting on a single discordant record (McNemar p=1.0). v3 still ranks first and has zero solo errors, but the "v3 clearly outperforms" narrative collapses on clean data. The dev set is doubly confounded — contaminated *and* enriched for hard cases by design — so the holdout cannot tell us whether v3 genuinely outperforms on new hard cases it wasn't calibrated on.

### What this changes about my thinking

1. **The paper should not lean on accuracy comparisons between evaluators.** Lead with v3's zero-FP and zero-solo-error properties (robust on holdout) and the structural construct-mismatch argument (mechanistic, from formula analysis). Do not lead with "86.5% vs 74.3%."

2. **The evaluator choice probably doesn't change the main intervention findings.** With all four evaluators above 90% on holdout, the D7 causal-head story likely survives regardless of evaluator. This should be verified by running the control comparison (priority #4 in the original list).

3. **The real gap is at the hard tail — and the hard tail needs new validation data.** 10-15 new refuse-then-comply responses, gold-labeled blind, tested against v3 and SR, would definitively settle whether v3's calibrated advantage transfers. Without that, the advantage is demonstrated on dev data only.

### What I will do next

Priorities 2-7 from the earlier list are unchanged. Priority 1 is now complete.

---

## 2026-04-12

### What I did

Three pieces of work.

**1. 4-way evaluator comparison.** CSV2 v3, CSV2 v2, StrongREJECT, and the legacy binary judge, all evaluated against the 74-record human gold set. Report: [2026-04-12-4way-evaluator-comparison.md](./act3-reports/2026-04-12-4way-evaluator-comparison.md).

**2. Error taxonomy on residual errors.** Mentor-requested manual classification of 10 v3 FN + 4 binary FP into six categories (true evaluator miss, gold/rubric boundary, missing minimum harmful element, disclaimer overweighting, span-localization miss, genuinely ambiguous). Report: [error-taxonomy-v3-fn-binary-fp.md](../error-taxonomy-v3-fn-binary-fp.md).

**3. Mentor feedback integration.** Received detailed mentor review that shifted framing on several axes. Integrated changes into both reports, 4-way comparison, and this log.

### What I expected vs what happened

**4-way comparison:** Expected the v3 advantage to come primarily from pivot detection (T flag). Partially correct — T flag is 7/7 on pivots vs SR 0/7, a useful mechanism sketch at small n. But most TP (33/35) are caught by C/S scoring as `substantive_compliance`. The StrongREJECT diagnosis was sharper than expected: all 19 FN have `refused=1` zeroing the score, despite convincingness 4.58/5 and specificity 4.89/5. A construct mismatch, not a model capability issue.

**Error taxonomy:** The 10 v3 FN split roughly 5:3:1:1 — evaluator weakness (disclaimer overweighting, fictional wrapper blindness, one true miss on the lottery scam), gold-spec boundary disputes, one strong gold re-adjudication candidate, one genuinely ambiguous case. The 4 binary FP split 3:1 — three clear binary overcalls (topic-sensitivity without substance) and one possible gold error in the other direction (bribery guide labeled SAFE that binary and v2 correctly flag).

**Mentor feedback — the biggest surprises:**

1. **Development-set contamination.** The 74-record gold set overlaps with the 24 gold-backed rows used during v3 smoke/calibration. This is the single biggest framing issue I missed — 86.5% is promising dev-set evidence, not clean holdout validation. A true holdout comparison (removing calibration rows, stratified by prompt family) is the highest-priority follow-up.

2. **Paired testing vs overlapping CIs.** Rows are paired evaluations, so CI overlap is the wrong inferential lens. McNemar on discordant counts would strengthen v3's case (p ≈ 0.039 vs binary). But prompt-level clustering (many rows share prompt IDs across alpha) weakens naive row-level certainty. Neither "obviously best" nor "CI overlap, so shrug."

3. **Strategic positioning shift.** CSV2 v3 has earned a real role in the main paper (not exploratory). But it has not earned standalone-paper status. The stronger story: "careful, intervention-aware evaluation changes the scientific conclusion about steering." That slots into the flagship thesis (detector ≠ intervention target) as a measurement section, not a replacement.

4. **Zero FP ≠ broad calibration.** SAFE examples are still responses to harmful prompts. "0 FP" means no overcall in that regime, not evaluator calibration on benign capability or over-refusal datasets.

### What this changes about my thinking

1. **v3 is the main evaluator for the Gemma intervention case study.** Not "strongest by every metric" — rather, best-performing on this curated gold set with the right construct for disclaimer-heavy intervention outputs. StrongREJECT stays as literature-legible comparator, binary as simple baseline.

2. **The residual v3 failures are scientifically meaningful, not just noise.** Two specific failure families: "disclaimer discount" (5 records) and "fictional wrapper blindness" (3 records). These tell us where not to overclaim and suggest a specific improvement path: anchored harmful-element specs, not more prompt tinkering.

3. **Non-monotonic cross-alpha gold labels warrant blind re-adjudication, not confident declarations of error.** Alpha can make outputs stranger, not just more harmful. But when non-monotonicity combines with a content assessment that reads as clear refusal (jbb_harmful_36_t4 α=3.0), the case for re-adjudication is strong.

4. **The consensus-core subset is a robustness appendix, not the primary surface.** Reporting only consensus-core effects removes the cases where v3 is supposed to add value.

5. **The 74-record result validates v3-as-binary-judge, not v3-as-full-structured-framework.** Field-level C/S/V/T auditing (~20-30 rows) is needed before those axes carry headline claims.

### What I will do next

Priority-ordered next actions (per mentor):
1. Re-run evaluator comparison on true holdout (remove 24 calibration rows, stratify by prompt family)
2. Blind-adjudicate the 10 v3 FN + 4 binary FP + 2-4 cross-alpha anomalies
3. Re-run StrongREJECT with gpt-4o (~$5)
4. Score seed 0 control with the same evaluator stack
5. Minimal capability/over-refusal battery
6. Tiny field-level audit of C/S/V/T (~20-30 rows)

---

## 2026-04-11

### What I did

Produced a comprehensive strategic assessment for the BlueDot submission: [2026-04-11-strategic-assessment.md](./act3-reports/2026-04-11-strategic-assessment.md). Synthesized inputs from the full project evidence base, GPT-5.4 pro strategic analysis (both myopic and high-vantage versions), and two rounds of independent of Amp's Oracle review. Assessed the current state of all data assets, running experiments, and the CSV v3 validation gap.

### What I expected vs what happened

Expected the narrow "measurement matters" framing to be the right Sunday paper. After reading the full evidence base against the BlueDot criteria and the GPT-5.4 pro's high-vantage analysis, realized the project has earned a much broader thesis: **"Detection Is Not Enough"** — strong readout performance does not reliably identify useful intervention targets.

The key evidence: SAE features match H-neuron detection quality (AUROC 0.848 vs 0.843) yet produce zero steering under both architectures; probe heads achieve AUROC 1.0 and produce null intervention at every alpha; D7 causal heads (gradient-ranked) succeed where probe heads fail (Jaccard 0.11); ITI improves MC selection (+6.3pp) but harms generation (-7pp to -9pp on bridge); D4 ITI beats H-neurons on TruthfulQA MC while H-neurons win on compliance tasks. These are not scattered results — they are one story about the gap between reading a model and controlling it.

The Oracle review confirmed the thesis is earned in softened form ("detection quality is not a reliable heuristic for intervention selection") but not in absolute form ("detection ≠ intervention" — H-neuron scaling is a counterexample). Strongest reviewer counter-arguments identified: mixing detector families/operators/surfaces; single model; missing D7 random-head control.

### What this changes about my thinking

The project’s real contribution is methodological. It tells safety researchers: if you find features that predict harmful behavior, do not assume you have found intervention targets. Prove it with task-local validation, matched controls, and capability checks. The 4288 L1 artifact, SAE dissociation, probe null, ITI MC/generation split, measurement discipline, and CSV v3 smoke test are all chapters in this one story.
Since we are close to deadline and about to receive further review, I want the title to capture the broadest defensible framing, and the write-up can be refined with additional evidence (D7 random-head control, seed 0 scoring, minimal capability battery). I thought limiting the write up scope to less than 10% of the research effort, would for the sake of a simpler story, miss the bulk of the effort and the recurring theme throughout all the experiments I ran. 

### What I will do next

1. Lock the title: *Detection Is Not Enough: Strong Readouts Often Fail as Steering Targets in Gemma-3-4B-IT*
2. Submit seed 0 jailbreak control (2000 rows, already generated) for batch scoring
3. Begin drafting §3 (Main Result I: matched detection, divergent steering) as the paper's center of gravity

---

## 2026-04-10

### What I did

Ran the first CSV v3 hard-case smoke test on 30 intentionally adversarial records via OpenAI Batch API. Full audit: [2026-04-10-csv2-v3-smoke-test-audit.md](./act3-reports/2026-04-10-csv2-v3-smoke-test-audit.md).

### What I expected vs what happened

Expected v3's architectural improvements (refusal_spans, pivot_span, 5-way primary_outcome) to improve over legacy v2 on the cases where v2 was known-broken (borderlines, pivots, wrapper tags). What actually happened: v3 is *more* conservative than legacy. Gold agreement on the hard-case cohort is 37.5% (9/24), with 13 false negatives where human annotators called HARMFUL but v3 said no. Pivot detection went from bad (1/30 T=true in legacy) to worse (0/30 in v3). C/S/V axes collapsed from legacy's 4-value range into a binary switch: {0,0,null} or {3,3,2}.

The root cause is a single calibration gap: the three few-shot examples don't model Gemma 3 4B's characteristic "refuse-then-educate" pattern. The judge sees heavy refusal framing and concludes "deflection" even when the response provides tactically specific harmful information underneath. This is a fourth archetype not represented in the current prompt. Legacy actually handled this better (calling it "borderline"), though legacy's borderline category had its own downstream problems.

### What this changes about my thinking

V3's *architecture* is correct — spans resolve, validation catches errors, the 5-way taxonomy is more informative than binary. But deploying the current prompt at scale would produce systematically conservative labels on exactly the cases that intervention science cares about. The [v2 root-cause analysis](./jailbreak-csv2/background/csv2-v2-tagging-root-cause-analysis.md) recommendation to "validate v3 on a small slice first" proved its worth: it caught a ~$200 mistake before it happened.

~~The gold labels also need attention: jbb_harmful_34_t0 appears mislabeled SAFE at two alphas where both legacy and v3 flag substantive compliance.~~ **Resolved 2026-04-11**: confirmed gold-label error. α=0.0 and α=1.5 should be HARMFUL (DAN persona bribery guide is substantive harmful content). α=3.0 stays SAFE (clean refusal). Relabel in Phase 2.

### What I will do next

1. Add a 4th few-shot example targeting the refuse-then-educate archetype.
2. ~~Re-audit jbb_harmful_34_t0 gold labels.~~ **Done 2026-04-11**: both α=0.0 and α=1.5 confirmed HARMFUL. See smoke-test audit report §4.
3. Re-run the smoke test with the calibrated prompt.
4. Only if smoke test passes: proceed to full-scale canonical v3 re-evaluation.

---

## 2026-04-09

### What I did

Built three evaluation infrastructure pieces for the upcoming scoring phase. First, the CSV v3 structured jailbreak evaluator — implements the v3 rubric with `refusal_spans`, `pivot_span`, 5-way `primary_outcome` taxonomy (full_compliance, partial_compliance, deflection, refusal, ambiguous), and span validation against actual response text. Second, a three-judge concordance analysis system (`analyze_concordance.py`, `evaluate_csv2_v2.py`) for cross-validating binary judge, CSV-v2 graded, and CSV-v3 structured evaluations against each other and gold labels. Third, a concordance negative control pipeline that orchestrates all three judges for seed-controlled jailbreak runs, designed for the seed 0 control (2000 rows, already generated).

### What I expected vs what happened

Expected the v3 evaluator to be a straightforward rubric translation from the prompt design. The span-validation logic required more care than anticipated: v3 needs to verify that judge-reported spans actually appear in the response text, and the `primary_outcome` taxonomy interacts with the C/S/V axes in ways that required explicit resolution rules for edge cases (e.g., what C/S/V values are consistent with a "deflection" outcome). The concordance system was the largest piece of work — aligning three different judge outputs with gold labels required careful scoping of which labels applied to which conditions, leading to a same-day bug fix for gold-label scope and control output preservation.

### What this changes about my thinking

Having the v3 evaluator and concordance pipeline ready means both pending scoring tasks — the CSV v3 smoke test and the seed 0 jailbreak control — can be executed quickly. The v3 evaluator is untested on real data; the next step is a small validation run before committing to full-scale scoring. The concordance system adds a cross-validation layer that didn't exist before — it will surface cases where different judges disagree, rather than relying on a single judge as ground truth.

### What I will do next

Run the CSV v3 hard-case smoke test on ~30 adversarial records to validate the v3 rubric before scaling. Queue the seed 0 jailbreak control for batch scoring once the pipeline is verified.

---

## 2026-04-08

### What I did

Audited the D7 full-500 directory and wrote the canonical report:
[2026-04-08-d7-full500-audit.md](./act3-reports/2026-04-08-d7-full500-audit.md).

This review recomputed the headline numbers from the row-level CSV2 files, checked prompt-ID parity, traced provenance, verified that the relevant generation/evaluation code paths did not change between the baseline and later runs, and reconciled the full-500 directory contents against both the original `d7_causal_pilot.sh` plan and the trimmed continuation script that actually produced the confirmatory artifacts.

### What I expected vs what happened

Expected a clean full-500 closure of the pilot story: causal vs probe vs control. What actually exists is narrower and more useful, but also less complete. The **trimmed** full-500 run gives a solid three-way benchmark comparison:

- baseline `csv2_yes`: **23.4%**
- L1-neuron comparator: **27.4%**, paired **+4.0 pp** **[+0.6, +7.6]**
- causal locked intervention: **14.4%**, paired **-9.0 pp** **[-12.2, -5.8]**

So the practical answer is clearer than in the pilot: on this benchmark surface, the causal intervention beats both no-op and the incumbent L1 comparator.

But two process gaps matter. First, the full-500 directory still contains an **interrupted** `probe_locked` branch (84/500 rows, no judge/CSV2 outputs), so the pilot's "causal beats probe" story was **not** re-run to completion at full scale. Second, the planned `causal_random_head` negative control was skipped in the trimmed script. That means D7 now has a strong benchmark-level result, but still lacks the selector-specificity control required for a stronger mechanistic claim.

The degeneration story also sharpened. Causal α=4.0 produces **112/500 token-cap hits**, which is real quality debt. But those cap-hit rows are mostly `csv2_no` (97/112) with very low mean harmful payload share (0.019), so the main safety win does **not** look like a truncation illusion. The safer reading is "the intervention helps, but it makes the model weird in a noticeable subset of cases."

### What this changes about my thinking

D7 is no longer "pending confirmatory evidence." It now supports a narrower but defensible claim:

> the locked causal head intervention is a better jailbreak mitigation than both no-op and the current L1-neuron comparator on the canonical 500-prompt D7 surface.

The stronger theory-facing claim remains open:

> the gain is specific to the gradient-ranked head set rather than a generic matched-K perturbation.

That distinction matters for D8. If the sprint needs a practical comparator result, D7 is good enough. If the sprint wants a mechanism-clean claim fit for a paper or a central thesis sentence, the missing random-head control is still the bottleneck.

### What I will do next

Use the new full-500 audit as the D7 canonical reference, update the sprint/context files to point at it, and remove the analysed full-500 entry from `notes/runs_to_analyse.md`. Treat the D7 random-head control as the highest-value optional follow-up if synthesis needs selector specificity rather than just benchmark performance.

---

## 2026-04-07

### What I did

Analysed the D7 causal pilot (pilot100_canonical). Both families — causal (gradient-based head selector) and probe (AUROC-based) — ran to completion on the 100-prompt canonical manifests with correct generation settings. Full audit: [2026-04-07-d7-causal-pilot-audit.md](./act3-reports/2026-04-07-d7-causal-pilot-audit.md).

### What I expected vs what happened

Expected the causal selector to show either a clear win or a clear null. Got something more nuanced. **Causal α=4.0 produces a statistically significant -13pp drop in csv2_yes [CI: -21, -6]** — a real effect. **Probe is null at every alpha** (best: -2pp [-10, +6]). The two selectors pick nearly disjoint heads (Jaccard = 0.11 on top-20), confirming they surface different model components.

Three surprises. First, model degeneration at high alpha is more severe than anticipated: at α=4.0 causal, 24% of samples hit the 5000-token cap; at α=8.0, it's 74%. The non-monotonic bounce-back of csv2_yes at α=8.0 is an artifact of this degeneration, not a real intervention reversal. The harmful_payload_share metric (which captures proportion, not just binary classification) decreases monotonically through all alphas, telling a cleaner story.

Second, the effect is template-concentrated: templates 1 ("Begin with Absolutely") and 2 (fiction framing) carry almost the entire signal (-25pp and -30pp respectively), while template 0 (DAN persona) shows zero effect. This is n=20/template so very noisy, but if it holds at full-500, it means the causal intervention counters instruction-following jailbreaks but not persona-based ones.

Third, the greedy↔canonical cross-validation is mixed. Both greedy and canonical agree that causal dominates probe at their respective best alphas, but they disagree on *which* alpha is best (greedy picks α=8.0, canonical picks α=4.0). The disagreement is explained by the greedy decode's 1024-token truncation hiding the degeneration at high alpha.

### What this changes about my thinking

The pilot result is directionally positive for the causal critique: gradient-based selection outperforms correlational selection on jailbreak severity. But -13pp at n=100 is a pilot signal, not a definitive claim. The full-500 run will narrow the CI to approximately ±7pp (vs ±8pp now), which is enough to confirm or disconfirm the effect.

The degeneration at α≥4.0 is a new consideration for D8. Any intervention that operates in decode-only mode is pushing the model out-of-distribution as alpha grows; the operating range has a practical ceiling that must be reported alongside the effect size. The α=1.0 causal result (-10pp [-17, -4]) is also significant and has zero degeneration, making it a potential conservative alternative.

The probe null is the cleanest result: mass-mean AUROC-ranked heads can perfectly discriminate harmful vs benign prompts (top AUROC = 1.0) yet fail to suppress jailbreak behavior at any alpha. This is consistent with the hypothesis that probe ranking selects heads that *reflect* the refusal decision rather than *cause* it.

### What I will do next

Execute the full-500 confirmatory run with causal locked at α=4.0, probe locked at α=1.0, plus the random-head control and baseline_noop. Baseline_noop generation is already in progress.

---

## 2026-04-06

### What I did

Built a reusable pipeline guard library for GPU pipeline orchestration (`scripts/lib/pipeline.py`) and migrated the D7 orchestrator (`d7_causal_pilot.sh`) to use it, replacing the ad-hoc guard logic with the shared library. Also fixed d7_monitor.sh paths for the canonical directory layout. The D7 pilot100 runs launched on Apr 5 continued overnight; evaluation and judging completed during this session.

### What I expected vs what happened

Expected the pipeline guard extraction to be a quick refactor. The library ended up at ~190 lines with its own test suite (244 lines), handling idempotency guards, stage gating, and failure visibility generically. The migration trimmed 23 lines from the D7 orchestrator but the guard library itself was a bigger piece of work than anticipated. The d7_monitor.sh path fix was a trivial correction exposed by the first real monitoring run.

### What this changes about my thinking

The pipeline guard library means future orchestrators (D5, any Phase 3 runs) won't need to reinvent the same idempotency/guard patterns. The D7 pilot data is now fully generated and judged; the next step is analysis, not infrastructure.

### What I will do next

Analyse the D7 pilot100 results for both causal and probe families. Check alpha-lock decisions, head overlap structure, and template-level heterogeneity before interpreting the full-500 data.

---

## 2026-04-05

### What I did

Built the D7 causal pilot pipeline end-to-end — the sprint's one scoped causal experiment, testing whether gradient-based head selection outperforms the correlational (mass-mean) selector that E0/E1/E2 all used.

The pipeline has five new components: (1) deterministic JBB paired manifest builder (pilot 100 / full 500, disjoint, seed-locked); (2) two new ITI extraction families (`iti_refusal_probe` and `iti_refusal_causal`) sharing a single activation surface but forking at ranking time; (3) causal head ranking via contrastive grad×activation on paired harmful/benign NLL objectives; (4) deterministic jailbreak decoding controls in `run_intervention.py`; (5) a resumable `systemd-inhibit` orchestrator (`d7_causal_pilot.sh`) for staged 100→500 execution. Supporting utilities: alpha-lock with paired sample-ID parity enforcement and tie-break to lower alpha, and paired CSV2 reporting (`csv2_yes`, `C`, `S`, `V`, payload share) with guardrails for missing/errored annotations.

Test coverage: manifest determinism/disjointness/alignment, synthetic causal ranking, refusal-family artifact compatibility, jailbreak decode control defaults/overrides, paired-report parity/locking. Verified with `ruff check`, `pytest`, `ty check`, `shellcheck`, `audit_ci_coverage.py`. Minor fixes: renamed `DummyITIModel.forward` param `x→input_ids` for LSP compliance, added `--api_mode` underscore alias to eval harness, switched hot inference paths to `torch.inference_mode()` to avoid unnecessary gradient tracking overhead during generation.

### What I expected vs what happened

Expected the causal/probe split to require a deep pipeline fork. Instead, the design fell out cleanly: both families share the same extraction tensor and steering artifact contract, diverging only at ranking time. The causal ranker computes ∂log p(benign continuation)/∂z · z and ∂log p(harmful continuation)/∂z · z on the same harmful prompt and ranks by the difference — 2 forward + 2 backward passes per pair. The probe ranker uses the existing AUROC path. Everything downstream (sweep, lock, intervention, reporting) is family-agnostic.

The implementation is execution-ready. Orchestrator stages are idempotent/resumable with failure-visible judge/CSV2 phases (removed silent `|| true` swallowing from earlier drafts).

**Pre-registered framing constraints for D7 results.** Allowed: "non-correlational selector," "gradient-based selector," "proxy for refusal-relevant discrimination under paired objectives," "motivated by GCM's attribution patching but using a simpler single-prompt variant." Not allowed: "we performed attribution patching" (different method — no counterfactual patching), "we identified causal mediators of refusal" (no clean intervention), "we replicated GCM" (different formula, different experimental design).

### What this changes about my thinking

D7 is the sprint's last novel experiment before synthesis. Its outcome shapes D8's central claim in one of two ways. If gradient-based selection outperforms correlational selection on jailbreak severity (CSV2), it validates the critique that mass-mean probes conflate relevant and irrelevant heads — and suggests the ITI generation failures (E0/E1 on bridge, E0 on SimpleQA) might be partially rescued by better selection. If it doesn't outperform, D8 can confidently say the problem is deeper than head selection: the directions themselves are the bottleneck, not the heads carrying them. Either outcome closes the sprint cleanly.

### What I will do next

Execute the D7 pilot run (`scripts/infra/d7_causal_pilot.sh`) and add the resulting run entry to `notes/runs_to_analyse.md` once generation/judging completes.

---

## 2026-04-04

### What I did

The biggest day of Act 3 — four threads that collectively closed the TriviaQA transfer question and built the generation evaluation surface the project was missing.

**Thread 1 — E2 transfer synthesis.** Wrote the formal closure of the TriviaQA-transfer ITI lane: [2026-04-04-e2-triviaqa-transfer-synthesis.md](./act3-reports/2026-04-04-e2-triviaqa-transfer-synthesis.md). Cross-validated every number from the auto-generated E2-B diagnostic against JSON data artifacts, verified the five-level classification logic against actual inputs (`wrong_source_still_likely` holds), and added cross-references between E2-A and E2-B data reports.

**Thread 2 — Bridge benchmark inception (Phase 1).** The E2 closure exposed a gap: all our generation evidence came from SimpleQA, where baseline compliance is 4.6% — too low to distinguish "model lacks the fact" from "steering suppressed the answer." Designed and built the TriviaQA bridge generation benchmark from scratch. Manifest builder with stratified seed-42 sampling (pilot 150 / dev 100 / test 500 / reserve 200, all disjoint). Deterministic + GPT-4o adjudicated two-tier grading pipeline with bidirectional blinded audit protocol. Paired bootstrap analysis. Plan: [2026-04-03-bridge-benchmark-plan.md](./act3-reports/2026-04-03-bridge-benchmark-plan.md).

Ran Phase 1 pilot (α=1.0 neuron-mode baseline, 150 questions) — headroom gate passed at 47% baseline accuracy. Then the grading edge cases forced a calibration sprint: added three high-precision tiers (alias simplification, numeric extraction, guarded reverse containment), fixed a curly-quote/em-dash normalization bug in `normalize_answer`, re-scored the pilot with the calibrated grader (audit agreement 87.5% → 91.9%), and locked the two-metric policy: adjudicated accuracy = primary, deterministic = conservative floor.

Also resolved 32 pre-existing `ty` diagnostics across the test suite and added a `ty` compliance check on commit to prevent regression.

**Thread 3 — Phase 2 dev validation (E0 ITI).** Ran 3 conditions (neuron baseline α=1.0, E0 ITI α=4.0, E0 ITI α=8.0) on the 100-question dev manifest. Full pipeline: GPU generation → GPT-4o batch judge (bidirectional audit) → paired analysis with sample-level flip taxonomy. Report: [2026-04-04-bridge-phase2-dev-results.md](./act3-reports/2026-04-04-bridge-phase2-dev-results.md).

**Thread 4 — E1 extension.** After the E0 results showed damage, hypothesized that E1's gentler perturbation (K=8 instead of K=12, +8pp attempt rate vs E0 on SimpleQA) might avoid the right-to-wrong flip asymmetry. Built `triviaqa_bridge_dev_e1.sh`, ran E1 at α=8.0 on the same dev manifest, and added the comparative analysis to the Phase 2 report (§9).

### What I expected vs what happened

**E2 synthesis confirmed what Apr 3 suggested, but sharpened the mechanism.** The calibration-to-held-out collapse is the sharpest signal: E2-B went from +6.2pp on calibration (n=81) to exactly +0.0pp on held-out (n=655); E2-A from +3.7pp to -0.9pp. E0 survived replication. Selector choice is second-order to data source — E2-A vs E2-B changed K from 40→8, ranking metric, and position policy, but moved held-out MC1 by at most 0.92pp (not significant). E0 vs E2-B (same pipeline, different data) shows a 6.26pp gap. Data source dominates everything.

**Bridge Phase 2 — expected the SimpleQA failure mode (refusal/evasion). Got something mechanistically sharper.** E0 ITI does not improve factual generation on TriviaQA: α=4 is flat (Δ adj = -1pp, CI [-6%, +4%]), α=8 is borderline harmful (Δ adj = -7pp, CI [-14%, 0%], McNemar p=0.096). But the *how* matters more than the *how much*. The dominant failure mode at α=8 is **confident wrong substitution** (5/10 right-to-wrong flips): the model replaces a correct entity with a plausible-but-wrong alternative. Terry Hall → Horace Panter (bassist, not vocalist). Two-up → Nimble Nick. Head size → Brain size. "The Sunday Post" → "The Scotsman" (real Scottish newspaper, wrong one). This isn't refusal or hedging — the truthfulness direction is reshuffling the model's factual distribution, promoting semantically adjacent but factually wrong completions.

The intervention is unambiguously active: 51% of correct responses change surface form, 73% of wrong responses change, mean response length grows 70% (15→26 chars), exact match tier drops from 28 to 14. But this activity is indiscriminate — it perturbs everything, not selectively the wrong answers. Damage concentrates in `qb` (quiz bowl: 45%→18%) and `bt` (brain teasers: 20%→0%), the question types where the model's knowledge is least robust. Straightforward factual categories (bb, sfq, wh) survive.

**E1 extension — expected the gentler artifact to at least reduce damage. It made it worse.** E1 α=8 is formally significant: Δ adj = -9pp, CI [-16%, -3%], McNemar p=0.016. The killer detail: both E0 and E1 produce *exactly* 10 right-to-wrong flips with the same damage profile. The difference is rescue capacity — E1 rescues only 1 wrong-to-right (vs E0's 3). On the 5 confident wrong substitutions (Horace Panter, Brain size, The Scotsman, Miep van Pels), **E1 produces the identical wrong entity as E0**. Fewer heads doesn't mean less damage; it means less perturbation in both directions, which kills rescue capacity while leaving the substitution mechanism untouched.

**Grader — clean.** 0/90 false positives across E0 match audits, 1/30 in E1. Bidirectional audit design validated.

### What this changes about my thinking

Three frame shifts:

1. **"Truthfulness direction" is dataset-specific, not universal.** The E2 synthesis makes this definitive. TriviaQA probes are weaker (AUROC 0.75 vs 0.82), the directions don't capture TruthfulQA-relevant representations, and calibration signals that looked promising on n=81 vanished entirely on held-out. The three-variant ranking (E0 > E1 > E2, each gap significant) confirms data source dominates extraction methodology. This constrains D8's universality claims about ITI mechanism.

2. **ITI's generation failure mode is substitution, not refusal.** This is the day's sharpest insight. We spent a week assuming ITI harms generation by inducing over-cautious hedging (the SimpleQA "I don't know" pattern). The bridge benchmark reveals a different mechanism: the truthfulness direction shifts probability mass among the model's existing completion candidates. When the wrong candidate is a TruthfulQA-style misconception, this helps (MC1 +6.3pp). When the model simply lacks robust knowledge (trivia, quiz bowl), the same shift promotes a related-but-wrong entity. The damage being identical across E0 and E1 proves this is method-level — you can't fix it by choosing a better artifact within the mass-mean ITI framework.

3. **The bridge benchmark fills a real gap.** Productive headroom (47%), clean grading (0% FP), and the paired flip analysis reveals mechanism that aggregate metrics would hide. The cross-alpha stability table (49 hard-wrong, 36 stable-correct, 7 progressive damage, 3 stable rescue) gives the kind of population structure SimpleQA's floor compliance can never provide. This is the generation evaluation surface the sprint was missing.

### What I will do next

TriviaQA transfer is closed on both selection (E2) and generation (bridge Phase 2). Shift to D5 (externality audit) and D7 (causal pilot) per sprint priority. Bridge Phase 3 (test set) is deferred until a candidate intervention actually warrants it — E0/E1 ITI are not those candidates.

---

## 2026-04-03

### What I did

Completed the E2-B diagnostic — the second and final TriviaQA-sourced ITI variant — to determine whether E2-A's null result was a selector problem or a data problem. E2-A used paper-faithful overrides (`val_accuracy` + `last_answer_token`); E2-B uses family-default selectors (`auroc` + `all_answer_positions`). Built the diagnostic pipeline with two frozen comparator sidecars to cleanly decompose the headline result: Sidecar A (E2-B artifact at K=40, α=6.0 vs E2-A) isolates the artifact change with K held fixed; Sidecar B (K=16 vs K=40 on the E2-B artifact) isolates the K change with the artifact held fixed. This avoids conflating multiple simultaneous changes in the headline comparison. Report: [2026-04-03-e2b-triviaqa-familydefault-diagnostic.md](./act3-reports/2026-04-03-e2b-triviaqa-familydefault-diagnostic.md).

Also audited the earlier E2-A results for completeness: [2026-04-02-e2-triviaqa-source-isolated-audit.md](./act3-reports/2026-04-02-e2-triviaqa-source-isolated-audit.md).

### What I expected vs what happened

Expected one of two outcomes: either E2-B's AUROC + multi-position selectors would rescue a weak signal that E2-A's paper-faithful overrides missed (selector-mismatch diagnosis), or it would confirm the null and we could confidently blame the data source (wrong-source diagnosis). Got the latter, decisively.

**E2-A is a clean null.** K=40, α=6.0 produces no detectable effect on any surface — all CIs include zero. The calibration signal (+3.70 pp = 3 extra correct on 81 questions) did not survive held-out evaluation. E3 complementarity gate not met: E1 is active-but-imperfect, E2-A is inert, mixing them won't help.

**E2-B changes the artifact but not the outcome.** Switching selectors changes the selected set dramatically — only 2/46 heads overlap (Jaccard 0.043), locked config shifts from K=40 to K=8. But on held-out n=655:

- Within E2-B (K=8, α=6.0 vs α=0.0): MC1 +0.00 pp [-1.98, +1.98]
- Sidecar A (artifact change, K fixed): MC1 -0.92 pp [-2.75, +0.92] — null
- Sidecar B (K change, artifact fixed): MC1 +0.76 pp [-1.22, +2.75] — null
- SimpleQA: compliance +0.00 pp [-3.00, +3.00] — dead flat

All four E2-B artifact SHA256 hashes are identical (3231a345...), confirming source-isolation. The calibration signal (+6.2 pp on n=81 at K=8) was *stronger* than E2-A's, but still collapsed entirely on held-out data — the clearest evidence that weak calibration signals on n=81 are noise when the underlying effect is absent.

**The head overlap structure is the most interesting finding.** E2-B vs E2-A: Jaccard 0.043 but Spearman ρ 0.914, with direction cosine 1.000 on the 2 shared heads. The two selector policies *agree on which heads are good* (nearly identical full ranking) but *disagree on where to draw the K threshold* (AUROC concentrates more sharply → K=8; val_accuracy has narrower dynamic range → K=40). Selector choice changes K, not direction quality.

Between datasets: E2-B vs E0 shows Spearman ρ 0.527 on all 272 heads (moderate agreement on head quality ordering) but negative direction cosine (-0.351) on the single shared selected head (L11H0). Same computational locations, different — possibly opposing — directions. This is suggestive of TriviaQA and TruthfulQA encoding genuinely different concepts at the head level, though n=1 shared head can't carry that weight alone.

### What this changes about my thinking

The E2-A null was not a selector problem. Both selector policies produce inert interventions from TriviaQA-sourced mass-mean directions on Gemma-3-4B-IT. The three-variant evidence is now complete: E0 (TruthfulQA, paper-faithful) produces a real +6.3pp MC1 effect; E1 (TruthfulQA, modernized) produces a weaker +2.75pp effect; E2 (TriviaQA, both selector policies) produces exactly zero. Data source explains the entire variance.

The "same heads, different directions" pattern — if it holds up with more overlap points — would mean that "truthfulness" is not a single direction in this model's representation space. TriviaQA's "factual recall" and TruthfulQA's "misconception resistance" live in the same computational locations but point in different (possibly opposing) directions. This directly constrains the universality claims in the ITI literature (Li et al. 2023) and narrows what D8 can say about the mechanism.

### What I will do next

Write the E2 transfer synthesis to formalize the closure, then pivot. The TriviaQA-transfer lane consumed four experiment-days (E2-A pipeline + audit, E2-B diagnostic + sidecars) and produced a clean, well-decomposed null. No further TriviaQA ITI variants are warranted. Next priorities: D5 (externality audit) and D7 (causal head selection).

---

## 2026-04-13

### What I did

Ran Bridge Phase 3: the one-shot test-set evaluation (n=500, held-out) of E0 ITI (paper-faithful, K=12, α=8.0) on TriviaQA open-ended factual generation. This was the frozen Phase 2 intervention applied to the untouched test split — no parameters were tuned on this data.

Report: [2026-04-13-bridge-phase3-test-results.md](./act3-reports/2026-04-13-bridge-phase3-test-results.md)

### What I expected vs what happened

Expected the dev-set signal (−7pp adjudicated, CI touching zero) to either sharpen into significance or collapse. **It sharpened.** The test-set delta is −5.8pp [−8.8, −3.0], CI excludes zero, McNemar p=0.0002. Dev-to-test replication is remarkably tight: the point estimate sits within the dev CI, the flip ratio is stable (3.3:1 → 3.1:1), and baseline accuracy is comparable (47% → 45%).

The failure-mode taxonomy scaled well. At dev scale, 5/10 R2W flips were wrong-entity substitution (50%). At test scale, 30/43 are wrong-entity (70%), with a secondary evasion/factual-denial mode emerging at 19% (8/43). The wrong-entity examples are vivid — "Trainspotting" → "Slumdog Millionaire" (same director, wrong film), "Ritchie Valens" → "J.P. Richardson" (same plane crash, wrong victim), "Little Dorrit" → "Bleak House" (same author, wrong novel). The intervention selects from the correct semantic neighborhood but picks the wrong member.

The evasion mode is new at this scale: 8 cases where ITI causes the model to deny well-established factual premises ("He did not complete a Puccini opera" — he did, it's Turandot). This could reflect a "skepticism" component in the truthfulness direction that inappropriately suppresses confident factual assertions.

### What this changes about my thinking

The externality-break claim moves from "partially earned" to **earned**. This was Limitation L5 in the paper draft — the main hedge preventing us from stating the externality claim without qualification. That hedge is now removed.

The failure-mode story is stronger than I expected. The 70% wrong-entity rate at scale, with such vivid same-neighborhood examples, makes the "indiscriminate probability mass redistribution" diagnosis paper-ready. The secondary evasion mode (19%) adds nuance — it's not just redistribution, there's also a skepticism-amplification pathway. Two distinct failure mechanisms, both flowing from the same intervention.

The rescue cases (14/500 = 2.8% of all questions) mirror the damage mechanism: when the probability shift happens to land on the correct answer, it helps. This symmetry is the strongest behavioral evidence that the direction is indiscriminate — it's a perturbation, not an improvement.

### What I will do next

Update Section 5.3 of the paper draft to replace dev-set numbers with test-set numbers and remove the L5 hedge. Update the strategic assessment's earned/not-earned boundary for the externality-break claim.

---

## 2026-04-21

### What I did

Closed Limitation 4 (bridge failure-mode single-rater). Built a dual-rater IRR pipeline for all 57 TriviaQA-bridge discordant test cases (43 R→W + 14 W→R): Rater A (first author, human) plus Rater B (`gpt-4o-2024-11-20` via OpenAI Batch API, zero-shot on the rubric, temperature=0, strict JSON schema). Adjudicated the 2 A/B disagreements under a pre-frozen rule (sha256 + git_commit pinned in the summary). Integrated across main.tex, figures, supplement, site, and tests; regenerated fig3 and `site/data/bridge_phase3.json`.

Then reviewed the pipeline and wrote a standalone analysis report at [`paper/icml/reports/2026-04-21-bridge-irr-review.md`](../paper/icml/reports/2026-04-21-bridge-irr-review.md).

### What I expected vs what happened

Expected the single-rater 70% to shift modestly under dual rating; expected a handful of disagreements including some rule-gap cases. Got raw agreement 55/57 = 96.5% [88.1, 99.0], Cohen's κ = 0.90, Gwet's AC1 = 0.96, and two disagreements — both resolved under clean rule coverage (rule_gap=0/2). The adjudicated R→W shares are 72.1% [57.3, 83.3] wrong-entity substitution, 20.9% evasion/denial, 7.0% dilution, 0.0% formal refusal. Dev calibration with the final Rater B prompt was 13/13 = 100%.

Two incidental findings surprised me. First, the W→R rescues are 14/14 wrong-entity — damage and rescue flow through the same category, consistent with the "indiscriminate redistribution" reading rather than a "truthfulness injection" reading. Second, formal refusal collapsed from 2 to 0 at the R→W level: the legacy taxonomy conflated NOT_ATTEMPTED-graded outputs with explicit-refusal language; the frozen rule's R1 requires explicit refusal phrasing, and neither case qualified.

### What this changes about my thinking

The §4.3 claim ("wrong-entity substitution is the dominant R→W failure mode") is now defensible with interval-qualified quantification rather than only qualitative framing. The paper no longer has a single-rater exposure surface on its most vivid qualitative finding.

The R→W / W→R symmetry reshapes the interpretive story. Both directions are ~100% or ~72% wrong-entity; the paper's reading as a redistribution operator over semantic neighbors gets stronger. This sets up Priority-3 (logprob-margin analysis on the 31 adjudicated substitution cases) as the natural next step — cheap, uses existing data, and would upgrade the behavioral taxonomy to a mechanistic measurement.

### What I will do next

(1) Write a short dev-calibration narrative report documenting the §9 rule/prompt revision (two dev runs, prompt_hash `ab951cf3c3ee0bb7` → `a135761b05bf1533`). (2) Consider running the Priority-3 logprob-margin experiment before deadline — the labeled subset is ready.

---

## 2026-04-22

### What I did

Reviewed the Priority-3 bridge logprob-margin pipeline post-run. The scorer/analyzer/tests (commit `7edabc5`) had already produced `margins.jsonl` and `results.json` for the full 57+200 case set under a precommit-locked analysis plan from 2026-04-21. I audited the code path (teacher-forcing, ITI scope gating, ragged-length handling, permutation/bootstrap setup), walked the precommit decision tree against the actual numbers, and rewrote the precommit file in place as a full post-run analysis at [`paper/icml/reports/2026-04-21-bridge-margin-report.md`](../paper/icml/reports/2026-04-21-bridge-margin-report.md) (sealing the original precommit as Appendix A). Updated the TODO Priority-3 entry, the IRR review's §4.3/§5/§7.2 pointers, the `runs_to_analyse` queue, and the analyzer docstring path.

### What I expected vs what happened

Expected the "clean result" shape the TODO predicted: A (substitution) > B (non-substitution) > D (random) in gold-vs-wrong margin-shift magnitude. What happened: the directional hypothesis A < B is **rejected** (first3 shift A = −10.16 nats [−12.81, −7.60] vs. B = −40.06 [−50.09, −30.37]; ΔA−B = +29.90 nats, one-sided p=1.0). B is ~4× more compressed than A, with the gap partially but not entirely driven by position 0 (B pos0 = −21.74, A pos0 = −3.49; on tokens 2–3 alone B still leads A by ~17 nats). A vs. D survives in the predicted direction but only modestly (ΔA−D = −5.10 [−8.22, −2.16], p=0.0081). C (rescue) cleanly reverses sign as expected (+4.73 [+2.27, +7.25]).

A surprise lurking in the baseline delta: B cases start with a much stronger baseline gold preference (+27.55 nats, 12/12 positive) than A cases (+3.41, 22/31 positive). Non-substitution R→W cases are behaviorally "model knew the answer and emitted a non-entity response anyway"; there is simply more standing margin for ITI to compress.

### What this changes about my thinking

The broad redistribution reading (§4.3 of the IRR review, item 7 in its uncertainty table) is *more* supported than before — the margin-shift signal is present across both R→W cohorts and reverses cleanly on rescues. The narrower substitution-specific mechanism claim the TODO was designed to test is *not* supported: the behavioral substitution taxonomy, though κ=0.90 reliable, does not index a distinct margin-shift signature. The paper's §4 externality claim (72.1% wrong-entity substitution, behavioral) is unaffected; this refines rather than undermines it. Planned paper placement: short supplement paragraph plus §4 one-line footnote, not a main-text mechanistic headline.

The precommit protocol itself deserves a line: the Gate 2 sanity peek had already tempted an endpoint retune; the lock held, and branch (iii) of the decision tree fired as predicted. This is exactly the kind of work researcher-degrees-of-freedom guards are supposed to do, and worth preserving as the pattern for future audits.

### What I will do next

(1) Decide with the authorship whether the margin-shift finding lands in the §4 footnote or a short supplement paragraph; recommend footnote + supplement. (2) Optional short follow-up: regression test that α=0 is a bit-exact no-op (closes the only medium-severity pipeline gap); figure panels that separate pos-0 from pos-1+2 to make the answer-frame story visually legible.
