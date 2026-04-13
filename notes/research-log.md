# Research Log — Closure & Pivot Phase

> Continues from [research-log-iti-artifact-exploration.md](./act3-reports/research-log-iti-artifact-exploration.md) (2026-03-24 to 2026-04-02), which covers the ITI artifact exploration phase: pipeline hardening, calibration sweep, E0/E1/E2 artifact variants, decode-scope ablation, and the bridge from D3.5 to D4.

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
