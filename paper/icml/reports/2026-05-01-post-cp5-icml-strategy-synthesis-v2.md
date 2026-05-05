# Post-CP5 ICML Strategy Synthesis v2

Canonical v2, 2026-05-01. Supersedes the historical strategy inputs now parked
under `notes/research-directions/`: R1
`2026-05-01-mistral24b-post-cp5-research-directions-r1.md`, R2
`2026-05-01-post-cp5-research-direction-triage-r2.md`, and v1
`2026-05-01-post-cp5-icml-strategy-synthesis-v1.md`.

## Bottom Line

The ICML path is a claim-defense and framing path, not a new-experiment path.
Mistral 2501 is a clean `readout-positive, intervention-null` stress test:
CP3 found a sparse held-out readout with test AUROC 0.8711 [0.8185, 0.9172],
but CP5 had alpha 0.0 -> 3.0 endpoint 0.0 pp [-4.01, +4.00] with 9
true-to-false and 9 false-to-true flips, and H1 stayed flat at +0.5 pp
[-3.0, +4.0] after selecting C=0.75 with 9 positive H-neurons
(`paper/icml/reports/2026-04-29-mistral24b-cp23-pipeline-review.md`:12-24,
107-124; `notes/icml/mistral24b/2026-04-28-5.5-pro-l1-mitigation-strategy.md`:60-63;
`paper/icml/reports/2026-04-30-mistral24b-cp5-faitheval-review.md`:72-83;
`paper/icml/reports/2026-04-30-mistral24b-h1-c-sweep-review.md`:134-145,
210-228).

Default allocation is zero additional pre-ICML GPU/API for new claim-bearing
runs. The one exception is not a recommendation but a live-state constraint:
`active-run-status` found a running SIMID prospective effect grid under
`run_simid-20260501_095303-39874`, already targeting the r2 manifest, selected
condition, five random-head controls, five random-direction controls, and alpha
grid [-8.0, 0.0, 4.0, 8.0] on `cuda:0`
(`.git/h-neurons-active-runs/run_simid-20260501_095303-39874.json`:2-45,
49-81; `data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/prospective_effect_calibrated_open_20260429/run_config.json`:13-31,
54, 151-152). Do not inspect or analyze those outcomes for the ICML draft; let
the run finish only as already-committed work, and do not stage or restructure
its output path while the live lock exists.

The highest-value deliverable is a reviewer-defense appendix plus a short
limitations rewrite: Gemma remains the primary causal comparison; Mistral is a
failed-gate stress test showing that good held-out readouts do not automatically
license steering claims. SIMID remains diagnostic historically, claimable only
under the prospective external-label contract; the probe-pivot sister plan
remains conditional post-ICML.

2026-05-03 update: the live SIMID grid has since completed, and the partial
selected/TruthfulQA alpha 0 versus 8 Opus label package was reviewed in
[2026-05-03-simid-prospective-partial-external-label-review.md](./2026-05-03-simid-prospective-partial-external-label-review.md).
That early look does not change the allocation: it misses the primary
open-correctness gate (+3.96 pp, 95% CI [-1.98, +9.91]), shows same-scope MC
degradation (-3.08 pp [-5.95, -0.66]), and remains partial/non-claim-bearing.
Do not promote SIMID into the ICML claim set from this evidence.

## Corrections to v1

| v1 statement | Verdict | Corrected v2 position |
|---|---|---|
| "Spend zero pre-ICML" as an unqualified current-state claim (`notes/research-directions/2026-05-01-post-cp5-icml-strategy-synthesis-v1.md`:24-34). | Stale as of v2; updated 2026-05-03. | Zero **additional** spend remains the recommendation. The SIMID prospective grid that was live when v2 was written has completed, and the partial selected/TruthfulQA label review missed the gate. Treat SIMID as containment/diagnostic, not as a new ICML direction. |
| Patch `scripts/run_negative_control.py::_classify_triage` (`notes/research-directions/2026-05-01-post-cp5-icml-strategy-synthesis-v1.md`:71-88). | Wrong symbol, right idea. | The implemented function is `negative_control_triage`, not `_classify_triage`; the existing branch only catches alpha-0-outside / alpha-3-inside baseline mismatch and otherwise returns `specificity_supported` (`scripts/run_negative_control.py`:1288-1329). Add `review_constant_offset` before the final return, using same-side, similar-magnitude endpoint offsets relative to control bands. |
| `review_constant_offset` can be copied from the audit as written. | Underspecified. | The audit's pseudocode is coherent, but the robust implementation should compare offsets from each endpoint's control-band midpoint, not just raw H alpha-0 vs alpha-3 rates. Use `eps = 1.0` pp as the audit proposes for n=200 (`paper/icml/reviews/2026-04-30-mistral24b-cp5-pipeline-audit.md`:255-277). Add regression tests beside the existing triage tests in `tests/test_run_negative_control.py`; if code is changed, run `ruff check scripts`, `ruff format scripts`, and `ty check` per repo hooks (`.pre-commit-config.yaml`:17-33). |
| The 203-row exact-metric crosswalk alone supports a claim -> metric artifact -> gate -> caveat appendix (`notes/research-directions/2026-05-01-post-cp5-icml-strategy-synthesis-v1.md`:71-88). | Overclaim. | The crosswalk supports claim/surface -> metric artifact/source locator: 4,836 rows total, 203 `exact_metric`, 0 unresolved (`notes/ground-truth/README.md`:21-29). Its rows carry locator/provenance keys, not explicit gate/caveat fields (`notes/ground-truth/surface_crosswalk.jsonl`:1). Gate and caveat columns must be manually joined from report prose, `paper/icml/main.tex` limitation rows, and `docs/quantitative-reporting-standards.md`. |
| SIMID mini-slice is a rankable fresh direction (`notes/research-directions/2026-05-01-post-cp5-icml-strategy-synthesis-v1.md`:107-121). | Obsolete. | The r2 package already pre-specifies the full prospective grid and the live run is executing that grid. A smaller mini-slice is mechanically possible with `scripts/run_simid.py`, but it would be a protocol variant, not the frozen r2 package (`data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/mvp_20260427_calibration/human_review_package/prospective_effect_run_gate_20260429_r2_external_labels/protocol.md`:119-134; `data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/prospective_effect_calibrated_open_20260429/run_config.json`:13-31). |
| R2's diagnostic SIMID effects can be repeated as +12 pp TruthfulQA, +3 pp pooled, -6 pp Bridge. | Partly stale. | After exact-propagation sensitivity, TruthfulQA alpha=8 remains +12.0 pp, pooled alpha=8 moves from +3.0 pp to +2.0 pp, and Bridge alpha=8 moves from -6.0 pp to -8.0 pp; still diagnostic only (`paper/icml/reports/2026-04-28-simid-open-calibration-review.md`:51-57). |
| Add a Mistral sentence to the abstract as the default reframing (`notes/research-directions/2026-05-01-post-cp5-icml-strategy-synthesis-v1.md`:157-185). | Too collision-prone. | The abstract and study orientation currently say "We test this assumption in Gemma-3-4B-IT" and define a single-model Gemma case study (`paper/icml/main.tex`:68-75, 112-120). Put Mistral in Section 6/Appendix limitations by default; only add an abstract sentence if Section 1 is edited in the same patch. |
| Bridge human IRR should be Rank 2 critical work. | Overweighted. | The existing bridge audit already reports 55/57 agreement, kappa=0.90, AC1=0.96, 96.5% [88.1, 99.0], and explicitly labels it an LLM second-rater sensitivity rather than human-human IRR (`paper/icml/reports/2026-04-21-bridge-irr-review.md`:3-21, 197-203). A 20-case human-human check is useful but optional supplement work, costed at about 45 minutes and 0 API (`paper/icml/reports/2026-04-21-bridge-irr-review.md`:251-258). |

## Arbiter Verdicts

| Topic | Ground-truth verdict | Arbitration |
|---|---|---|
| Mistral evidence integrity | CP2/CP3 passed a held-out readout gate: disjoint train/dev/test, test AUROC 0.8711 [0.8185, 0.9172], accuracy 0.775 [0.715, 0.830], F1 0.7783 [0.7184, 0.8349] (`paper/icml/reports/2026-04-29-mistral24b-cp23-pipeline-review.md`:12-24, 119-126). CP5 contracts passed, parse failures were 0/200, endpoint 0.0 pp [-4.01, +4.00], flips 9/9, with constant H-control offset (`paper/icml/reports/2026-04-30-mistral24b-cp5-faitheval-review.md`:47-57, 72-83). H1 selected C=0.75 but stayed +0.5 pp [-3.0, +4.0] with 7 false-to-true and 6 true-to-false flips (`paper/icml/reports/2026-04-30-mistral24b-h1-c-sweep-review.md`:134-145, 210-228). | Call it `readout-positive, intervention-null`. Do not call it a failed replication of the full paper; do call it a failed H-neuron FaithEval intervention gate on Mistral-Small-24B-Instruct-2501. |
| CP5 pipeline/audit | Independent recompute matched every CP5 headline number; controls flipped 2/1600 alpha-0-to-alpha-3 items while H flipped 18/200, balanced 9/9 (`paper/icml/reviews/2026-04-30-mistral24b-cp5-pipeline-audit.md`:30-63, 220-236). | Pipeline is sound enough to publish the null. The weakness is script triage, not artifact integrity. |
| SIMID claimability | Production calibration had 402 cases, 368/402 = 91.5% [88.4, 93.9], kappa 0.7594, AC1 0.8974, and failed the 0.8 kappa gate; historical metrics remain diagnostic (`paper/icml/reports/2026-04-28-simid-open-calibration-review.md`:11-29). Prospective Opus passes 138/150 = 92.0% [86.5, 95.4], kappa=0.8734, AC1=0.8831, 0 rule gaps, but future effect claims require complete external blind labels (`paper/icml/reports/2026-04-28-simid-open-calibration-review.md`:65-93; `data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/mvp_20260427_calibration/human_review_package/prospective_effect_run_gate_20260429_r2_external_labels/protocol.md`:52-60, 69-97). The 2026-05-03 partial selected/TruthfulQA Opus package missed the primary open gate and showed MC degradation (`paper/icml/reports/2026-05-03-simid-prospective-partial-external-label-review.md`). | Historical SIMID is diagnostic-only. The r2 grid no longer looks like a likely ICML rescue; no ICML claim should depend on it. |
| Probe-pivot | Integrated plan is "conditional go"; pre-SIMID work may be label-free or outcome-blinded, including activation caching, probes, readout margins, ITI projections, predictor table with blinded prompt IDs, and pseudo/permuted-label validation, while forbidding outcome/generation inspection (`notes/research-directions/2026-04-30-claude-opus-4-7-probe-pivot-sister-paper-plan.md`:28-49, 355-369). Go requires claim-bearing SIMID variation, predictor dynamic range, and preregistration before outcome inspection (`notes/research-directions/2026-04-30-claude-opus-4-7-probe-pivot-sister-paper-plan.md`:380-405). | Pre-ICML work may freeze schema/prereg fields only if it stays blind. Defer GPU/predictor execution because of attention and active-run risk, not because label-free work is intrinsically leakage. |
| Gemma appendix feasibility | The ground-truth pack has 203 `exact_metric` rows and 0 unresolved crosswalk rows (`notes/ground-truth/README.md`:21-29). Gemma FaithEval H effects are already summarized: +6.3 pp [4.2, 8.5] alpha 0-to-max, +4.5 pp [2.9, 6.1] no-op-to-max, +2.09 pp/alpha [1.4, 2.8] (`notes/ground-truth/metric_tables_only.md`:49-61; `data/gemma3_4b/intervention/faitheval/experiment/results.json`:201-240). SAE is summarized as slope +0.16 pp/alpha and neuron-minus-SAE slope +1.93 pp/alpha [0.9, 2.9] (`notes/ground-truth/metric_tables_only.md`:81-87; `data/gemma3_4b/intervention/faitheval_sae/experiment/results.json`:204-239). | Feasible with manual caveat/gate augmentation. Paired endpoint/flip summaries are derivable from existing per-alpha JSONL, but an audit-ready checked-in flip table is the smallest missing artifact if the appendix needs flip texture for Gemma H and SAE. |
| Manuscript collision | The current abstract, Section 1, Section 6, and Appendix limitation table are Gemma-first: single-model Gemma case study; all claims surface-specific; L1 is "Single model (Gemma-3-4B-IT); all claims" (`paper/icml/main.tex`:68-75, 112-120, 419-428, 622-640). | Do not imply Mistral mitigates all single-model concerns. Use it to sharpen L1: same-family Mistral readout exists, but intervention fails under this operator/prompt/checkpoint. |

## Portfolio Allocation

| Horizon | Workstream | Allocation | Rationale and guardrail |
|---|---|---:|---|
| Pre-ICML | Rank 1 claim-defense bundle | 60% writing/engineering attention, 0 new GPU/API | Patch or at least specify `negative_control_triage`; build appendix tables from existing artifacts; verify every manuscript number against ground-truth ledgers. This directly defends the current claims and fixes a real script-vs-manual gate divergence (`scripts/run_negative_control.py`:1288-1329; `paper/icml/reviews/2026-04-30-mistral24b-cp5-pipeline-audit.md`:255-277). |
| Pre-ICML | Manuscript reframing | 25%, 0 GPU/API | Revise Section 6/L1 and appendix wording only. Keep Gemma as the primary empirical study and Mistral as failed-gate stress evidence (`paper/icml/main.tex`:419-428, 622-640). |
| Pre-ICML | SIMID containment | 10%, no new launch | 2026-05-03 update: the r2 grid has completed and the partial selected/TruthfulQA label review missed the gate. Do not promote, backfill, or launch more SIMID work for ICML without a new pre-registered diagnostic question. |
| Pre-ICML | Optional bridge human-human sensitivity | <=5%, 0 GPU/API | Only if an external human is immediately available; supplement-only and non-blocking (`paper/icml/reports/2026-04-21-bridge-irr-review.md`:251-258). |
| Post-submission | SIMID r2 analysis | Conditional | Only after external blind labels are complete and the r2 authority/gates are recorded (`data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/mvp_20260427_calibration/human_review_package/prospective_effect_run_gate_20260429_r2_external_labels/protocol.md`:52-60, 137-142). |
| Post-submission | Probe-pivot sister paper | Conditional | Start only if SIMID r2 has claim-bearing heterogeneity and the prereg/predictor schema was frozen blind (`notes/research-directions/2026-04-30-claude-opus-4-7-probe-pivot-sister-paper-plan.md`:247-260, 380-405). |

No recommendation here depends on current dollar pricing. The repo has durable
budget evidence for Mistral only: a $300 ceiling excluding judge/API and human
rater costs, with CP5/H1 already having consumed the primary FaithEval/C-sweep
spend (`notes/icml/mistral24b/2026-04-28-5.5-pro-l1-mitigation-strategy.md`:121-123,
242-244). Any future dollar budget should be rechecked at execution time and
recorded in the run plan; this synthesis uses calibration-pass-equivalent and
live-run-state constraints instead of uncited price estimates.

## Ranked Direction List

| Rank | Direction | Decision | Concrete next move |
|---:|---|---|---|
| 1 | Claim-defense appendix and triage patch | Do now. | Implement `review_constant_offset` in `negative_control_triage` or file a patch-ready issue with exact predicate: both endpoint H rates outside their corresponding random percentile bands, same sign relative to each band midpoint, and offset difference <= 1.0 pp; return before `specificity_supported`. Add regression tests in `tests/test_run_negative_control.py`. If code changes, run `ruff check scripts`, `ruff format scripts`, and `ty check` (`scripts/run_negative_control.py`:1288-1329; `.pre-commit-config.yaml`:17-33). |
| 2 | Gemma claim-defense appendix | Do now. | Table columns: claim, exact metric ID/source artifact, paired estimator/CI method, gate status, caveat. Use `notes/ground-truth/metric_ledger.jsonl` and `surface_crosswalk.jsonl` for source anchoring; manually add gates/caveats from reports and `paper/icml/main.tex` limitation rows because the crosswalk lacks those fields (`notes/ground-truth/README.md`:9-12, 21-29; `notes/ground-truth/surface_crosswalk.jsonl`:1; `paper/icml/main.tex`:622-640). |
| 3 | ICML Mistral reframing | Do now, no new run. | Add an appendix/Section 6 paragraph: Mistral 2501 has a held-out readout but no FaithEval steering endpoint under AUROC-selected or intervention-aware C selection. Include exact caveats: 2501 not 2503, FFN positive-weight scaling, standard FaithEval prompt, reserve-200 H1 proxy (`paper/icml/reports/2026-04-30-mistral24b-h1-c-sweep-review.md`:289-301). **Could be outdated; verify before acting.** |
| 4 | SIMID r2 handling | Contain, do not promote. | The r2 grid has completed, but the partial selected/TruthfulQA external-label review missed the primary gate and showed MC degradation. Full claim-bearing open correctness would still require canonical complete external labels and control/Bridge gates (`data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/mvp_20260427_calibration/human_review_package/prospective_effect_run_gate_20260429_r2_external_labels/protocol.md`:52-60, 69-97; `paper/icml/reports/2026-05-03-simid-prospective-partial-external-label-review.md`). **Could be outdated; verify before acting.** |
| 5 | Optional bridge human-human sensitivity | Optional. | If an external human rater is available immediately, run the 20-case stratified supplement check; otherwise leave L4 as the current LLM-second-rater sensitivity, already disclosed (`paper/icml/reports/2026-04-21-bridge-irr-review.md`:197-203, 251-258). **Could be outdated; verify whether bridge human labeling has already been completed before acting.** |
| 6 | Probe-pivot prereg/schema | Defer beyond ICML unless it is paperwork-only. | It is label-free if it freezes cell IDs, split hashes, blinded prompt IDs, purpose-probe AUROC, frozen-ITI signed-projection AUROC, AUROC gap, nuisance features, and random-label controls without open-label/outcome/generation columns. Do not compute or join outcomes pre-ICML (`notes/research-directions/2026-04-30-claude-opus-4-7-probe-pivot-sister-paper-plan.md`:355-369). |

Explicit rejections for the next two weeks:

| Rejected direction | Reason |
|---|---|
| New Mistral SAE branch, Mistral bridge/ITI branch, CP5 rerun, or 2503 migration before ICML | No prior evidence now makes these informative enough to justify new claim-bearing GPU. CP5/H1 already falsified the current "readout -> FaithEval steering" continuation path (`paper/icml/reports/2026-04-30-mistral24b-cp5-faitheval-review.md`:94-98; `paper/icml/reports/2026-04-30-mistral24b-h1-c-sweep-review.md`:289-301). |
| Fresh SIMID mini-slice as Rank 3 | Obsolete and protocol-dirty relative to the r2 package; full grid is already running under the frozen manifest/conditions (`data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/mvp_20260427_calibration/human_review_package/prospective_effect_run_gate_20260429_r2_external_labels/protocol.md`:119-134; `data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/prospective_effect_calibrated_open_20260429/run_config.json`:13-31). |
| gpt-4o as claim-bearing SIMID adjudicator | The r2 manifest explicitly marks gpt-4o diagnostic-only; complete external blind labels are required (`data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/mvp_20260427_calibration/human_review_package/prospective_effect_run_gate_20260429_r2_external_labels/effect_run_manifest.json`:69-77, 175-190). |
| Historical SIMID retrofit | Production calibration failed kappa < 0.8, and exact propagation remains diagnostic-only (`paper/icml/reports/2026-04-28-simid-open-calibration-review.md`:11-29, 51-57). |
| Probe-pivot outcome join or broad C3/cross-operator story | The integrated plan is measurement-first C2 and conditional; broad "readable directions need not steer" novelty is crowded (`notes/research-directions/2026-04-30-claude-opus-4-7-probe-pivot-sister-paper-plan.md`:28-49, 117-128). |
| Crosswalk-only appendix without manual caveats | The crosswalk does not contain gate/caveat fields (`notes/ground-truth/surface_crosswalk.jsonl`:1). |

## ICML Reframing

Do not change the manuscript's center of gravity. The existing paper is a
Gemma-3-4B-IT comparative case study across H-neurons, SAE features, ITI, and
measurement surfaces (`paper/icml/main.tex`:90-120). Mistral should be a
limitations/appendix stress test, not a new headline model axis.

Recommended Section 6 / L1 wording:

> All primary causal, cross-representational, transfer, and measurement claims
> remain about Gemma-3-4B-IT. As a same-family stress test, we also audited
> Mistral-Small-24B-Instruct-2501 on the H-neuron FaithEval path. The audit found
> a sparse held-out readout (test AUROC 0.8711 [0.8185, 0.9172], 10 positive
> H-neuron targets) but no FaithEval endpoint movement under either the
> AUROC-selected C=1.0 classifier or an intervention-aware C=0.75 selector
> (0.0 pp [-4.01, +4.00] and +0.5 pp [-3.0, +4.0], respectively). We therefore
> treat Mistral as a failed-gate stress test: it shows that readout quality alone
> is insufficient for this operator/prompt/checkpoint, but it is not a
> cross-model replication of the Gemma steering result and does not address SAE,
> ITI bridge, exact 2503, prompt, or operator-transfer claims.

Source anchors for that paragraph are CP3, CP5, H1, and the exact-checkpoint
caveats (`paper/icml/reports/2026-04-29-mistral24b-cp23-pipeline-review.md`:12-24,
107-124; `paper/icml/reports/2026-04-30-mistral24b-cp5-faitheval-review.md`:72-98;
`paper/icml/reports/2026-04-30-mistral24b-h1-c-sweep-review.md`:289-301).

Abstract default: no Mistral sentence. If coauthors insist, edit the abstract and
study-orientation paragraphs together so the "We test this assumption in
Gemma-3-4B-IT" sentence does not become false (`paper/icml/main.tex`:68-75,
112-120). The lower-risk abstract sentence would be:

> A same-family Mistral stress test found a strong held-out readout but a null
> steering endpoint, reinforcing the need to report readout, control, and
> externality gates separately.

Use that only if Section 1 explicitly frames it as an appendix stress test, not
as a second-model replication.

## Two-Week Decision Tree

| Trigger | Decision | Action |
|---|---|---|
| Rank 1 appendix/triage numbers all verify from existing artifacts | Proceed with manuscript/reporting patch. | Add appendix tables and L1/Section 6 reframing; no new GPU. |
| `review_constant_offset` patch cannot be implemented cleanly | Do not block manuscript. | Document the manual gate and smallest code gap: `negative_control_triage` lacks a same-side constant-offset branch (`scripts/run_negative_control.py`:1288-1329). |
| Gemma paired flip table is needed but absent | Create the smallest non-claim-running artifact. | Derive from existing per-alpha JSONL under `data/gemma3_4b/intervention/faitheval*/experiment/`; no model run. If the appendix only needs endpoints/CIs, current `results.json` and metric ledgers suffice (`data/gemma3_4b/intervention/faitheval/experiment/results.json`:201-240; `data/gemma3_4b/intervention/faitheval_sae/experiment/results.json`:204-239). |
| SIMID grid finishes before ICML | Do not promote from partial labels. | The completed-grid partial label review missed the gate; claim-bearing analysis would still require canonical complete external blind labels and r2 authority (`paper/icml/reports/2026-05-03-simid-prospective-partial-external-label-review.md`). |
| SIMID external-label work stalls or remains partial | Freeze it out of ICML. | Do not start diagnostic adjudication or new label spend for claim rescue; keep historical SIMID diagnostic-only. |
| Opus prospective rubric drift or external labels disagree | No ICML upgrade. | Treat as measurement evidence; do not retrofit historical MVP or use gpt-4o as authority (`data/gemma3_4b/intervention/simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/mvp_20260427_calibration/human_review_package/prospective_effect_run_gate_20260429_r2_external_labels/effect_run_manifest.json`:69-77). |
| Bridge second-human check is partially completed | Supplement-only. | Include only if the protocol and denominators are clean; otherwise keep current L4 disclosure. Existing L4 is already explicit (`paper/icml/main.tex`:425-426, 635-639). |
| Coauthors disagree on Mistral framing | Prefer Section 6/Appendix over abstract. | The manuscript already commits to Gemma-first wording; avoid an abstract collision unless Section 1 is updated too (`paper/icml/main.tex`:68-75, 112-120). |
| Coauthors want another Mistral run | Require a new pre-registered question. | It must name prior evidence that makes the run informative after CP5/H1 nulls; "rescue the story" is not enough. |
| Probe-pivot pressure rises pre-ICML | Permit paperwork-only freeze. | Schema/prereg may proceed if it has blinded IDs and no outcome/generation columns; all predictor/outcome joins wait for SIMID r2 claimability (`notes/research-directions/2026-04-30-claude-opus-4-7-probe-pivot-sister-paper-plan.md`:355-405). |

## Risks and Guardrails

| Risk | Guardrail |
|---|---|
| Active-run corruption | Run `uv run python -m scripts.lib.pipeline active-run-status` before staging or restructuring any output path; pre-commit has `active-run-git-guard` enabled (`data/AGENTS.md`:5-16; `.pre-commit-config.yaml`:10-15). |
| Outcome leakage from SIMID/probe-pivot | Keep SIMID r2 outcomes and generations out of ICML work. Probe-pivot schemas must use blind prompt IDs and no outcome-derived columns (`notes/research-directions/2026-04-30-claude-opus-4-7-probe-pivot-sister-paper-plan.md`:355-369). |
| Mistral overclaim | Say 2501, not 2503; say H-neuron FaithEval operator/prompt/checkpoint; say failed-gate stress test, not broad model replication (`paper/icml/reports/2026-04-30-mistral24b-h1-c-sweep-review.md`:294-301). |
| Crosswalk overtrust | Treat `surface_crosswalk.jsonl` as a source-locator map only; gate/caveat text requires manual review (`notes/ground-truth/README.md`:9-12, 21-29; `notes/ground-truth/surface_crosswalk.jsonl`:1). |
| Script triage silently green-lights future CP5-shaped runs | Add `review_constant_offset` to `negative_control_triage` and test it against a CP5-like summary (`scripts/run_negative_control.py`:1288-1329; `paper/icml/reviews/2026-04-30-mistral24b-cp5-pipeline-audit.md`:255-277). |
| Abstract/manuscript contradiction | Do not insert Mistral into the abstract without editing the study-orientation paragraph and L1 row in the same patch (`paper/icml/main.tex`:68-75, 112-120, 622-640). |

## Blind Spots and Hidden Alternatives

| Alternative | v2 treatment |
|---|---|
| Second-judge audit for Gemma FaithEval/SAE rows | Not a priority: FaithEval is parser/deterministic-label based, not an open-judge endpoint. The useful audit is paired endpoint/flip texture for H and SAE from existing JSONL, plus parser-failure accounting already in `results.json` (`data/gemma3_4b/intervention/faitheval/experiment/results.json`:19-30, 187-199; `data/gemma3_4b/intervention/faitheval_sae/experiment/results.json`:22-33, 190-201). |
| CP5 control pairing relative to Gemma claims | Worth adding to the appendix as a contrast: Mistral controls are nearly inert and H flips are balanced; Gemma H has a paired positive endpoint and control slope separation (`paper/icml/reviews/2026-04-30-mistral24b-cp5-pipeline-audit.md`:220-236; `notes/ground-truth/metric_tables_only.md`:49-61). |
| Externality re-audit | Optional only through the 20-case human-human supplement. The existing LLM-second-rater bridge audit is already disclosed and strong enough for the current claim scope (`paper/icml/reports/2026-04-21-bridge-irr-review.md`:3-21, 197-203, 251-258). |
| Reviewer-defense rebuttal pack | High value as a derivative of Rank 1: one page mapping each likely criticism to artifact, gate, and caveat. No new experiment. |
| Abstract A/B against stripped-Mistral variant | Useful for coauthor alignment: default A keeps Mistral out of the abstract; B includes the stress-test sentence only if Section 1 changes. |
| Fresh prior-art reaction | Probe-pivot prior-art already downgraded the broad novelty claim; no need for more pre-ICML literature search unless a reviewer-facing sentence depends on a new citation (`notes/research-directions/2026-04-30-probe-pivot-prior-art-discovery.md`:186-234). |

## Supersession

This v2 should be treated as the canonical strategy synthesis. R1, R2, and v1
remain useful historical inputs, but all current routing, allocation, and
recommendation decisions should point here. The authoritative experiment ledgers
remain the Mistral ledger, SIMID calibration/effect-gate package, probe-pivot
integrated plan, ground-truth ledgers, and `paper/icml/main.tex`; this report
does not edit or supersede those canonical sources.
