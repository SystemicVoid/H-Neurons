# Research-Grade Draft Audit

Date: 2026-04-16  
Scope: [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:1), source shards under `paper/draft/`, figure scripts under `paper/draft/figures/`, and current canonical evidence under `notes/act3-reports/` and `data/`  
Review mode: findings-first, publication-rigor audit  
Status: canonical merged review and tracking memo

## Canonical Status

This file is the single review document to use for implementation and status tracking. It now incorporates the independent verification pass that had temporarily been written to a separate memo. That duplicate surface has been superseded and removed so review state lives here only.

Freshness checks completed on 2026-04-16:

- `uv run python scripts/build_full_paper.py --check` passed.
- `uv run pytest tests/test_build_full_paper.py` passed (`6 passed`).

## Review Status

| Area | Type | Status | What was checked against what | Current conclusion |
|---|---|---|---|---|
| D7 evidence hierarchy | overstated inference | Resolved, guard against regression | Manuscript vs [2026-04-16-d7-full500-two-seed-current-state-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-16-d7-full500-two-seed-current-state-audit.md:1) | FaithEval is now the sole load-bearing anchor; D7 is benchmark-local supporting evidence only. |
| Section 5 bridge interpretation | overstated inference | Open | Manuscript vs [2026-04-13-bridge-phase3-test-results.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-bridge-phase3-test-results.md:1) | Output-level diagnosis is strong; mechanism-closing language is too strong. |
| Figure 4 uncertainty logic | presentation/consistency defect | Open | Figure script vs manuscript text vs holdout validation note | Panel C uses row-level Wilson intervals while the text reports prompt-clustered bootstrap CIs; Panel D is mostly redundant. |
| Figure 2 precision signaling | presentation/consistency defect | Open | Figure script vs manuscript wording | Detection panel visually overstates precision/equivalence. |
| Figure 3 readability | presentation/consistency defect | Open | Figure script vs rendered PNG | Panel C table clips at the right edge in the current render. |
| Jailbreak measurement framing | unsupported claim | Resolved, keep narrow | Manuscript vs [2026-04-13-jailbreak-measurement-cleanup.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-jailbreak-measurement-cleanup.md:1) | Safe claim is tie on binary holdout; prefer CSV-v3 for richer structure, not superiority. |
| BioASQ summary framing | presentation/consistency defect | Resolved, keep narrow | Manuscript vs [bioasq_pipeline_audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/data/gemma3_4b/intervention/bioasq/bioasq_pipeline_audit.md:1) | Safe claim is endpoint-flat but behaviorally active, not a clean null. |
| Citation registry integrity | unsupported claim | Open | [registry.json](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/citations/registry.json:1) vs local `papers/` files | Registry is too corrupted to trust for sync or provenance. |
| Sections 2 and 3 narrative economy | presentation/consistency defect | Open | Reader-facing prose vs evidence spine | Early sections still spend too much budget on governance and novelty fencing. |

## Executive Verdict

The draft now has a coherent evidence hierarchy and a defensible paper-level thesis. The strongest scientific core remains intact: the FaithEval neuron-versus-SAE dissociation, the held-out TriviaQA bridge externality result, and the post-cleanup jailbreak measurement story all survive audit.

The main remaining blockers are narrower and more concrete than before. The top scientific risk is still inferential discipline in Section 5. The top presentation risks are Figure 4's interval mismatch, Figure 2's precision signaling, Figure 3's clipped table, and the corrupted citation registry.

## Did Finding 1 Actually Land?

Yes, substantially.

- FaithEval is now the only load-bearing localization/control anchor in the abstract, introduction, Section 4, synthesis, limitations, and appendix: [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:17), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:29), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:126), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:177), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:307), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:360).
- D7 is now described in the narrow way the April 16 current-state audit supports: benchmark-local supporting evidence that selector choice matters on this surface, not mechanism-clean closure: [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:171), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:173), [2026-04-16-d7-full500-two-seed-current-state-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-16-d7-full500-two-seed-current-state-audit.md:191).
- The old main-text synthesis table is gone; no remaining main-text surface now visually flattens FaithEval and D7 into the same evidential tier.
- Recommendation 1 now stands on FaithEval alone in [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:317).

Residual risk: later edits could easily reintroduce D7 inflation, but D7 is no longer the draft's main blocker in its current form.

## Open Findings

### 1. Critical: Section 5 still overstates mechanism relative to the bridge evidence

Type: overstated inference  
Checked: manuscript vs [2026-04-13-bridge-phase3-test-results.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-bridge-phase3-test-results.md:1)

The bridge result itself is strong and publication-grade. The overreach appears in the explanatory language layered on top of it.

- Reader-facing overreach appears in [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:222), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:224), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:226), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:239), and [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:245).
- Related work already pre-frames the paper as characterizing a probability-mass redirection mechanism in [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:110), which is also stronger than the evidence now warrants.
- What the canonical report clearly supports is that the held-out test result is real, the harm is statistically significant, and wrong-entity substitution is the most frequent manually diagnosed failure mode at `30/43` right-to-wrong flips. The coding is single-rater, so the taxonomy is descriptive and approximate: [2026-04-13-bridge-phase3-test-results.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-bridge-phase3-test-results.md:72), [2026-04-13-bridge-phase3-test-results.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-bridge-phase3-test-results.md:186), [2026-04-13-bridge-phase3-test-results.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-bridge-phase3-test-results.md:197), [2026-04-13-bridge-phase3-test-results.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-bridge-phase3-test-results.md:212).
- What is not cleanly earned is the stronger interpretation that the paper now "explains why" constrained selection helps while open-ended generation worsens, or that the rescues and failures together support an "indiscriminate redistribution" account. Those are plausible hypotheses, not closed conclusions.

Required revision direction: keep the behavioral diagnosis, but recast the explanatory layer as bounded hypothesis. "Consistent with" is earned; "explains why" and "supports an indiscriminate redistribution interpretation" are too strong.

### 2. High: Figure 4 is not fully aligned with the paper text on uncertainty, and Panel D adds load without adding much evidence

Type: presentation/consistency defect  
Checked: [fig4_measurement.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/figures/fig4_measurement.py:1) vs [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:282) vs [2026-04-12-4way-evaluator-holdout-validation.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-12-4way-evaluator-holdout-validation.md:1)

- The manuscript's holdout-evaluator table reports prompt-clustered bootstrap intervals over 17 prompt IDs in [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:282), matching the canonical holdout note: [2026-04-12-4way-evaluator-holdout-validation.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-12-4way-evaluator-holdout-validation.md:87), [2026-04-12-4way-evaluator-holdout-validation.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-12-4way-evaluator-holdout-validation.md:94).
- Figure 4 Panel C instead computes Wilson intervals directly from 50 row-level counts in [fig4_measurement.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/figures/fig4_measurement.py:57), [fig4_measurement.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/figures/fig4_measurement.py:146), and [fig4_measurement.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/figures/fig4_measurement.py:159). That is a direct mismatch for the same displayed quantity.
- Panels B and D are both built from the same graded seed-0 specificity surface. Panel D is not incorrect, but it largely duplicates Panel B with dotted trend lines while dropping the uncertainty shading that makes Panel B interpretable: [fig4_measurement.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/figures/fig4_measurement.py:263), [fig4_measurement.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/figures/fig4_measurement.py:364).

Required revision direction: either make Panel C use the same prompt-clustered intervals as the text or label it explicitly as a different interval method, and cut or repurpose Panel D because it currently increases panel load more than evidential clarity.

### 3. High: The citation registry is corrupted enough to be a submission risk

Type: unsupported claim  
Checked: [registry.json](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/citations/registry.json:1) vs local `papers/` files

The manuscript bibliography is broadly salvageable, but the registry is not trustworthy as a source-of-truth.

- `tan2024steering` points to the Pres paper's local files in [registry.json](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/citations/registry.json:92).
- `pres2024reliable` points to the Bhalla paper's local files in [registry.json](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/citations/registry.json:110).
- `bhalla2024unifying` points to FaithEval files while still saying "Need to acquire" in [registry.json](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/citations/registry.json:128).
- `ming2025faitheval` claims no local files despite local markdown and PDF existing in [registry.json](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/citations/registry.json:182). The files do exist: `papers/faitheval-2410.03727.md` and `papers/faitheval-2410.03727.pdf`.

This is not harmless bookkeeping noise. In current form, `registry.json` cannot safely drive citation maintenance, novelty checking, or outward-facing export automation.

### 4. Medium: Figure 2 still over-signals precision, and Figure 3 has a rendered readability defect

Type: presentation/consistency defect  
Checked: figure scripts vs rendered figures vs manuscript wording

- Figure 2 Panel A presents the matched readout result as two exact bars plus an exact `Δ = 0.005` annotation in [fig2_matched_readouts.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/figures/fig2_matched_readouts.py:111). The text is careful that "matched" means similar readout quality, not formal equivalence; the figure visually pushes harder than the prose does.
- Figure 3 is conceptually the strongest main-text figure, but the current rendered table in Panel C clips at the right boundary, stemming from the current full-width table layout in [fig3_bridge_failure.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/figures/fig3_bridge_failure.py:302) and the rendered asset [fig3_bridge_failure.png](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/figures/fig3_bridge_failure.png).

Required revision direction: add an uncertainty or equivalence caveat to Figure 2's detection panel, and reflow Figure 3 Panel C so the example text remains legible at manuscript scale.

### 5. Medium-High: Sections 2 and 3 still front-load too much governance prose relative to the evidence

Type: presentation/consistency defect  
Checked: reader-facing prose vs evidence spine

The framing is much better than earlier drafts, but the paper still spends too much prime early space on claim-boundary management and novelty calibration in [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:47), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:53), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:80), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:108), and [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:118).

That prose is mostly correct. The problem is narrative economy. The reader should reach the paper's evidence spine faster.

Required revision direction: compress early governance language, keep the novelty claim narrow, and move more quickly into the FaithEval anchor and the surface-validity question.

## Resolved Or Narrowed Findings That Still Need Guardrails

### D7 is no longer structurally over-promoted

The draft now treats FaithEval as the only load-bearing localization anchor in the introduction and Section 4, keeps D7 in a narrower corroborative role, and removes the old synthesis table that had visually flattened the two evidence tiers. The key corrected surfaces are [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:30), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:126), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:161), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:171), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:177), and [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:308).

The current canonical D7 reading is narrower. The April 16 audit says the surviving sentence is benchmark-local supporting evidence that selector choice matters, while explicitly rejecting both mechanism-clean closure and the older probe-null framing: [2026-04-16-d7-full500-two-seed-current-state-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-16-d7-full500-two-seed-current-state-audit.md:173), [2026-04-16-d7-full500-two-seed-current-state-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-16-d7-full500-two-seed-current-state-audit.md:184), [2026-04-16-d7-full500-two-seed-current-state-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-16-d7-full500-two-seed-current-state-audit.md:191).

No further structural retiering is required right now. The remaining risk is regression.

### The measurement section is scientifically stronger once framed narrowly

The clean result after the GPT-4o rerun is that CSV-v3 and StrongREJECT tie on holdout binary accuracy; the reason to prefer CSV-v3 is richer measurement structure, not binary superiority. The body now says this correctly in [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:289), and the canonical cleanup report is explicit that the discriminating question shifted away from binary accuracy superiority: [2026-04-13-jailbreak-measurement-cleanup.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-jailbreak-measurement-cleanup.md:227), [2026-04-13-jailbreak-measurement-cleanup.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-jailbreak-measurement-cleanup.md:239), [2026-04-13-jailbreak-measurement-cleanup.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-jailbreak-measurement-cleanup.md:243).

The remaining risk here is rhetorical drift. The science is stronger when phrased as "measurement changed what the paper could honestly conclude" rather than as a generic story of repeated evaluator reversals.

## Untouched Surfaces

- [Abstract](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/abstract.md:1), [Section 2](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_2_scope_constructs.md:1), [Section 6](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_6_measurement.md:1), [Section 8](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_8_limitations.md:1), and the [appendix](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/appendix.md:1) are broadly coherent with the revised evidence hierarchy.
- BioASQ is now described in the defensible way: endpoint-flat but behaviorally active, not behaviorally inert. That matches the canonical pipeline audit: [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:194), [bioasq_pipeline_audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/data/gemma3_4b/intervention/bioasq/bioasq_pipeline_audit.md:170), [bioasq_pipeline_audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/data/gemma3_4b/intervention/bioasq/bioasq_pipeline_audit.md:185).
- The measurement story is also correctly narrowed: the post-GPT-4o claim is tie on binary holdout, prefer CSV-v3 for richer outcome structure, not superiority: [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:289), [2026-04-13-jailbreak-measurement-cleanup.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-jailbreak-measurement-cleanup.md:227), [2026-04-13-jailbreak-measurement-cleanup.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-jailbreak-measurement-cleanup.md:243).
- The main remaining untouched-surface weakness is narrative economy, not falsity. Sections 2 and 3 still spend more early-page budget on claim-boundary management and novelty fencing than the evidence spine really needs: [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:47), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:90), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:118).

## Claim Status Highlights

| Claim family | Status | Canonical basis | Required action |
|---|---|---|---|
| Four-stage measurement/localization/control/externality scaffold | Supported | The paper's three case studies jointly support this synthesis | Keep as the paper's main thesis |
| FaithEval matched readout quality does not imply matched steering utility | Supported | FaithEval control and slope-difference artifacts under `data/gemma3_4b/intervention/faitheval/` | Keep as flagship empirical result |
| Detector readouts are real but readout interpretation is fragile | Supported | Draft Section 4.1 plus detector audits and caveats | Keep the narrowed wording |
| Full-500 D7 evidence supports benchmark-local selector divergence | Supported, narrowly | April 16 current-state audit | Keep as supporting evidence only |
| H-neuron scaling is behaviorally active on BioASQ but endpoint-flat | Supported | `data/gemma3_4b/intervention/bioasq/bioasq_pipeline_audit.md` | Keep the endpoint-flat plus behavioral-activity framing |
| ITI improves TruthfulQA MC but harms open-ended factual generation | Supported | [2026-04-01-priority-reruns-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-01-priority-reruns-audit.md:130) and bridge test report | Keep |
| Wrong-entity substitution is the dominant bridge failure mode and coarse reweighting explains the harm | Partially supported / overstated | Bridge report supports the first half, not a settled mechanism | Soften explanation language |
| Measurement choices changed the jailbreak conclusion | Supported | Truncation, graded scoring, evaluator cleanup, schema bug audit | Keep |
| CSV-v3 is preferred because it beats StrongREJECT on holdout accuracy | Unsupported / stale | Post-GPT-4o rerun removes that claim | Do not state this anywhere |

## Figure And Table Audit

- `Figure 3` earns inclusion and is the closest main-text figure to publication-ready. Its only important scientific caveat is that the failure-mode taxonomy is descriptive and single-rater; its current practical defect is the clipped Panel C table.
- `Figure 2` should stay, but Panel A needs less precision theater and clearer caveating around what "matched" means.
- `Figure 4` should stay only if the interval methodology is reconciled with the text and the panel load is simplified. The current Panel C mismatch and Panel D redundancy are the main issues.
- `Figure 1` is conceptually useful but over-detailed for an opener; its stage labels should do less.
- `Table 1` and `Table 6` both earn inclusion.
- The old `Table 5` problem has been resolved by removal. The main remaining visual risks are Figure 2 precision signaling, Figure 3 readability, and Figure 4 interval inconsistency.

## Prose And Structure Audit

- The draft is strongest when it stays at the level of what the evidence shows and weakest when it infers latent mechanism from output behavior.
- The paper's true flagship is FaithEval. The structure should make that obvious earlier and more consistently.
- Section transitions still repeat the thesis more often than they advance inference.
- Internal audit language remains visible in reader-facing prose. Terms like `mixed-ruler`, `locked causal branch`, and `token-cap debt` read like internal state summaries rather than paper prose.
- The novelty framing is mostly calibrated, but it should continue to emphasize the matched FaithEval case study and integrated scaffold, not a slogan-level detect/control dissociation claim that prior work already occupies.

## Reference Integrity Audit

- The draft's literature framing is broadly reasonable, but the local citation registry is unreliable enough that every claim-to-source mapping should be checked manually before sync.
- Novelty should remain anchored on the matched cross-target-type FaithEval comparison and the integrated audit scaffold, not on a generic detect/control dissociation claim already occupied by Arad, AxBench, Bhalla, and related work.
- If the introduction leans heavily on the decodability-versus-causal-use distinction, it would benefit from one deeper foundational citation layer beyond the current lightweight setup.

## Safe Thesis

The strongest version currently earned is:

> In Gemma-3-4B-IT, strong held-out internal readouts are not sufficient evidence for useful steering targets. The matched FaithEval neuron-versus-SAE comparison shows that similar readout quality can diverge sharply in control utility; the TriviaQA bridge result shows that even successful steering can externalize harm on nearby generation tasks; and the jailbreak case study shows that measurement choices can change what the paper can honestly conclude. Together, these justify a four-stage audit standard separating measurement, localization, control, and externality.

This version keeps FaithEval as the primary anchor, treats bridge as strong output-level diagnosis rather than mechanism closure, keeps D7 as supporting only, and leaves evaluator preference grounded in measurement structure rather than binary superiority.

## Priority Order

1. Rewrite Section 5 interpretation so the bridge result stays strong while the mechanism claims become properly bounded.
2. Reconcile Figure 4's interval methodology with the text, remove or repurpose Panel D, and reduce any remaining precision overstatement in Figure 2.
3. Fix the Figure 3 Panel C clipping defect.
4. Compress Sections 2 and 3 so the reader reaches the evidence spine faster.
5. Repair `paper/citations/registry.json` and recheck every citation used for novelty or framing.
6. Guard against D7 regression, but do not spend more main-text budget promoting it.

## Best-Practice Lens Consulted

- Nature Cell Biology checklist on clarity, accessibility, and design best practices: https://www.nature.com/articles/s41556-025-01684-z
- PLOS, "Ten Simple Rules for Better Figures": https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833
- Nature, "Points of view: how to draw figures that will make your paper stand out": https://www.nature.com/articles/513463e
