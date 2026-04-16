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
- `uv run pytest tests/test_citation_registry.py` passed (`6 passed`).
- `uv run python scripts/check_citation_registry.py` passed.

Progress updates landed after the initial audit:

- `6ce91d0` `fix(fig2): add AUROC uncertainty and soften equivalence wording`
- `9b01ec8` `fix(citations): validate registry against local paper files`
- `7610fc2` `fix(fig4): simplify measurement figure and align holdout text`

## Review Status

| Area | Type | Status | What was checked against what | Current conclusion |
|---|---|---|---|---|
| D7 evidence hierarchy | overstated inference | Resolved, guard against regression | Manuscript vs [2026-04-16-d7-full500-two-seed-current-state-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-16-d7-full500-two-seed-current-state-audit.md:1) | FaithEval is now the sole load-bearing anchor; D7 is benchmark-local supporting evidence only. |
| Section 5 bridge interpretation | overstated inference | Resolved, keep bounded | Manuscript vs [2026-04-13-bridge-phase3-test-results.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-bridge-phase3-test-results.md:1) | Output-level diagnosis remains strong; explanatory language is now bounded to behavior-level interpretation. |
| Figure 4 uncertainty logic | presentation/consistency defect | Resolved, keep aligned | Figure script vs manuscript text vs holdout validation note | Panel C now uses the canonical prompt-clustered bootstrap CIs, the stale StrongREJECT table values are fixed, and the redundant Panel D has been removed. |
| Figure 2 precision signaling | presentation/consistency defect | Resolved, keep bounded | Figure script vs manuscript wording | Detection panel now shows uncertainty and explicitly avoids implying formal equivalence. |
| Figure 3 readability | presentation/consistency defect | Open | Figure script vs rendered PNG | Panel C table clips at the right edge in the current render. |
| Jailbreak measurement framing | unsupported claim | Resolved, keep narrow | Manuscript vs [2026-04-13-jailbreak-measurement-cleanup.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-jailbreak-measurement-cleanup.md:1) | Safe claim is tie on binary holdout; prefer CSV-v3 for richer structure, not superiority. |
| BioASQ summary framing | presentation/consistency defect | Resolved, keep narrow | Manuscript vs [bioasq_pipeline_audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/data/gemma3_4b/intervention/bioasq/bioasq_pipeline_audit.md:1) | Safe claim is endpoint-flat but behaviorally active, not a clean null. |
| Citation registry integrity | unsupported claim | Resolved, validator added | [registry.json](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/citations/registry.json:1) vs local `papers/` files | Registry mappings were repaired and a dedicated validator plus tests now guard against recurrence. |
| Sections 2 and 3 narrative economy | presentation/consistency defect | Open | Reader-facing prose vs evidence spine | Early sections still spend too much budget on governance and novelty fencing. |

## Executive Verdict

The draft now has a coherent evidence hierarchy and a defensible paper-level thesis. The strongest scientific core remains intact: the FaithEval neuron-versus-SAE dissociation, the held-out TriviaQA bridge externality result, and the post-cleanup jailbreak measurement story all survive audit.

The main remaining blockers are narrower and more concrete than before. The earlier Section 5 inferential-discipline issue has been resolved in the manuscript by bounding the bridge interpretation to behavior-level claims, and the Figure 2, Figure 4, and citation-registry defects flagged in the first pass have now been fixed in committed follow-up work. The top remaining risks are Figure 3's clipped table and the paper's still-slow early narrative economy in Sections 2 and 3.

## Did Finding 1 Actually Land?

Yes, substantially.

- FaithEval is now the only load-bearing localization/control anchor in the abstract, introduction, Section 4, synthesis, limitations, and appendix: [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:17), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:29), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:126), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:177), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:307), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:360).
- D7 is now described in the narrow way the April 16 current-state audit supports: benchmark-local supporting evidence that selector choice matters on this surface, not mechanism-clean closure: [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:171), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:173), [2026-04-16-d7-full500-two-seed-current-state-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-16-d7-full500-two-seed-current-state-audit.md:191).
- The old main-text synthesis table is gone; no remaining main-text surface now visually flattens FaithEval and D7 into the same evidential tier.
- Recommendation 1 now stands on FaithEval alone in [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:317).

Residual risk: later edits could easily reintroduce D7 inflation, but D7 is no longer the draft's main blocker in its current form.

## Open Findings

### 1. Medium: Figure 3 still has a rendered readability defect in Panel C

Type: presentation/consistency defect  
Checked: figure script vs rendered figure

- Figure 3 is conceptually the strongest main-text figure, but the current rendered table in Panel C clips at the right boundary, stemming from the current full-width table layout in [fig3_bridge_failure.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/figures/fig3_bridge_failure.py:302) and the rendered asset [fig3_bridge_failure.png](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/figures/fig3_bridge_failure.png).

Required revision direction: reflow Figure 3 Panel C so the example text remains legible at manuscript scale.

### 2. Medium-High: Sections 2 and 3 still front-load too much governance prose relative to the evidence

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

The figure and manuscript alignment issue that was open in the first pass is now fixed. [fig4_measurement.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/figures/fig4_measurement.py:1) now pulls Panel C intervals from the canonical holdout bootstrap artifact, the rendered figure has been simplified to three panels, and the Section 6 table now reports the post-rerun StrongREJECT-GPT-4o tie consistently. This landed in commit `7610fc2` (`fix(fig4): simplify measurement figure and align holdout text`).

### Figure 2 no longer overstates equivalence

The earlier precision-signaling problem in Figure 2 has been repaired. [fig2_matched_readouts.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/figures/fig2_matched_readouts.py:1) now shows AUROC uncertainty and the caption/surface language explicitly says the result is similar held-out readout quality, not formal equivalence. This landed in commit `6ce91d0` (`fix(fig2): add AUROC uncertainty and soften equivalence wording`).

### Citation registry integrity is now guarded by code

The specific registry corruption cited in the first pass has been repaired in [registry.json](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/citations/registry.json:1), and the repo now has an explicit validator in [check_citation_registry.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/scripts/check_citation_registry.py:1) plus focused tests in [test_citation_registry.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/tests/test_citation_registry.py:1). This landed in commit `9b01ec8` (`fix(citations): validate registry against local paper files`). The remaining risk is ordinary future drift rather than current corruption.

### Bridge interpretation is now aligned with the bridge report

The Section 5 bridge interpretation has been brought into line with the canonical bridge report. The paper now keeps the held-out harm result, significance, and manually diagnosed wrong-entity pattern strong, while bounding the explanatory layer to behavior-level hypotheses and placing the single-rater coding caveat next to the taxonomy claim. The key repaired surfaces are [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:110), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:212), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:222), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:226), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:239), and [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:245).

This finding should now be treated as resolved, with the remaining risk being regression toward stronger mechanism language in later prose edits. The paper is strongest when bridge remains an output-level diagnosis plus bounded interpretation rather than mechanism closure. The narrowing landed in commit `72665c2` (`docs(paper): bound bridge mechanism claims`).

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
| Wrong-entity substitution is the dominant bridge failure mode and coarse reweighting explains the harm | Supported, narrowly | Bridge report supports the dominant diagnosed failure mode plus bounded behavior-level interpretation, not settled mechanism | Keep the softened explanation language |
| Measurement choices changed the jailbreak conclusion | Supported | Truncation, graded scoring, evaluator cleanup, schema bug audit | Keep |
| CSV-v3 is preferred because it beats StrongREJECT on holdout accuracy | Unsupported / stale | Post-GPT-4o rerun removes that claim | Do not state this anywhere |

## Figure And Table Audit

- `Figure 3` earns inclusion and is the closest main-text figure to publication-ready. Its only important scientific caveat is that the failure-mode taxonomy is descriptive and single-rater; its current practical defect is the clipped Panel C table.
- `Figure 2` now earns inclusion more cleanly: Panel A carries uncertainty and no longer over-implies formal equivalence.
- `Figure 4` now earns inclusion more cleanly: the holdout panel is aligned with the canonical bootstrap artifact and the redundant fourth panel has been removed.
- `Figure 1` is conceptually useful but over-detailed for an opener; its stage labels should do less.
- `Table 1` and `Table 6` both earn inclusion.
- The old `Table 5` problem has been resolved by removal. The main remaining visual risk is Figure 3 readability.

## Prose And Structure Audit

- The draft is strongest when it stays at the level of what the evidence shows and weakest when it infers latent mechanism from output behavior.
- The paper's true flagship is FaithEval. The structure should make that obvious earlier and more consistently.
- Section transitions still repeat the thesis more often than they advance inference.
- Internal audit language remains visible in reader-facing prose. Terms like `mixed-ruler`, `locked causal branch`, and `token-cap debt` read like internal state summaries rather than paper prose.
- The novelty framing is mostly calibrated, but it should continue to emphasize the matched FaithEval case study and integrated scaffold, not a slogan-level detect/control dissociation claim that prior work already occupies.

## Reference Integrity Audit

- The draft's literature framing is broadly reasonable, and the citation registry now has a validator-backed integrity check rather than relying on manual trust alone.
- Novelty should remain anchored on the matched cross-target-type FaithEval comparison and the integrated audit scaffold, not on a generic detect/control dissociation claim already occupied by Arad, AxBench, Bhalla, and related work.
- If the introduction leans heavily on the decodability-versus-causal-use distinction, it would benefit from one deeper foundational citation layer beyond the current lightweight setup.

## Safe Thesis

The strongest version currently earned is:

> In Gemma-3-4B-IT, strong held-out internal readouts are not sufficient evidence for useful steering targets. The matched FaithEval neuron-versus-SAE comparison shows that similar readout quality can diverge sharply in control utility; the TriviaQA bridge result shows that even successful steering can externalize harm on nearby generation tasks; and the jailbreak case study shows that measurement choices can change what the paper can honestly conclude. Together, these justify a four-stage audit standard separating measurement, localization, control, and externality.

This version keeps FaithEval as the primary anchor, treats bridge as strong output-level diagnosis rather than mechanism closure, keeps D7 as supporting only, and leaves evaluator preference grounded in measurement structure rather than binary superiority.

## Priority Order

1. Fix the Figure 3 Panel C clipping defect.
2. Compress Sections 2 and 3 so the reader reaches the evidence spine faster.
3. Guard against bridge-claim and D7 regression, but do not spend more main-text budget promoting either.
4. Keep the new citation-registry validator in the loop for any future bibliography edits.

## Best-Practice Lens Consulted

- Nature Cell Biology checklist on clarity, accessibility, and design best practices: https://www.nature.com/articles/s41556-025-01684-z
- PLOS, "Ten Simple Rules for Better Figures": https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833
- Nature, "Points of view: how to draw figures that will make your paper stand out": https://www.nature.com/articles/513463e
