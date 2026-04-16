# Research-Grade Draft Audit

Date: 2026-04-16  
Scope: [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:1), figure scripts under `paper/draft/figures/`, and current canonical evidence under `notes/act3-reports/` and `data/`  
Review mode: findings-first, read-only, publication-rigor audit

## Executive Verdict

The draft has a real paper in it, but it is not yet publication-tight. The strongest scientific core is sound: the FaithEval H-neuron versus SAE dissociation, the held-out TriviaQA bridge externality result, and the post-cleanup jailbreak measurement story all survive audit. The main problem is not missing evidence. It is evidence hierarchy, inferential discipline, and reference integrity.

In current form, the paper still gives too much structural weight to caveated D7 evidence, lets Section 5 slide from strong behavioral diagnosis into stronger mechanism language than the evidence earns, and carries a broken citation registry that should not be trusted for submission-facing synchronization.

## Findings

### 1. Critical: D7 remains structurally over-promoted relative to its earned evidential status

The paper says the right caveat locally, but the draft still gives the D7 package too much headline weight in the paper-level synthesis. The risky surfaces are the section setup and synthesis register in [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:161), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:171), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:182), and [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:315).

The current canonical D7 reading is narrower. The April 16 audit says the surviving sentence is benchmark-local supporting evidence that selector choice matters, while explicitly rejecting both "mechanism-clean" closure and the older "probe is null at full-500" framing: [2026-04-16-d7-full500-two-seed-current-state-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-16-d7-full500-two-seed-current-state-audit.md:173), [2026-04-16-d7-full500-two-seed-current-state-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-16-d7-full500-two-seed-current-state-audit.md:184), [2026-04-16-d7-full500-two-seed-current-state-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-16-d7-full500-two-seed-current-state-audit.md:191).

Recommended fix: retier D7 everywhere outside its local subsection. It should remain supporting evidence, not a co-anchor beside the FaithEval result.

### 2. Critical: Section 5 overstates mechanism relative to the bridge evidence

The bridge result itself is strong and publication-grade. The overreach appears in the interpretive language layered on top of it. The problem lines are [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:229), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:231), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:233), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:246), and [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:252).

What the canonical report clearly supports is: the held-out test result is real, the harm is statistically significant, and wrong-entity substitution is the most frequent manually diagnosed failure mode at 30/43 right-to-wrong flips. It does not establish an internal mechanism, and the exact taxonomy shares are approximate because the coding is single-rater: [2026-04-13-bridge-phase3-test-results.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-bridge-phase3-test-results.md:74), [2026-04-13-bridge-phase3-test-results.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-bridge-phase3-test-results.md:81), [2026-04-13-bridge-phase3-test-results.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-bridge-phase3-test-results.md:190), [2026-04-13-bridge-phase3-test-results.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-bridge-phase3-test-results.md:197).

Recommended fix: keep the behavioral diagnosis, but recast the explanatory language as bounded hypotheses. "Consistent with" is earned; "explains why" and "supports an indiscriminate redistribution interpretation" are too strong in current form.

### 3. High: Table 5 and parts of the figure package flatten clean and caveated evidence into the same visual register

This is the draft's clearest figure/table hierarchy problem. [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:177) and [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:182) place the clean FaithEval anchor and the caveated D7 package into one synthesis table with roughly equal visual status. That works against the paper's own claim calibration.

Figure 2 is directionally right, but Panel A presents exact AUROC bars and a raw `Δ = 0.005` without uncertainty or any explicit equivalence caveat, which visually overstates precision: [fig2_matched_readouts.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/figures/fig2_matched_readouts.py:111). Figure 4 has a more concrete integrity problem: the evaluator panel computes Wilson intervals from counts in the script, while the body text reports prompt-clustered holdout CIs: [fig4_measurement.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/figures/fig4_measurement.py:146), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:289).

Recommended fix: remove or substantially demote Table 5, keep Figure 3 as the strongest main-text figure, add local uncertainty/caveat cues where the main inference depends on them, and reconcile Figure 4's interval method with the manuscript text.

### 4. High: The measurement section is scientifically strong, but some framing still misstates what actually changed

The clean result after the GPT-4o rerun is that CSV-v3 and StrongREJECT tie on holdout binary accuracy; the reason to prefer v3 is richer measurement structure, not binary superiority. The body eventually says this correctly in [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:296), but the section still risks reading as a generic "measurement reversals" story rather than a narrower claim about construct choice, truncation, scoring granularity, and schema hygiene.

The canonical cleanup report is explicit that the discriminating question shifted away from binary accuracy superiority: [2026-04-13-jailbreak-measurement-cleanup.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-jailbreak-measurement-cleanup.md:227), [2026-04-13-jailbreak-measurement-cleanup.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-jailbreak-measurement-cleanup.md:239), [2026-04-13-jailbreak-measurement-cleanup.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-jailbreak-measurement-cleanup.md:243).

Recommended fix: sharpen the framing to "measurement changed what the paper could honestly conclude" rather than "the evaluator story kept flipping." The current science is stronger when presented more narrowly.

### 5. High: The citation registry is corrupted enough to be a submission risk

The manuscript bibliography is broadly salvageable, but [registry.json](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/citations/registry.json:92) is not trustworthy as a source-of-truth. Concrete mismatches include `tan2024steering` pointing to Pres files, `pres2024reliable` pointing to Bhalla files, `bhalla2024unifying` pointing to FaithEval files while saying "Need to acquire," and `ming2025faitheval` claiming no local files despite local markdown existing: [registry.json](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/citations/registry.json:92), [registry.json](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/citations/registry.json:110), [registry.json](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/citations/registry.json:128), [registry.json](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/citations/registry.json:182).

Recommended fix: repair registry identity and status mapping before any outward-facing sync. Do not let this file drive automatic citation reconciliation in its current state.

### 6. Medium-High: Sections 2 and 3 still front-load too much governance prose relative to evidence

The framing is much better than earlier drafts, but the paper still spends too much prime early space on claim-boundary management and novelty calibration in [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:47), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:53), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:80), [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:108), and [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:118).

That prose is mostly correct. The problem is narrative economy. The reader should reach the paper's evidence spine faster.

Recommended fix: compress early governance language, keep the novelty claim narrow, and move more quickly into the FaithEval anchor and the surface-validity question.

## Claim Status Highlights

| Claim family | Status | Canonical basis | Required action |
|---|---|---|---|
| Four-stage measurement/localization/control/externality scaffold | Supported | The paper's three case studies jointly support this synthesis | Keep as the paper's main thesis |
| FaithEval matched readout quality does not imply matched steering utility | Supported | FaithEval control and slope-difference artifacts under `data/gemma3_4b/intervention/faitheval/` | Keep as flagship empirical result |
| Detector readouts are real but readout interpretation is fragile | Supported | Draft Section 4.1 plus detector audits and caveats | Keep the narrowed wording |
| Pilot probe-ranked D7 heads were inert while gradient-ranked heads were not | Stale / outdated as headline framing | Superseded by April 16 full-500 current-state audit | Recast as historical corroboration only |
| Full-500 D7 evidence supports benchmark-local selector divergence | Supported, narrowly | April 16 current-state audit | Keep as supporting evidence only |
| H-neuron scaling is behaviorally active on BioASQ but endpoint-flat | Supported | `data/gemma3_4b/intervention/bioasq/bioasq_pipeline_audit.md` | Keep the endpoint-flat plus behavioral-activity framing |
| ITI improves TruthfulQA MC but harms open-ended factual generation | Supported | [2026-04-01-priority-reruns-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-01-priority-reruns-audit.md:130) and bridge test report | Keep |
| Wrong-entity substitution is the dominant bridge failure mode and coarse reweighting explains the harm | Partially supported / overstated | Bridge report supports the first half, not a settled mechanism | Soften explanation language |
| Measurement choices changed the jailbreak conclusion | Supported | Truncation, graded scoring, evaluator cleanup, schema bug audit | Keep |
| CSV-v3 is preferred because it beats StrongREJECT on holdout accuracy | Unsupported / stale | Post-GPT-4o rerun removes that claim | Do not state this anywhere |

## Figure And Table Audit

- `Figure 3` earns inclusion and is the closest main-text figure to publication-ready. Its only important caveat is that the failure-mode taxonomy is descriptive and single-rater.
- `Figure 2` should stay, but Panel A needs less precision theater and clearer caveating around what "matched" means.
- `Figure 4` should stay only if the interval methodology is reconciled with the text and the panel load is simplified.
- `Figure 1` is conceptually useful but over-detailed for an opener; its stage labels should do less.
- `Table 1` and `Table 6` both earn inclusion.
- `Table 5` is the weakest main-text table because it visually flattens anchor and supporting evidence. It is the first item I would cut or redesign.

## Prose And Structure Audit

- The draft is strongest when it stays at the level of what the evidence shows and weakest when it infers latent mechanism from output behavior.
- The paper's true flagship is FaithEval. The structure should make that obvious much earlier and more consistently.
- Section transitions still repeat the thesis more often than they advance inference.
- Internal audit language remains visible in reader-facing prose. Terms like `mixed-ruler`, `locked causal branch`, and `token-cap debt` read like internal state summaries rather than paper prose.
- The novelty framing is mostly calibrated, but it should continue to emphasize the matched case study and integrated scaffold, not the slogan-level "decodability does not imply control" point that prior work already occupies.

## Reference Integrity Audit

- The draft's literature framing is broadly reasonable, but the local citation registry is unreliable enough that every claim-to-source mapping should be checked manually before sync.
- Novelty should remain anchored on the matched cross-target-type FaithEval comparison and the integrated audit scaffold, not on a generic detect/control dissociation claim already occupied by Arad, AxBench, Bhalla, and related work.
- If the introduction leans heavily on the decodability-versus-causal-use distinction, it would benefit from one deeper foundational citation layer beyond the current lightweight setup.

## Priority Order

1. Retier D7 everywhere so it cannot be mistaken for co-anchor evidence.
2. Rewrite Section 5 interpretation so the bridge result stays strong while the mechanism claims become properly bounded.
3. Remove or redesign Table 5 and reconcile Figure 4's interval methodology with the text.
4. Compress Sections 2 and 3 so the reader reaches the evidence spine faster.
5. Repair `paper/citations/registry.json` and recheck every citation used for novelty or framing.

## Best-Practice Lens Consulted

- Nature Cell Biology checklist on clarity, accessibility, and design best practices: https://www.nature.com/articles/s41556-025-01684-z
- PLOS, "Ten Simple Rules for Better Figures": https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833
- Nature, "Points of view: how to draw figures that will make your paper stand out": https://www.nature.com/articles/513463e
