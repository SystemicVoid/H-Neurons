# Publication-Readiness Review

Date: 2026-04-15  
Scope: [full_paper.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/full_paper.md:1) and source shards under `paper/draft/`  
Review mode: submission blockers  
Deliverable shape: one consolidated memo

## Executive Verdict

**Needs substantive revision before publication sync.**

The draft has a real paper in it. The FaithEval neuron-vs-SAE comparison, the bridge externality result, and the measurement section's core scientific point are strong enough to support a publishable manuscript. The current blocker is not lack of evidence; it is evidence hierarchy. The draft still over-promotes some caveated results, carries stale claim language in a few critical places, and reads too much like an internal audit document in Sections 2 and 6.

Two status checks are already clean:

- `uv run python scripts/build_full_paper.py --check`
- `uv run python scripts/audit_ci_coverage.py`

## Blocking Findings

### 1. D7 / probe-head evidence is over-promoted beyond the current earned boundary

**Severity:** Critical  
**Action:** Downgrade and localize the claim everywhere outside the caveated selector subsection.

The manuscript still states or strongly implies "perfect readout, null intervention" as if that were a clean headline result. That wording appears in the abstract, introduction, claim ledger, and synthesis: [abstract.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/abstract.md:3), [section_1_introduction.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_1_introduction.md:11), [section_2_scope_constructs.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_2_scope_constructs.md:73), [section_4_case_study_I.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_4_case_study_I.md:98), [section_7_synthesis.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_7_synthesis.md:11).

The canonical audits are narrower:

- The pilot says the probe result is **uninformative, not negative**, and explicitly says the pilot is not a controlled selector comparison because the locked alphas differ: [2026-04-07-d7-causal-pilot-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-07-d7-causal-pilot-audit.md:137), [2026-04-07-d7-causal-pilot-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-07-d7-causal-pilot-audit.md:144)
PILOT AUDIT results are SUPERSEDED, take that into account and DO NOT rely on superseded results. 
- The current full-500 audit says the "probe-null" framing is no longer supported and that the surviving claim is benchmark-local selector divergence with caveats: [2026-04-14-d7-full500-current-state-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-14-d7-full500-current-state-audit.md:159), [2026-04-14-d7-full500-current-state-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-14-d7-full500-current-state-audit.md:168), [2026-04-14-d7-full500-current-state-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-14-d7-full500-current-state-audit.md:176)

**Required revision direction**

- Keep the probe-head story as supporting evidence inside §4.3–4.4.
- Remove or soften "perfect readout produced null intervention" from abstract, intro, claim ledger, and synthesis.
- Replace it with language like: `pilot-local probe-ranked heads showed no clear effect, while the current full-500 panel supports only a benchmark-local, caveated selector-divergence claim`.

### 2. The evaluator-comparison caveat is stale and now contradicts later sections

**Severity:** Critical  
**Action:** Update the claim ledger and limitation language to match the SR-4o cleanup result.

The claim ledger still says the holdout gap compresses to 2.0 pp and that the judge-model confound remains unresolved: [section_2_scope_constructs.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_2_scope_constructs.md:81). But the body later reports the GPT-4o rerun and a 0.0 pp holdout gap with identical error sets: [section_6_measurement.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_6_measurement.md:217), [section_6_measurement.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_6_measurement.md:223).

The canonical source is already clear:

- The measurement cleanup report says the discriminating question has shifted away from binary accuracy superiority and toward rubric structure / granularity: [2026-04-13-jailbreak-measurement-cleanup.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-jailbreak-measurement-cleanup.md:227), [2026-04-13-jailbreak-measurement-cleanup.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-13-jailbreak-measurement-cleanup.md:239)
- The provenance ledger is already updated to `Holdout gap after upgrade = 0.0 pp`: [number_provenance.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/number_provenance.md:105)

**Required revision direction**

- Update the claim ledger caveat.
- Update Limitation L8, which still frames the judge-model confound as live in its older form: [section_8_limitations.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_8_limitations.md:16)
- Reframe the evaluator story as: `binary holdout accuracy is now tied after the SR-4o upgrade; the paper's reason to prefer v3 is measurement granularity and outcome taxonomy, not superior heldout binary accuracy`.

### 3. BioASQ is oversimplified as a clean null in summary-level prose

**Severity:** High  
**Action:** Propagate the nuanced body wording into abstract, intro, claim ledger, synthesis, and any figure-adjacent summaries.

The body now says the right thing: the endpoint is flat, but the intervention is behaviorally active and strongly changes answer style: [section_5_case_study_II.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_5_case_study_II.md:11). But the abstract, intro, claim ledger, synthesis, and other summary surfaces still compress this into "works on FaithEval, not on BioASQ": [abstract.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/abstract.md:3), [section_1_introduction.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_1_introduction.md:13), [section_2_scope_constructs.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_2_scope_constructs.md:74), [section_7_synthesis.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_7_synthesis.md:13).

The canonical BioASQ audit is explicit:

- "The BioASQ intervention is not a clean null": [bioasq_pipeline_audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/data/gemma3_4b/intervention/bioasq/bioasq_pipeline_audit.md:13)
- The safe claim is that H-neuron scaling changes behavior much more than alias accuracy: [bioasq_pipeline_audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/data/gemma3_4b/intervention/bioasq/bioasq_pipeline_audit.md:14), [bioasq_pipeline_audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/data/gemma3_4b/intervention/bioasq/bioasq_pipeline_audit.md:210), [bioasq_pipeline_audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/data/gemma3_4b/intervention/bioasq/bioasq_pipeline_audit.md:225)

**Required revision direction**

- Replace "null on BioASQ" with wording closer to: `no robust net alias-accuracy effect on BioASQ despite substantial behavioral perturbation`.
- Avoid using BioASQ as a simple portability-null slogan in high-level framing.

### 4. Section 2 is still written like internal claim governance, not paper narrative

**Severity:** High  
**Action:** Compress, demote, or move author-workflow material.

Section 2 front-loads claim management before the reader has enough empirical context. The heaviest offenders are the "evaluation contract" and claim ledger language: [section_2_scope_constructs.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_2_scope_constructs.md:36), [section_2_scope_constructs.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_2_scope_constructs.md:65).

This also leaks internal status vocabulary into the paper:

- `headline-safe`
- `qualified evidence`
- `pre-commitment`

These appear in the intro and Section 2: [section_1_introduction.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_1_introduction.md:7), [section_2_scope_constructs.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_2_scope_constructs.md:38), [section_2_scope_constructs.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_2_scope_constructs.md:67)

**Required revision direction**

- Keep the construct table.
- Shrink the evaluation contract to a short methods paragraph or a smaller box.
- Either remove the claim ledger from the main body or rewrite it in ordinary scientific prose.
- Strip internal workflow phrasing from the reader-facing manuscript.

### 5. Section 4 does not cleanly separate claim-bearing selector evidence from supporting selector evidence

**Severity:** High  
**Action:** Make the FaithEval comparison unmistakably primary and demote the selector comparison in both prose and visuals.

The section says it moves from cleanest to most informative, but the selector story is split across a pilot result, a mixed-ruler full-500 comparator, Figure 2C, Table 5, and limitation prose in a way that leaves the evidential status blurry: [section_4_case_study_I.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_4_case_study_I.md:5), [section_4_case_study_I.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_4_case_study_I.md:96), [section_4_case_study_I.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_4_case_study_I.md:140)

The section's strongest clean claim is the FaithEval neuron-vs-SAE dissociation. The selector evidence is explicitly caveated by the paper itself and by the current audit: [section_8_limitations.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_8_limitations.md:14), [2026-04-14-d7-full500-current-state-audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/notes/act3-reports/2026-04-14-d7-full500-current-state-audit.md:176)

**Required revision direction**

- Rewrite the opening paragraph and Figure 2 caption so the FaithEval comparison is clearly the anchor result.
- Treat the selector comparison as corroborative, not co-equal.
- Ensure Table 5 and Figure 2 do not visually flatten the distinction.

### 6. Section 6 still reads like an internal audit/postmortem in places

**Severity:** High  
**Action:** Tighten the section around the scientific result and push some process detail to appendix or companion note.

Section 6 has strong science, but too much space goes to operational history, schema debugging, and evaluator-version process details: [section_6_measurement.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_6_measurement.md:16), [section_6_measurement.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_6_measurement.md:116), [section_6_measurement.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_6_measurement.md:152), [section_6_measurement.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_6_measurement.md:172)

The scientific core is much simpler:

- truncation changed the story
- binary vs graded changed significance on the same outputs
- evaluator identity changed interpretation
- a schema bug would have erased a real specificity contrast

**Required revision direction**

- Keep the scientific reversals.
- Compress implementation-history narrative.
- Move some pipeline-detail paragraphs and worked debugging explanations to appendix or companion note.

### 7. Anchor 3’s stage transition is internally inconsistent

**Severity:** Medium-High  
**Action:** Standardize the stage mapping across intro, Figure 1, synthesis, and any strategic-support language reused in the draft.

The draft maps Anchor 3 to `measurement → localization`: [section_1_introduction.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_1_introduction.md:22), [section_7_synthesis.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_7_synthesis.md:9). But the measurement section itself says the point is that measurement changed **what you would conclude about whether the intervention worked**: [section_6_measurement.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_6_measurement.md:20).

**Required revision direction**

- Make Anchor 3 consistently about `measurement → conclusion` or equivalent wording.
- Update Figure 1 and related text to match.

## Figure Blockers

### Figure 4 is not publication-ready

**Severity:** Critical  
**Action:** Redesign before submission sync.

Panel A places binary delta in `pp` and graded slope in `pp/α` on a shared axis, which invites a false visual comparison: [fig4_measurement.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/figures/fig4_measurement.py:191), [fig4_measurement.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/figures/fig4_measurement.py:264).

The figure also under-reports uncertainty where the scientific claim depends on it:

- Panel A encodes binary significance as text rather than showing the interval
- Panel B shows evaluator ranking without uncertainty bars: [fig4_measurement.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/figures/fig4_measurement.py:301)

**Required revision direction**

- Separate incompatible units into separate panels or normalize them to a common comparison.
- Show uncertainty explicitly where comparisons are inferential.
- Simplify the panel load; the current figure is too dense for main-text use.

### Figure 2 overstates the selector story and is too annotation-heavy

**Severity:** High  
**Action:** Keep Panels A/B as flagship; visually demote or redesign Panel C.

Figure 2A plots `1.000` for the probe result by using the **max top-head AUROC**, which makes the selector evidence look cleaner and more matched than the manuscript's actual claim boundary: [fig2_matched_readouts.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/figures/fig2_matched_readouts.py:109), [section_4_case_study_I.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_4_case_study_I.md:104)

Figure 2C also gets equal visual weight despite being mixed-ruler supporting evidence: [section_4_case_study_I.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_4_case_study_I.md:10), [section_4_case_study_I.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_4_case_study_I.md:142)

**Required revision direction**

- Either change Panel A's probe summary to match the real claim, or remove probe from the "matched detection quality" panel.
- Add an explicit caveat label to Panel C or move it to a secondary/supporting figure.
- Reduce annotation density and make uncertainty encoding comparable across series.

### Figure 1 is conceptually useful but carries too much tiny detail

**Severity:** Medium  
**Action:** Simplify.

The scaffold itself works, but the anchor text is too small and too specific for a conceptual opener: [fig1_four_stage_scaffold.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/figures/fig1_four_stage_scaffold.py:35), [fig1_four_stage_scaffold.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/figures/fig1_four_stage_scaffold.py:205)

The left-to-right anchor labels `3 / 1 / 2` add needless cognitive load.

**Required revision direction**

- Remove most numeric detail from the anchor boxes.
- Label the stage breaks directly rather than relying on anchor numbering.

### Figure 3 is closest to ready, but Panel C is cramped

**Severity:** Medium  
**Action:** Targeted cleanup only.

The core story works. The main remaining issue is that the example table is too cramped and loses the specificity it is supposed to provide: [fig3_bridge_failure.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/figures/fig3_bridge_failure.py:311)

**Required revision direction**

- Use fewer examples or more space.
- Keep this figure as the closest thing to a ready main-text figure.

## Secondary Improvements

- The title, abstract, intro, synthesis, and conclusion repeat the same thesis too many times; later sections should cash it out rather than re-announce it: [abstract.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/abstract.md:3), [section_1_introduction.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_1_introduction.md:17), [section_7_synthesis.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_7_synthesis.md:3), [section_9_conclusion.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_9_conclusion.md:3)
- The bridge section should mark "coarse reweighting" and "indiscriminate redistribution" as behavioral hypotheses earlier, not only via a later disclaimer: [section_5_case_study_II.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_5_case_study_II.md:42), [section_5_case_study_II.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_5_case_study_II.md:46), [section_5_case_study_II.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_5_case_study_II.md:69)
- The manuscript still contains too much internal status jargon beyond Section 2, including `mixed-ruler`, `mechanism-clean`, `paper-faithful ITI variant`, and similar phrases that belong in audits more than in the paper: [section_4_case_study_I.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_4_case_study_I.md:142), [section_5_case_study_II.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/paper/draft/section_5_case_study_II.md:28)

## Safe Thesis After Fixes

The strongest publishable version of the paper is:

> In Gemma-3-4B-IT, held-out readout quality alone did not reliably identify useful steering targets. The cleanest evidence is a matched FaithEval comparison where H-neurons and SAE features have similar detection quality but sharply different steering behavior. When steering did work, it was surface-local and could externalize to nearby generation tasks. Measurement choices also changed the apparent scientific conclusion on the same underlying outputs. Together, these results support a staged audit framework separating measurement, localization, control, and externality.

That thesis is strong enough. The draft weakens itself mainly when it reaches beyond it.

## Revision Priority Order

1. Fix D7/probe claim scope in abstract, intro, ledger, §4, §7, and figures.
2. Update evaluator-comparison language to reflect the SR-4o cleanup and 0.0 pp holdout gap.
3. Propagate the nuanced BioASQ wording to every high-level summary surface.
4. Rewrite Section 2 into paper-facing prose.
5. Tighten Section 6 around scientific takeaways, not audit history.
6. Redesign Figure 4 and demote/rework Figure 2C.
7. Trim repeated thesis restatements and internal jargon.
