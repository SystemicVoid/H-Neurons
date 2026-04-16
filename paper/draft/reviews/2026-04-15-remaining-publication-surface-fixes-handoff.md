# Reviewer Handoff Prompt

Date: 2026-04-15  
Scope: remaining publication-surface fixes in `paper/draft/`  
Review mode: direct manuscript sync pass, source-shard-only edits, rebuild and verify

## Context

This is a follow-up to `paper/draft/reviews/2026-04-15-reviewer-handoff-prompt.md`.
The broad paper-surface revision is already on `main`. This pass is narrower:
fix the remaining publication-surface issues in `paper/draft/` and leave the
manuscript in a clean, verified state.

- The worktree is dirty in unrelated `data/`, `scripts/`, and `site/data/`
  paths. Do not touch or revert unrelated files.
- Treat `paper/draft/full_paper.md` as generated output only. Edit source
  shards and figure scripts, then rebuild.
- Follow `paper/draft/AGENTS.md`.

## Primary Source of Truth for Claim Boundaries

- `notes/2026-04-11-strategic-assessment.md`
- `notes/act3-reports/2026-04-14-d7-full500-current-state-audit.md`
- `notes/act3-reports/2026-04-13-jailbreak-measurement-cleanup.md`
- `data/gemma3_4b/intervention/bioasq/bioasq_pipeline_audit.md`
- `notes/act3-reports/2026-04-13-bridge-phase3-test-results.md`

## What Is Already Correct and Must Be Preserved

- FaithEval is the flagship localization/control result.
- D7 is supporting, benchmark-local, and caveated.
- BioASQ is not a clean null; it is a flat endpoint with active behavioral
  perturbation.
- CSV-v3 is not preferred because of better holdout binary accuracy after the
  StrongREJECT GPT-4o rerun; holdout binary accuracy is tied.
- The reason to prefer CSV-v3 is granularity / outcome taxonomy.
- Anchor 3 is `measurement -> conclusion`.

## Findings To Turn Into Direct Fixes

### 1. D7 pilot still has too much headline weight outside §4

Action:
- Reduce D7 prominence on summary surfaces so it no longer reads as a
  co-headline result.
- Keep D7 as supporting selector evidence, subordinate to the FaithEval anchor
  result.

Preferred outcome:
- Remove D7 from the abstract entirely, unless a very short supporting clause
  is genuinely necessary.
- In summary surfaces, mention D7 only briefly and only as supporting evidence
  after the FaithEval anchor claim.

Files to inspect/edit:
- `paper/draft/abstract.md`
- `paper/draft/section_1_introduction.md`
- `paper/draft/section_2_scope_constructs.md`
- `paper/draft/section_7_synthesis.md`
- possibly `paper/draft/section_9_conclusion.md` if needed for consistency

Problem lines from prior review:
- `paper/draft/abstract.md:3`
- `paper/draft/section_1_introduction.md:11`
- `paper/draft/section_2_scope_constructs.md:57`
- `paper/draft/section_7_synthesis.md:11`

Guardrail:
- Do not strengthen D7 beyond “benchmark-local supporting evidence that
  selector choice matters on this surface.”

### 2. Figure 4 caption is stale against the post-GPT-4o tie framing

Action:
- Update the Figure 4 caption so it no longer says holdout validation merely
  “compresses the apparent evaluator gap.”
- It should reflect that the post-StrongREJECT-GPT-4o holdout binary result is
  tied, and that the remaining reason to keep CSV-v3 is richer measurement
  structure / taxonomy.

Files to inspect/edit:
- `paper/draft/section_6_measurement.md`
- `paper/draft/figures/fig4_measurement.py` only if figure text itself also
  needs syncing

Rendered symptom from prior review:
- `paper/draft/full_paper.md:425`

Guardrail:
- Do not reintroduce any wording that implies CSV-v3 beats StrongREJECT on
  holdout binary accuracy.

### 3. Figure label / title drift remains in Figures 1–3

Action:
- Fix reader-facing figure text so it matches the manuscript framing exactly.

Required changes:
- Figure 1:
  - The Measurement stage subtitle should refer to trusting the evaluation /
    measurement surface, not “trust the readout”.
- Figure 2:
  - Panel A should not read as purely “FaithEval matched detection” if it still
    includes jailbreak probe-head AUROC context.
  - Keep FaithEval clearly primary and jailbreak clearly supporting.
- Figure 3:
  - Panel A label for SimpleQA should use accuracy / correct-answer language,
    not “compliance”.

Files to inspect/edit:
- `paper/draft/figures/fig1_four_stage_scaffold.py`
- `paper/draft/figures/fig2_matched_readouts.py`
- `paper/draft/figures/fig3_bridge_failure.py`

Relevant current lines:
- `paper/draft/figures/fig1_four_stage_scaffold.py:27`
- `paper/draft/figures/fig2_matched_readouts.py:150`
- `paper/draft/figures/fig2_matched_readouts.py:209`
- `paper/draft/figures/fig3_bridge_failure.py:94`

Guardrail:
- Do not increase D7 visual prominence while fixing Figure 2 wording.

## Secondary Guidance

- Preserve the current manuscript improvements. This is a sync / weighting
  cleanup, not a broad rewrite.
- If you make further trims for duplication, keep them minimal and safe.
- Do not remove unique nuance about:
  - why the D7 full-500 result is caveated
  - why BioASQ is behaviorally active despite a flat endpoint
  - why evaluator choice still matters despite tied holdout binary accuracy
  - why the bridge failure mode matters scientifically

## Workflow

1. Edit only source shards and figure scripts.
2. Regenerate changed figures if their visible text changes.
3. Rebuild the manuscript.
4. Run verification.

## Required Verification

```bash
uv run python scripts/build_full_paper.py
uv run python scripts/build_full_paper.py --check
uv run python scripts/audit_ci_coverage.py
uv run pytest tests/test_build_full_paper.py
```

## Success Criteria

- D7 is visibly demoted on summary surfaces relative to FaithEval.
- Figure 4 caption honestly reflects the post-upgrade holdout tie.
- Figure 1–3 labels/titles match the manuscript’s current framing.
- No unrelated files are touched.
- `full_paper.md` rebuilds cleanly from source shards.

## Deliverables

Make the edits directly, then report:

- what changed
- any judgment calls made
- verification results
- any residual risks, if any
