# Paper Draft Agent Guardrails

This directory is the paper-authoring surface. Treat it like source code with a build step.

## Source Of Truth

1. `notes/2026-04-11-strategic-assessment.md` owns the paper's earned / not-earned boundary.
2. `notes/act3-reports/*.md` own numbers, CIs, and experiment-level conclusions.
3. Section shards in this directory own manuscript prose.
4. `full_paper.md` is generated output, not an editing target.

## Non-Negotiables

- Do not edit `full_paper.md` by hand.
- Do not weaken or drop still-valid caveats just because a section is being revised.
- Do not replace canonical report provenance with higher-level summaries.
- Do not leak internal project jargon into reader-facing prose when a paper-facing term exists.
- Do not rewrite unaffected sections opportunistically; preserve valid existing content.

## Required Workflow

1. Check claim boundary before editing:
   - confirm the claim is earned in the strategic assessment
   - if not, write it with the required caveat or update the strategic assessment first
2. Edit only the relevant source shard(s):
   - `front_matter.md`, `abstract.md`, `section_*.md`, `references.md`, `appendix.md`
   - support files such as `number_provenance.md` and figure scripts only when needed
3. If visible figure text changes, update the figure script and regenerate the asset.
4. Rebuild the manuscript:
   - `uv run python scripts/build_full_paper.py`
5. Verify the build is clean:
   - `uv run python scripts/build_full_paper.py --check`
6. If quantitative reporting surfaces changed, run:
   - `uv run python scripts/audit_ci_coverage.py`
7. If the builder or manifest changed, run:
   - `uv run pytest tests/test_build_full_paper.py`

## Assembly Rules

- Assembly inputs are defined only by `assembly_manifest.json`.
- `front_matter.md`, `references.md`, and `appendix.md` are source shards, not compiled leftovers.
- `reviews/*.md` and `number_provenance.md` are not assembly inputs.
- Keep `full_paper.md` reproducible from shards with no hidden manual state.

## Current Paper-Specific Guardrails

- Treat the **2026-04-16 two-seed current-state jailbreak selector audit** as the live current-state source.
- Treat the **2026-04-14 current-state jailbreak selector audit** as a superseded reconciliation note unless a section explicitly discusses the pre-seed-2 state.
- Treat the **2026-04-08 jailbreak audit** as historical provenance unless a section explicitly discusses the legacy panel.
- Keep the selector-comparison result benchmark-local and caveated unless new evidence actually closes the mechanism question.
- Keep the clean pilot selector comparison separate from the noisier full-500 mixed-ruler panel.

## Writing Standard

- Optimize for scientific honesty over rhetorical neatness.
- Prefer explicit caveats to implied certainty.
- When in doubt, preserve the stronger provenance trail and the narrower claim.
