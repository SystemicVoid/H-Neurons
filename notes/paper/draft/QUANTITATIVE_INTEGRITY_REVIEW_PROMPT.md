# Quantitative Integrity Review Prompt

You are reviewing the quantitative integrity of the current **"Detection Is Not Enough"** draft.

## Primary files

- `notes/paper/draft/full_paper.md`
- `notes/paper/draft/number_provenance.md`

## Key supporting files

- `notes/measurement-blueprint.md`
- `notes/act3-reports/*.md`
- `data/gemma3_4b/intervention/**`
- `data/gemma3_4b/pipeline/**`
- `notes/paper/draft/figures/fig1_four_stage_scaffold.py`
- `notes/paper/draft/figures/fig2_matched_readouts.py`
- `notes/paper/draft/figures/fig3_bridge_failure.py`
- `notes/paper/draft/figures/fig4_measurement.py`

## Earlier audits for context only

- `notes/paper/draft/reviews/final_claim_audit.md`
- `notes/paper/draft/reviews/cross_section_audit.md`

## Current draft state

The last pass already:

- removed `[UNCERTAINTY]` tags,
- clarified the SAE slope split,
- added Appendix A detector/verbosity summaries,
- disambiguated Table 2's jailbreak delta,
- attached bridge dev-set caveats near promoted claims,
- and updated `number_provenance.md`.

Your job is to verify those fixes and extend review into any remaining quantitative weak spots.

## Your tasks

1. Verify that every promoted number in the abstract, tables, and section topic sentences matches its source.
2. Re-check the areas touched in the latest pass:
   - FalseQA provenance wording
   - BioASQ provenance wording
   - Appendix A detector / verbosity numbers
   - Table 2 jailbreak delta vs Section 6 slope distinction
   - bridge dev-set caveat placement near promoted values
3. Check that table numbering, box naming, and figure numbering are sequential and referenced in the body.
4. Check that each figure reference in the text matches what the figure script/image appears to contain.
5. Check whether any number in the assembled draft drifted from the section files during manual reassembly.

## Important rules

1. Treat `data/` files as acceptable canonical sources if the paper now explicitly says so.
2. Do **not** demand `act3-reports/` traces for numbers that are transparently sourced to `data/` or blueprint files, unless that sourcing choice itself is misleading.
3. Flag any place where a confidence interval, sample size, or evaluation split is missing but needed for claim safety.
4. Distinguish:
   - `VERIFIED`
   - `MISMATCH`
   - `AMBIGUOUS`
   - `MISSING CAVEAT`

## Output format

- Findings first, ordered by severity.
- For each finding:
  - `file:line`
  - quoted claim or number
  - checked source
  - `status = VERIFIED / MISMATCH / AMBIGUOUS / MISSING CAVEAT`
  - concrete fix
- End with:
  - `Quant claims I fully trust`
  - `Quant claims still too shaky for headline use`
