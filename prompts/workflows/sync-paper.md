Conduct a research-grade audit of the current paper draft before any edits are made.

Objective:
Determine whether the draft is internally coherent, whether each substantive claim is actually supported by the current evidence, whether the framing is calibrated, and whether the figures/tables are pulling their weight.

Scope:
- Review the current paper draft in full.
- Review the current repo evidence, latest canonical notes/reports/results, and any relevant best-practices material for AI/ML research writing and figures/tables.
- Focus on claim validity, internal coherence, narrative discipline, figure/table quality, redundancy, clarity, and reference integrity.

Source hierarchy:
- Prefer the latest canonical evidence artifacts and reports over older summaries or outdated prose.
- Do not preserve a claim just because it already appears in the draft.

Subagent orchestration:
Use only high-reasoning subagents where specialization materially helps. Recommended split:
1. Claim-evidence auditor
2. Figure/table auditor
3. Prose/structure auditor
4. Citation/reference integrity auditor

Method:
1. Build a claim ledger for the draft.
2. For each nontrivial claim, classify it as:
   - supported
   - partially supported / overstated
   - unsupported
   - stale / outdated
   - internally inconsistent
3. Check whether the narrative matches the data we actually found, not the story we previously thought we had.
4. Audit figures and tables for:
   - whether they earn inclusion
   - whether they show the right comparisons and controls
   - whether they hide important uncertainty, caveats, or boundary conditions
   - whether any are overloaded, redundant, or under-explained
5. Audit prose for:
   - redundancy
   - vagueness
   - imprecise causal language
   - weak transitions
   - evaluator-centric or benchmark-fragile overclaiming
   - mismatch between evidence strength and wording

Deliverables:
Produce a structured audit with:
1. Executive verdict on overall paper quality
2. Claim-by-claim audit table with evidence status and recommended fix
3. Internal coherence / framing issues
4. Figure and table audit
5. Prose and structure audit
6. Prioritized fix list, ordered by impact on scientific quality

Constraints:
- Do not edit the draft yet.
- Do not optimize for diplomacy; optimize for truth, precision, and publication-quality rigor.
- Follow best practices in AI/ML research writing.