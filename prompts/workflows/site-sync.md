Orchestrate and execute the integration of new results into /site, respecting existing patterns and avoiding editorial drift.

Objective:
Update /site so it reflects all materially relevant changes since the last full paper/site content update, while preserving all unaffected information that is still valid. Determine the exact boundary of required updates before editing.

Critical requirement:
Do not lose prior information that remains correct. This is an additive and corrective integration task, not a license for broad rewrites.

Inputs:
- Current /site content
- Current paper draft
- Latest canonical repo evidence, notes/reports/results
- Existing site patterns, data structures, and narrative conventions
- @multi-agent-coordination-patterns-five-approaches-.md

Orchestration requirement:
Think hard about the correct coordination pattern before acting. Avoid bloating the main context window. Default to an editor-in-chief main agent coordinating a small set of narrow high-reasoning subagents with written handoffs, unless the repo clearly requires a different pattern.

Use only GPT-5.4 and Codex 5.3.

Recommended subagent split:
1. Boundary-mapping agent:
   identify exactly what changed since the last full site/content update
2. Claim-dependency agent:
   map which site sections, disclaimers, visuals, and summaries are downstream of those changes
3. Content integration agent:
   patch the affected site content while preserving existing patterns
4. Drift/preservation auditor:
   verify that valid prior information was not accidentally dropped or weakened

Method:
1. Establish the exact update boundary first.
2. Build a dependency map from new evidence to impacted site claims/sections.
3. Edit only the files/sections that require change.
4. Preserve unaffected valid material.
5. Respect existing content patterns, schemas, and stylistic conventions.
6. Run a post-edit preservation audit:
   - what changed
   - what stayed intentionally unchanged
   - what was removed and why
7. Flag any unresolved uncertainties rather than papering over them.

Deliverables:
1. Boundary memo: what changed since the last full update, and why those changes matter
2. Implemented /site updates
3. File-by-file or section-by-section summary of changes
4. Preservation audit proving that unaffected valid content was retained
5. List of unresolved issues or judgment calls

Constraints:
- No broad rewrite unless the evidence truly forces it.
- No silent deletion of still-valid content.
- No drift from current paper and evidence.
- Maintain a careful editorial stance toward all writer/researcher subagents.