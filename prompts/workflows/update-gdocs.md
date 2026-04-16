Use gws CLI Docs to update the Google Doc below, but ONLY the “Main-write-up” tab:
https://docs.google.com/document/d/1XUTmk7EQHSNh2vNky1FSupxEKt3HYbEXRl5Li-MEwLI/edit?tab=t.3fhkjrva311q

Do NOT touch any other tab.

Objective:
Bring the Main-write-up fully in line with the latest current story and evidence, especially all material changes since the 2026-04-12 content update. The result must be faithful to the current paper, scientifically calibrated, and polished in content and prose.

Scope:
- Read the current Main-write-up.
- Compare it against the latest canonical repo evidence, notes/reports/results, and the current paper draft.
- Update only what is outdated, incomplete, miscalibrated, or missing.
- Preserve content that is still valid and strong.

Subagent orchestration:
Use only GPT-5.4 and Codex 5.3. Use subagents only where they pay off. Recommended specialists:
1. Evidence delta checker
2. Scientific writer / prose polisher
3. Figures/tables consistency checker
4. References / citation consistency checker

Method:
1. First establish the evidence delta since the last meaningful Main-write-up update.
2. Identify all claims, caveats, figures/tables references, and narrative sections that must change.
3. Update the Main-write-up so it reflects the current evidence and paper positioning.
4. Preserve unaffected content rather than rewriting mechanically.
5. Tighten claim wording where the evidence is narrower than the prior prose.
6. Improve clarity, flow, transitions, and scientific polish without diluting technical precision.

Editorial standard:
- The narrative must match the current evidence, not legacy framing.
- Avoid hype, vagueness, and unearned certainty.
- Keep the document polished, concise, and career-grade.
- Treat this as a high-stakes scientific artifact.

Deliverables:
1. The updated Main-write-up tab in the Google Doc
2. A concise change log covering:
   - sections changed
   - major claims updated
   - important caveats/disclaimers added or revised
   - unresolved evidence gaps or decisions deferred

Hard constraints:
- ONLY edit the Main-write-up tab.
- Do not touch other tabs.
- Do not lose valid existing content.
- Use only high-reasoning flagship models.
- Prevent and correct redundancy in the paper, convey the points with minimal fluff and don't repeat information in many places if not absolutely necessary.

---

 Do a final publication-readiness review of `Main-write-up` only.

  Context:
  A previous pass already synced the draft to current repo evidence, updated Figures 2–4 and related prose, refreshed
  Sections 4–6 and Limitations, and restored Appendix B–E. The main known residual risks are:
  1. appendix tables were restored as plain text blocks rather than native Google Docs table objects
  2. visible table numbering still follows the draft’s legacy scheme rather than a full sequential renumbering pass

  Your job is not to re-summarize the draft. Your job is to find the remaining issues that would block submission or make the
  paper look unfinished.

  Review priorities:
  - publication-level polish, coherence, and internal consistency
  - incorrect, overstated, ambiguous, or under-qualified claims
  - mismatches between prose, tables, figures, captions, and appendix
  - table/figure numbering, naming, caption quality, and cross-references
  - statistical reporting quality: every quantitative claim should have appropriate uncertainty / interval language where
  expected, and sample sizes should be present where needed
  - terminology consistency across sections (for methods, benchmarks, evaluators, selectors, baselines, “bridge”, “holdout”,
  “endpoint”, “slope”, etc.)
  - wording that is too casual, defensive, redundant, or draft-like
  - transitions and narrative flow, especially where Section 4 anchors on FaithEval and D7 is supporting
  - appendix presentation issues that would matter for a publication-ready version
  - any leftover editorial artifacts, stale notes, placeholders, duplicated ideas, or formatting problems

  Instructions:
  - Be strict and act like an external reviewer or final internal editor preparing this for submission.
  - Prioritize findings that materially improve publication readiness.
  - Do not suggest broad speculative rewrites unless there is a clear problem.
  - Do not focus on repo process, only on the manuscript.
  - If something is correct but still not publication-ready in presentation, call that out.
  - If a statement is potentially true but too strong for the evidence as written, flag it.

  Output format:
  1. Findings first, ordered by severity.
  2. For each finding, include:
     - severity (`high`, `medium`, or `low`)
     - exact location (section / table / figure / appendix)
     - a short quote or precise description of the problematic text
     - why it is a problem
     - the minimal concrete fix
  3. Then include:
     - `Open questions / assumptions`
     - `Publication-ready if fixed?` with a yes/no judgment and one-sentence rationale
