# Notes Folder — Documentation Structure

**Rule: each fact lives in exactly one place. Everything else links to it.**

DO NOT update the notes/paper/draft/full_paper.md directly unless it's absolutely necessary. ALWAYS prefer editing individual sections, and programatically recreating the final paper, avoiding drift and edits in multiple places.

## Project Phases (Chronological)

The notes folder documents three overlapping phases. Each phase has its own authoritative document; later phases reference earlier ones rather than duplicating content.

| Phase | Period | Authoritative document | What it owns |
|---|---|---|---|
| **Sprint execution** | Act 3 start → ongoing | `act3-sprint.md` + `act3-reports/*.md` | Deliverable status, per-experiment audits with numbers and CIs |
| **Experiment-discovery strategy** | 2026-04-01 → 2026-04-08 (terminal) | `optimise-intervention-ac3.md` | Five-stage plan, gate decisions, terminal outcomes; links to all canonical reports |
| **Paper framing** | 2026-04-11 → artifact deadline | `2026-04-11-strategic-assessment.md` | Title, evidence inventory, earned/not-earned claims, BlueDot submission plan, 2-week execution roadmap |

The measurement blueprint (`measurement-blueprint.md`) is concurrent with the sprint and applies across all phases as the evaluation contract.

**Phase transitions:** The optimise doc built on the sprint's experimental reports and measurement blueprint. The strategic assessment built on the optimise doc's terminal outcomes (§10) to derive the paper's thesis and structure. Each layer adds synthesis and framing; none replaces the raw evidence in `act3-reports/*.md`.

## File Roles

| File | Owns | Never contains |
|---|---|---|
| `act3-sprint.md` | What we're doing, in what order, current status (one line + link per deliverable) | Numbers, CIs, inline results |
| `research-log.md` | Chronological narrative: what happened, what surprised us, what changed our thinking | Raw data tables, plot file paths |
| `act3-reports/*.md` | Authoritative per-experiment audit: numbers, CIs, data file paths | Forward-looking plans |
| `act3-reports/plot-registry.md` | Data file → chart mapping for site generation | Anything else |
| `runs_to_analyse.md` | Ephemeral GPU job queue | Analysed runs (remove them, don't mark done) |
| `measurement-blueprint.md` | Evaluation rules, measurement standards, negative-control requirements, MDE table | Experiment results, status updates |
| `optimise-intervention-ac3.md` | Five-stage experiment-discovery strategy; stage gates and terminal outcomes (§10) | Raw numbers (links to reports); paper-level claims |
| `2026-04-11-strategic-assessment.md` | Paper thesis, three-layer evidence inventory, earned/not-earned boundary (§3, §10), paper structure, BlueDot criteria mapping, 2-week experiment priority stack | Experiment numbers (cites reports via optimise doc) |
| `scratchpad.md` | Ephemeral working notes, draft reasoning | Anything authoritative |

## After Each Run

1. Run completes → add entry to `runs_to_analyse.md`
2. Analysis done → write `act3-reports/YYYY-MM-DD-<name>.md` with all numbers
3. Add one dated entry to `research-log.md` (narrative + link to report)
4. Update one status cell in `act3-sprint.md`: `**done** — [report-name.md](./act3-reports/report-name.md)`
5. Remove entry from `runs_to_analyse.md`

## Paper-Writing Workflow

The strategic assessment defines three layers of evidence and maps them to paper sections (§5). When drafting or reviewing paper content:

1. **Claim → earned status**: check `2026-04-11-strategic-assessment.md` §3 and §10 before writing any claim. If a claim is "partially earned" or "not earned," do not state it without the required qualification.
2. **Number → canonical report**: every number in the paper traces back to an `act3-reports/*.md` file. Do not pull numbers from the optimise doc or strategic assessment — those cite reports, they are not the source of truth.
3. **Strategy → optimise doc**: for experiment rationale, gate decisions, and why branches were tried or closed, `optimise-intervention-ac3.md` §10 owns the terminal summary. The strategic assessment reframes this for the paper narrative but does not override the experimental record.
4. **Narrative → research-log.md**: for what surprised us and what changed our thinking, the research log is authoritative.
5. **Jargon → translation**: the strategic assessment §9 defines a translation guide (D7 → gradient-based causal intervention, csv2_yes → strict harmfulness rate, etc.). Use paper-facing language in any draft content.

## Superseding a Result

Add a header note to the old report pointing to the new one. Leave the old report body frozen — don't edit numbers in place. The new report is the authority; the old one is historical context.

## What Not To Do

- Do not write numbers or CIs into `act3-sprint.md` status cells — they will go stale.
- Do not add new entries to `act3-reports/week3-log.md` — it is a frozen archive for week-3 data (D1 FaithEval/Jailbreak, D2, D3, D3.5).
- Do not keep "done" entries in `runs_to_analyse.md` — remove them once the report is written.
- Do not duplicate the decision rationale across sprint.md and research-log.md — research-log owns it.
- Do not add any specific results into this guidance file, it serves as guidelines for reports.
- Do not treat the strategic assessment as the source of truth for numbers — it synthesizes and cites reports, it does not replace them.
- Do not edit the optimise doc's terminal status summary (§10) — it is a frozen record of the experiment-discovery phase. Append pointers to new work below §10; do not rewrite concluded stages.
- Do not make paper claims that exceed the earned/not-earned boundary in the strategic assessment §3 and §10. If new evidence shifts a boundary, update the strategic assessment first.
