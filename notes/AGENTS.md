# Notes Folder — Documentation Structure

**Rule: each fact lives in exactly one place. Everything else links to it.**

## File Roles

| File | Owns | Never contains |
|---|---|---|
| `act3-sprint.md` | What we're doing, in what order, current status (one line + link per deliverable) | Numbers, CIs, inline results |
| `research-log.md` | Chronological narrative: what happened, what surprised us, what changed our thinking | Raw data tables, plot file paths |
| `act3-reports/*.md` | Authoritative per-experiment audit: numbers, CIs, data file paths | Forward-looking plans |
| `plot-registry.md` | Data file → chart mapping for site generation | Anything else |
| `runs_to_analyse.md` | Ephemeral GPU job queue | Analysed runs (remove them, don't mark done) |

## After Each Run

1. Run completes → add entry to `runs_to_analyse.md`
2. Analysis done → write `act3-reports/YYYY-MM-DD-<name>.md` with all numbers
3. Add one dated entry to `research-log.md` (narrative + link to report)
4. Update one status cell in `act3-sprint.md`: `**done** — [report-name.md](./act3-reports/report-name.md)`
5. Remove entry from `runs_to_analyse.md`

## Superseding a Result

Add a header note to the old report pointing to the new one. Leave the old report body frozen — don't edit numbers in place. The new report is the authority; the old one is historical context.

## What Not To Do

- Do not write numbers or CIs into `act3-sprint.md` status cells — they will go stale.
- Do not add new entries to `act3-reports/week3-log.md` — it is a frozen archive for week-3 data (D1 FaithEval/Jailbreak, D2, D3, D3.5).
- Do not keep "done" entries in `runs_to_analyse.md` — remove them once the report is written.
- Do not duplicate the decision rationale across sprint.md and research-log.md — research-log owns it.
