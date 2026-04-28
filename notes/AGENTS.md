# Notes Guide

Keep notes lean. Each fact has one home; everything else points to it.

Framing is not fixed — challenge claims continuously. There is no single static framing source of truth.

## Current Routing (ICML)

Write all new claim-bearing analysis here:

- `icml/reports/YYYY-MM-DD-<name>.md` — experiment audits, numbers, CIs, file paths, interpretation.
- `icml/reviews/YYYY-MM-DD-<name>.md` — adversarial audits, framing reviews, integration reviews.

`icml/reports` and `icml/reviews` are symlinks into `paper/icml/{reports,reviews}/`. Write through the symlink.

## Start Here

1. `measurement-blueprint.md` — evaluation contract and reporting rules.
2. `icml/reports/*.md` — canonical numbers, CIs, claim-bearing audits.
3. `research-log.md` — chronology, surprises, reasoning shifts.

Manuscript and site numbers must trace to `icml/reports/*.md`.

## What Lives Where

- `icml/reports/*.md`: canonical experiment audits (current).
- `icml/reviews/*.md`: adversarial / framing / integration reviews (current).
- `research-log.md`: narrative of what changed and why.
- `measurement-blueprint.md`: metric definitions, judge rules, reporting requirements.
- `runs_to_analyse.md`: queue of completed runs awaiting analysis.
- `scratchpad.md`: disposable working notes.

## Run Lifecycle

After a claim-relevant run finishes:

1. Add it to `runs_to_analyse.md`.
2. Write the analysis in `icml/reports/YYYY-MM-DD-<name>.md`.
3. Add a short dated entry to `research-log.md`.
4. Remove the run from `runs_to_analyse.md`.

## Writing Rules

- Numbers and uncertainty live in `icml/reports/`.
- Do not duplicate the same result across notes.
- When superseded, add a pointer to the newer report; do not rewrite in place.
- Do not keep analysed runs in `runs_to_analyse.md`.
- Do not put authoritative content in `scratchpad.md`.
- Re-examine framing each time evidence changes; do not pin to any one framing doc.

## Practical Routing

- Measurement / judge: `measurement-blueprint.md`.
- Number or CI: `icml/reports/`.
- What changed and why: `research-log.md`.

## Closed — do not write to o
These directories and files are historical. Do not consult them for current decisions and do not write to them.

- `act3-reports/`
- `act3-sprint.md`
- `optimise-intervention-ac3.md`
- `2026-04-11-strategic-assessment.md`
- `2026-04-21-claim-framing-governance.md`
- `V2-critique-of-mentor-review-strategic.md`
- older handoff, critique, and outline-style notes
