# Notes Guide

Keep notes lean. Each fact should have one home, and everything else should point to it.

## Start Here

Use notes in this order:

1. `2026-04-21-claim-framing-governance.md` for current framing defaults and evidence routing.
2. `measurement-blueprint.md` for the evaluation contract and reporting rules.
3. `act3-reports/*.md` for canonical numbers, CIs, and claim-bearing audits.
4. `act3-sprint.md` for current execution status and priority order.
5. `research-log.md` for chronology, surprises, and reasoning shifts.

If a claim is going into the manuscript or site, the number should trace back to `act3-reports/*.md`, not to sprint notes or strategy docs.

Treat the governance note as the intended framing target for the manuscript as well. If paper files still reflect older framing, that is lag, not a reason to preserve the old hierarchy.

## What Lives Where

- `act3-reports/*.md`: canonical experiment audits, numbers, CIs, file paths, interpretation.
- `act3-sprint.md`: current priorities and status only.
- `research-log.md`: narrative of what changed and why.
- `measurement-blueprint.md`: metric definitions, judge rules, and reporting requirements.
- `runs_to_analyse.md`: queue of completed runs awaiting analysis.
- `scratchpad.md`: disposable working notes only.

## Current vs Historical

Current defaults:

- `2026-04-21-claim-framing-governance.md`
- `measurement-blueprint.md`
- `act3-reports/*.md`
- `act3-sprint.md`
- `research-log.md`

Historical or reference-only unless the task explicitly needs them:

- `2026-04-11-strategic-assessment.md`
- `optimise-intervention-ac3.md`
- `V2-critique-of-mentor-review-strategic.md`
- older handoff, critique, and outline-style notes

Do not treat older strategy or framing docs as current authority just because they are more detailed or rhetorically stronger.

## Run Lifecycle

After a claim-relevant run finishes:

1. Add it to `runs_to_analyse.md`.
2. Write the analysis in `act3-reports/YYYY-MM-DD-<name>.md`.
3. Add a short dated entry to `research-log.md`.
4. Update the status line in `act3-sprint.md`.
5. Remove the run from `runs_to_analyse.md`.

## Writing Rules

- Put numbers and uncertainty in reports, not in `act3-sprint.md`.
- Do not duplicate the same result across multiple notes.
- Do not rewrite old reports in place when superseded; add a pointer to the newer report.
- Do not keep analysed runs in `runs_to_analyse.md`.
- Do not put authoritative content in `scratchpad.md`.

## Practical Routing

- Framing question: start with `2026-04-21-claim-framing-governance.md`.
- Measurement or judge question: start with `measurement-blueprint.md`.
- Need a number or CI: find the relevant file in `act3-reports/`.
- Need to know current priority: open `act3-sprint.md`.
- Need to know what changed and why: open `research-log.md`.

This folder supports the live manuscript and site, but it is not the manuscript. Notes should stay operational, source-disciplined, and easy to update.
