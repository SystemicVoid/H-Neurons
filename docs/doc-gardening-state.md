# Doc Gardening State

Cross-run tracker for the documentation-gardening routine. Updated at the end of each run.

## Last Run

- **Date**: 2026-05-01
- **Base commit**: `147436d` (origin/main)
- **Branch**: `docs/daily-garden-20260501-147436d`
- **PR**: TBD
- **Focus area**: Broken-reference sweep (root docs, `scripts/infra/`, `scripts/infra/cloud/`, `papers/`) + follow-up from PR #1

### Changed Files Scanned

All AGENTS.md files, README.md, docs/, notes/AGENTS.md, notes/icml/mistral24b/AGENTS.md,
scripts/AGENTS.md, scripts/infra/AGENTS.md, scripts/lib/AGENTS.md, data/AGENTS.md, site/AGENTS.md,
tests/gold_labels/AGENTS.md, vendor/AGENTS.md, papers/INDEX.md.

Recent repo changes audited: commits `01ded23`–`147436d` (11 commits, 2026-04-30 to 2026-05-01).
Key changes in scope: `inhibit_suspend.sh` helper added/hardened (`f0f7256`, `147436d`),
RunPod template launch hardened (`637f311`, `f278827`, `8efb268`), mistral h1 c-sweep data + review
added (`28d5c85`, `a9e8551`), 8 new papers imported to INDEX (`74e4a11`).

### Files Edited

- `scripts/infra/cloud/runbooks/lambda.md`: removed stale `scripts/infra/lambda-AGENTS.md`
  reference (file never existed; follow-up from PR #1).
- `README.md`: replaced broken `docs/archive/project-structure.md` link with `data/AGENTS.md`
  (`docs/archive/` directory does not exist); removed
  `notes/2026-04-21-claim-framing-governance.md` from "Current project-facing reading order"
  (file is "Closed — do not write to" per `notes/AGENTS.md`; inappropriate as first item in
  current reading order).
- `scripts/infra/AGENTS.md` (and its `CLAUDE.md` symlink): updated Pipeline Wrapper Shape
  example to use `${PROJECT_DIR:-/home/hugo/...}` env-var override pattern instead of hardcoded
  assignment; env-var override is functionally required for the `inhibit_suspend` re-exec path
  to propagate `PROJECT_DIR` into the re-executed process (matches actual wrapper scripts).
- `papers/INDEX.md`: removed broken `notes/causal-decomposition-dossier/2026-04-30-probe-pivot-prior-art-discovery.md`
  path from HTML comment (directory does not exist anywhere in the repo).

### Checks Run

- `git diff --check`: passed (no trailing whitespace).
- Python link-checker: all relative links in `README.md` resolve to committed paths.
- `prek` not available in this environment (not on PATH).
- Did not touch Python or site docs; `audit_ci_coverage.py` not applicable this run.
- Did not stage anything under `data/`; `active-run-status` not required.

### Stale Info Removed

- `lambda.md` phantom `lambda-AGENTS.md` script reference (file does not exist; never did).
- README `docs/archive/project-structure.md` link (path does not exist; data layout is in
  `data/AGENTS.md`).
- README listing of `notes/2026-04-21-claim-framing-governance.md` as item 1 of the current
  reading order (that file is closed per `notes/AGENTS.md`; `notes/measurement-blueprint.md`
  is the correct first stop).
- `papers/INDEX.md` comment reference to a non-existent notes directory
  (`notes/causal-decomposition-dossier/`).

### Open Follow-Ups

- `scripts/infra/` wrappers (~37 scripts) still use ad-hoc `systemd-inhibit --what=sleep:idle`
  rather than sourcing the new `scripts/lib/inhibit_suspend.sh` helper. `scripts/lib/AGENTS.md`
  documents the helper as the intended replacement. No documentation action needed (the lib AGENTS
  correctly describes intent); actual migration is a code task, not a doc task.
- `notes/icml/reports/2026-04-30-mistral24b-h1-c-sweep-review.md` was added but not yet in
  the README "Current project-facing reading order". That list is already long; add only if it
  becomes a primary navigation anchor.
- `scripts/` (non-infra) still queue for a full pass: audit new scripts
  (`select_intervention_aware_c.py`, `validate_mistral24b_cp23.py`) against `scripts/AGENTS.md`
  guidance on output safety and resumability.

---

## Previous Run

- **Date**: 2026-04-30
- **Base commit**: `2bededd` (origin/main)
- **Branch**: `docs/daily-garden-20260430-2bededd`
- **PR**: `#1`
- **Focus area**: Root docs/guidance + `scripts/infra/` (first-run: no prior state)

### Files Edited (Previous Run)

- `README.md`: removed stale "framing governor" pointer to closed note
  `notes/2026-04-21-claim-framing-governance.md`; removed absolute local archive path
  (`/home/hugo/...`).
- `scripts/infra/AGENTS.md`: fixed stale `scripts/infra/lambda-AGENTS.md` pointer →
  `scripts/infra/cloud/runbooks/lambda.md`; replaced absolute local path in minimal wrapper
  example with `${PROJECT_DIR:-/workspace/02-h-neurons}` to match the env-variable pattern
  used in `mistral24b_replication.sh`.

---

## Coverage Queue

Rotate through these areas. Check off when audited; re-add when material changes.

| Area | Last audited | Notes |
|------|-------------|-------|
| root docs/guidance (AGENTS.md, README.md) | 2026-05-01 | README broken link + reading-order fix |
| `scripts/` | — | queue (new: `select_intervention_aware_c.py`, `validate_mistral24b_cp23.py`) |
| `scripts/infra/` | 2026-05-01 | PROJECT_DIR env-var pattern fix |
| `scripts/lib/` | 2026-04-30 | reviewed, no drift |
| `data/` | 2026-04-30 | reviewed, no drift |
| `notes/` | 2026-05-01 | reviewed, no drift; routing accurate |
| `paper/icml/` | — | queue (new: `2026-04-30-mistral24b-h1-c-sweep-review.md`) |
| `site/` | — | queue |
| `papers/` | 2026-05-01 | INDEX.md broken comment ref fixed |
| `prompts/` | — | queue |
| `tests/gold_labels/` | 2026-04-30 | reviewed, no drift |
| `scripts/infra/cloud/` | 2026-05-01 | lambda.md stale ref removed (follow-up from PR #1) |
| `vendor/` | 2026-04-30 | reviewed, no drift |
