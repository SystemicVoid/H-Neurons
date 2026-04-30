# Doc Gardening State

Cross-run tracker for the documentation-gardening routine. Updated at the end of each run.

## Last Run

- **Date**: 2026-04-30
- **Base commit**: `2bededd` (origin/main)
- **Branch**: `docs/daily-garden-20260430-2bededd`
- **PR**: _pending_
- **Focus area**: Root docs/guidance + `scripts/infra/` (first-run: no prior state)

### Changed Files Scanned

All AGENTS.md files, README.md, docs/, notes/AGENTS.md, notes/icml/mistral24b/AGENTS.md,
scripts/AGENTS.md, scripts/infra/AGENTS.md, scripts/lib/AGENTS.md, data/AGENTS.md, site/AGENTS.md,
tests/gold_labels/AGENTS.md, vendor/AGENTS.md.

Recent repo changes audited: commits `3f311c6`–`2bededd` (last 20 commits, ~2026-04-29–30).

### Files Edited

- `README.md`: removed stale "framing governor" pointer to closed note
  `notes/2026-04-21-claim-framing-governance.md`; removed absolute local archive path
  (`/home/hugo/...`).
- `scripts/infra/AGENTS.md`: fixed stale `scripts/infra/lambda-AGENTS.md` pointer →
  `scripts/infra/cloud/runbooks/lambda.md`; replaced absolute local path in minimal wrapper
  example with `${PROJECT_DIR:-/workspace/02-h-neurons}` to match the env-variable pattern
  used in `mistral24b_replication.sh`.

### Checks Run

- `git diff --check`: passed (no trailing whitespace).
- Verified all README relative links resolve to committed paths.
- `prek` not available in this environment (not on PATH).
- Did not touch Python or site docs; `audit_ci_coverage.py` not applicable this run.
- Did not stage anything under `data/`; `active-run-status` not required.

### Stale Info Removed

- README framing-governor sentence (points to a closed doc; no longer the current
  framing source — see `notes/AGENTS.md` for current routing).
- README absolute external archive path (not repo-relative, no value to agents or readers).
- `scripts/infra/AGENTS.md` broken `lambda-AGENTS.md` reference (file does not exist;
  lambda docs live in `scripts/infra/cloud/runbooks/lambda.md`).
- `scripts/infra/AGENTS.md` hardcoded local machine path in wrapper example
  (replaced with env-variable form consistent with the actual wrapper scripts).

### Open Follow-Ups

- `scripts/infra/cloud/runbooks/lambda.md` still lists `scripts/infra/lambda-AGENTS.md`
  as a script reference (line 9). That file does not exist. Needs a follow-up pass to
  either drop the line or confirm whether it was meant to be `scripts/infra/lambda-bootstrap.sh`
  or a different file. Kept out of this PR because it is inside `cloud/runbooks/` scope.
- `notes/2026-04-21-claim-framing-governance.md` remains in notes/ as a closed historical doc.
  No action needed unless it is actively misleading a current workflow.

---

## Coverage Queue

Rotate through these areas. Check off when audited; re-add when material changes.

| Area | Last audited | Notes |
|------|-------------|-------|
| root docs/guidance (AGENTS.md, README.md) | 2026-04-30 | this run |
| `scripts/` | — | queue |
| `scripts/infra/` | 2026-04-30 | this run (lambda ref + path fix) |
| `scripts/lib/` | 2026-04-30 | reviewed, no drift |
| `data/` | 2026-04-30 | reviewed, no drift |
| `notes/` | 2026-04-30 | reviewed, no drift |
| `paper/icml/` | — | queue |
| `site/` | — | queue |
| `papers/` | — | queue |
| `prompts/` | — | queue |
| `tests/gold_labels/` | — | queue |
| `scripts/infra/cloud/` | — | queue (lambda.md has stale ref; follow-up) |
| `vendor/` | — | queue |
