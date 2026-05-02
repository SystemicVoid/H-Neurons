# Doc Gardening State

Cross-run tracker for the documentation-gardening routine. Updated at the end of each run.

## Last Run

- **Date**: 2026-05-02
- **Base commit**: `3243234` (origin/main)
- **Branch**: `docs/daily-garden-20260502-3243234`
- **PR**: `#10`
- **Focus area**: `site/` first audit (absolute-path fix + scratchpad removal in
  `site-maintenance-model.md`); README.md stale Mistral critical-path status

### Changed Files Scanned

All AGENTS.md files (root, scripts/, scripts/infra/, scripts/lib/, data/, notes/,
notes/icml/mistral24b/, site/, tests/gold_labels/, vendor/), README.md,
site/site-maintenance-model.md, notes/research-log.md (first 100 lines + tail for
current state), notes/icml/mistral24b/2026-04-28-5.5-pro-l1-mitigation-strategy.md
(header + verdict + CP table), paper/icml/reports/ (listing), paper/icml/reviews/ (listing),
scripts/select_intervention_aware_c.py, scripts/validate_mistral24b_cp23.py,
prompts/workflows/ (all files, no AGENTS.md present), papers/INDEX.md (quick-map section).

Recent repo changes audited: commit `3243234` (2026-05-01 garden run #2).
Key changes since last garden: none new since `3243234`; all 2026-04-30 commits
(CP5/H1 data+reviews, docker files, c-sweep infra) were already covered by PR #2.

### Files Edited

- `site/site-maintenance-model.md`: replaced absolute `/home/hugo/...` paths in the
  "Why it is only a partial fit" Clarity subsection with repo-relative links
  (e.g. `index.html`, `assets/shared.js`); removed 19-line informal scratchpad appended
  after the "Bottom Line" close (progressive-disclosure UX brainstorm, Options A/B/C notes
  — working draft that was never removed).
- `README.md`: updated stale Mistral status bullet in "Current Status" from "has passed
  CP2/CP3 gate, has not yet produced Mistral intervention evidence" to reflect that the full
  critical path (CP5 + H1 c-sweep) is complete and closed (both null FaithEval); removed
  the now-superseded standalone CP23-report link from that bullet.

### Checks Run

- `git diff --check`: passed (no trailing whitespace).
- Python link-checker: 36 README links checked; all resolve to committed paths (36/36 OK).
- Verified all new relative links in `site/site-maintenance-model.md` resolve from `site/`
  (`index.html`, `story.html`, `results/gemma-3-4b.html`, `methods.html`,
  `assets/shared.js`, `assets/charts.js` — all confirmed present).
- `prek run --files README.md docs/doc-gardening-state.md site/site-maintenance-model.md`: passed.
- `uv run python scripts/audit_ci_coverage.py`: passed.
- Did not stage anything under `data/`; `active-run-status` not required.

### Stale Info Removed

- `site/site-maintenance-model.md` absolute local paths in the Clarity subsection (six
  occurrences of `/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/site/…`).
- `site/site-maintenance-model.md` scratchpad / brainstorm block (lines 614–631): four-choice
  UX brainstorm ("Goal: Find world-class progressive disclosure…", Options A/B/C) appended
  after the document's closing paragraph; belongs in scratch notes, not the maintenance model.
- `README.md` stale "has not yet produced Mistral intervention evidence" framing: CP5 (full
  FaithEval H/controls) and H1 (intervention-aware C-sweep) are complete; path is closed.

### Audited Areas With No Changes

- `scripts/select_intervention_aware_c.py` and `scripts/validate_mistral24b_cp23.py`:
  both use `start_run_provenance()`/`finish_run_provenance()`, carry `SCHEMA_VERSION`
  constants, and follow `scripts/AGENTS.md` output-safety rules. No doc update needed.
- `prompts/workflows/` (12 files): no AGENTS.md present; checked for stale references
  (`/home/hugo`, deprecated file paths) — none found. No drift.
- `paper/icml/reports/` and `paper/icml/reviews/`: `2026-04-30-mistral24b-h1-c-sweep-review.md`
  is already linked from the canonical strategy memo header. No separate index needed.
- `papers/INDEX.md`: 8 papers imported on 2026-04-30 appear correctly in the quick-map
  across all relevant categories. No drift.
- All AGENTS.md files: no new drift found beyond issues fixed in PRs #1 and #2.

### Open Follow-Ups

- `scripts/infra/` wrappers (~37 scripts) still use ad-hoc `systemd-inhibit --what=sleep:idle`
  rather than sourcing `scripts/lib/inhibit_suspend.sh`. Code task, not a doc task; lib AGENTS
  already describes the intent.
- `site/extensions.html` roadmap is intentionally manual and periodically goes stale; noted
  in `site-maintenance-model.md` as "needs periodic pruning." Worth a focused pass when
  the Mistral null is formally reflected in manuscript/site.

---

## Previous Run

- **Date**: 2026-05-01
- **Base commit**: `147436d` (origin/main)
- **Branch**: `docs/daily-garden-20260501-147436d`
- **PR**: `#2`
- **Focus area**: Broken-reference sweep (root docs, `scripts/infra/`, `scripts/infra/cloud/`, `papers/`) + follow-up from PR #1

### Files Edited (2026-05-01)

- `scripts/infra/cloud/runbooks/lambda.md`: removed stale `scripts/infra/lambda-AGENTS.md`
  reference (file never existed; follow-up from PR #1).
- `README.md`: replaced broken `docs/archive/project-structure.md` link with `data/AGENTS.md`;
  removed `notes/2026-04-21-claim-framing-governance.md` from reading order.
- `scripts/infra/AGENTS.md`: updated Pipeline Wrapper Shape to use `${PROJECT_DIR:-/home/hugo/…}`
  env-var override pattern.
- `papers/INDEX.md`: removed broken `notes/causal-decomposition-dossier/` comment reference.

---

## Previous Run (2026-04-30)

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
| root docs/guidance (AGENTS.md, README.md) | 2026-05-02 | README Mistral status updated (CP5+H1 closed) |
| `scripts/` | 2026-05-02 | new scripts audited, compliant; no AGENTS.md drift |
| `scripts/infra/` | 2026-05-01 | PROJECT_DIR env-var pattern fix |
| `scripts/lib/` | 2026-04-30 | reviewed, no drift |
| `data/` | 2026-04-30 | reviewed, no drift |
| `notes/` | 2026-05-01 | reviewed, no drift; routing accurate |
| `paper/icml/` | 2026-05-02 | h1-c-sweep-review already in strategy memo header; no drift |
| `site/` | 2026-05-02 | absolute-path fix + scratchpad removal in site-maintenance-model.md |
| `papers/` | 2026-05-01 | INDEX.md broken comment ref fixed |
| `prompts/` | 2026-05-02 | 12 workflow files checked; no AGENTS.md, no stale refs |
| `tests/gold_labels/` | 2026-04-30 | reviewed, no drift |
| `scripts/infra/cloud/` | 2026-05-01 | lambda.md stale ref removed (follow-up from PR #1) |
| `vendor/` | 2026-04-30 | reviewed, no drift |
