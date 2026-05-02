Goal: open one reviewable PR that keeps repository documentation, agent guidance, workflow notes, and doc-routing current with the actual repo state. Prefer durable usefulness over churn.

Start every run from `origin/main`:

- `git fetch origin main --prune`
- create a fresh branch named like `docs/daily-garden-YYYYMMDD-<shortsha>`
- read root `AGENTS.md` first, then any narrower `AGENTS.md` for paths you touch
- never run GPU jobs, API-spend jobs, long experiments, publication deploys, or remote infrastructure actions

State tracker:

- Search first for an existing documentation-gardening tracker with `rg -i "doc gardening|documentation gardening|docs maintenance|doc maintenance" docs notes prompts README.md AGENTS.md`.
- If none exists, create `docs/doc-gardening-state.md`.
- Maintain this tracker as the only cross-repo state file for the routine.
- It must record: last run date, base commit, PR link, changed files scanned, focus area, files edited, checks run, open follow-ups, and a coverage queue.
- The coverage queue should rotate through: root docs/guidance, `scripts/`, `scripts/infra/`, `scripts/lib/`, `data/`, `notes/`, `paper/icml/`, `site/`, `papers/`, `prompts/`, `tests/gold_labels/`, `scripts/infra/cloud/`, and `vendor/`.
- On each run, first handle docs affected by repo changes, then take the oldest or most important open queue item. If there is no substantive drift, update the tracker with the audited scope and evidence rather than manufacturing prose changes.

Repository-specific current-doc routing:

- Root `AGENTS.md` is for always-on defaults and pointers only.
- `README.md` is the project entrypoint and should stay concise, current, and repo-relative.
- `notes/AGENTS.md` controls notes routing. Treat `notes/icml/reports/`, `notes/icml/reviews/`, `notes/measurement-blueprint.md`, `notes/research-log.md`, and `notes/icml/mistral24b/` as current working surfaces.
- `notes/icml/mistral24b/AGENTS.md` says the canonical Mistral strategy/progress source is `notes/icml/mistral24b/2026-04-28-5.5-pro-l1-mitigation-strategy.md`; do not create competing Mistral trackers.
- `docs/quantitative-reporting-standards.md` governs claim-bearing quantitative docs.
- `site/AGENTS.md` and `site/site-maintenance-model.md` govern site docs and data-export drift.
- `data/AGENTS.md` governs committed run outputs, provenance, and `notes/runs_to_analyse.md`.
- `scripts/AGENTS.md`, `scripts/infra/AGENTS.md`, and `scripts/lib/AGENTS.md` govern code/workflow documentation in those areas.
- Historical or closed areas normally stay untouched: `docs/archive/`, `docs/act2-pre-pivot-archive/`, `notes/act3-reports/`, old sprint/strategy notes, and old handoffs. Edit them only when a current workflow depends on a stale pointer or when a minimal supersession note prevents real confusion.

Guidance and skill placement rule:

- Put agent guidance as low and as nested as practical.
- Root guidance should contain only instructions that apply everywhere.
- If guidance applies only to papers/literature, keep it under `papers/` or `papers/.agents/skills/`.
- If guidance applies only to prompts, site, tests, cloud infra, data, or notes, keep it in that subtree’s `AGENTS.md` or local skill directory.
- Do not broaden a local convention into root docs just because you discovered it.
- Preserve `CLAUDE.md` mirrors/symlinks when present; edit the canonical `AGENTS.md` source when they point there.
- Treat absolute symlinks to local skill stores as declarations only in the cloud. Do not chase or vendor private local paths. If a cloud-runner usability issue is real, document the limitation or add minimal repo-local guidance in the relevant subtree.

Editing policy:

- Prefer editing existing docs over creating new ones, except for the first-run state tracker.
- Keep PRs small: usually one coherent area plus the state tracker.
- Update current-state docs when code, scripts, tests, run layout, workflows, or canonical reports have changed.
- Trim stale current-state information. Remove low-value stale text outright when it is not historically useful.
- Preserve historical records unless they actively mislead current work; when needed, add a dated supersession pointer rather than rewriting history.
- Prefer one canonical source plus links over duplicated summaries across many files.
- Use repo-relative links. Avoid adding absolute local paths to current docs.
- Do not invent missing provenance, results, seeds, commands, claims, or workflow guarantees.
- For quantitative or claim-bearing changes, trace statements to committed reports/data and follow `docs/quantitative-reporting-standards.md`.
- For literature/paper work, start with `papers/INDEX.md` and update concise index references only from local committed paper files.

Useful inspection commands:

- `find . -name AGENTS.md -print`
- `rg --files -g '*.md' -g '*.mdx' -g '*.rst' -g '*.txt'`
- `git diff --name-status <last_doc_garden_commit_or_origin_main>..HEAD`
- `git log --name-status --since='14 days ago'`
- targeted `--help` commands for repo-specific CLIs before documenting them

Verification before commit:

- Always inspect `git diff`.
- Run `git diff --check`.
- Verify new or changed relative links point to real committed paths.
- If touching Python workflow docs, run the relevant cheap `--help` or targeted command when possible.
- If touching site claims or `docs/ci_manifest.json`, run `uv run python scripts/audit_ci_coverage.py`.
- If staging anything under `data/` or committed run outputs, first run `uv run python -m scripts.lib.pipeline active-run-status`.
- Run `prek run` on staged files if available; otherwise run the targeted checks you can justify.
- If a check is unavailable in the cloud, record the exact reason in the PR body and state tracker.

Self-audit the final diff:

- Does each changed doc match current repo state?
- Did you avoid broad rewrites and cosmetic churn?
- Did you avoid editing historical docs unless necessary?
- Did you preserve or improve canonical routing?
- Did you place new guidance at the narrowest useful scope?
- Did you update the state tracker after the actual work and checks?

Commit and PR:

- Stage only the intended files.
- Use a conventional commit, usually `docs(garden): ...`.
- Push the branch and open a PR against `main`.
- PR title should be conventional and specific.
- PR body must include: scope, why this area was chosen, changed files, stale info removed or superseded, tracker update, checks run, skipped checks with reasons, and remaining follow-ups.
- If uncertainty remains but the diff is useful, open the PR as draft and say exactly what needs human review.

Finally, you may have to go beyond recent changes and look at the docs at a higher level since the docs have not been gardened for a while in some cases, don't overanchor on recent changes rather deeply inspect the repo and reduce drift, staleness, duplication/redundancy (prefer linking), etc.. Take all your time, ultrathink and do your absolute best !