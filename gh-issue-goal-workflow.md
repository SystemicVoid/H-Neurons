# GitHub Issue Goal Workflow

Use this document as the operating contract for any `/goal` session launched
from a GitHub issue. The chat prompt supplies the issue number and title; this
document defines how to complete the issue without losing engineering judgment.

## Objective

Fully address GitHub issue #[ISSUE_NUMBER], "[ISSUE_TITLE]," without stopping
until the issue is implemented, independently reviewed, verified, committed,
and either merged or left in a clearly merge-ready state with the exact external
blocker documented in GitHub.

Treat the issue as the source of intent and the repository as the source of
truth. If they disagree, prefer current tested behavior for compatibility, then
record the interpretation in the issue or PR before making risky changes.

## Read First

- `AGENTS.md` and any narrower `AGENTS.md` files for touched areas
- `CONTEXT.md`, relevant ADRs, and repo docs named by the issue
- `gh issue view [ISSUE_NUMBER] --comments`
- Related open issues, PRs, commits, and ADRs that affect ordering or scope
- Current code, tests, scripts, data contracts, and artifacts that implement the
  behavior described by the issue
- `.pre-commit-config.yaml` before staging generated data or run outputs

## Operating Principles

- Preserve claim-bearing semantics first: public contracts, stored artifact
  shapes, filenames, hashes, provenance fields, and compatibility guards remain
  stable unless the issue explicitly requires migration.
- Prefer domain-named Modules and Interfaces for claim-bearing behavior. Generic
  helpers are fine for mechanics such as JSONL IO, hashing, path handling, or
  formatting, but they should not absorb the scientific meaning owned by domain
  Modules.
- Strict validation means fixing real failures, not relaxing checks, deleting
  assertions, or narrowing tests to make the run pass.
- Do not widen the issue into a backlog item. File or link follow-up issues for
  adjacent discoveries instead of bundling them silently.
- Keep moving without hiding risk. A failed test, type error, lint finding, or
  review finding in touched behavior is work to fix, not a blocker.
- A hard blocker is external: missing credentials, unavailable remote service,
  required hardware that is not accessible, conflicting requirements with no
  safe local assumption, or lack of merge permission. Document hard blockers in
  GitHub and continue any remaining in-scope work that reduces risk.

## Impact Map Before Editing

Before changing code, write a compact map in the progress log:

- the issue outcome in one sentence
- protected contracts and artifacts
- likely implementation files and tests
- dependency order across related issues
- explicit non-goals
- the validation evidence that will prove completion
- the rollback or compatibility story if behavior changes

If a prerequisite issue is clearly required and higher impact, either address
the prerequisite first when it is within the same architectural slice, or record
why this issue can proceed independently. If ordering is ambiguous, make the
least risky local assumption and document it.

## Implementation Workflow

Work in small checkpoints. At each checkpoint, state the current objective,
files changed, validation run, and remaining risk.

Use existing repo patterns, package managers, scripts, and style. Prefer editing
existing files unless a new Module, ADR, test file, or documentation file is
clearly justified. Keep entrypoints thin when extracting shared behavior, but
leave enough Adapter coverage to prove the CLI or workflow still calls the new
Module correctly.

Preserve unrelated user changes. Check `git status --short` before editing and
before staging. If a file contains unrelated user edits, inspect the diff and
stage only your own hunks. Do not revert, overwrite, or normalize unrelated
changes.

If progress on the main implementation is temporarily blocked, continue with
productive in-scope work: characterization tests, compatibility tests, docs,
ADR notes, smaller dependency extraction, review cleanup, or issue/PR updates.
Do not invent unrelated cleanup just to stay busy.

## Success Criteria

- The issue body and relevant comments are satisfied or explicitly superseded by
  a documented repo-truth interpretation.
- Architectural intent is reflected in code structure, tests, docs, or ADRs
  where appropriate.
- Current behavior remains compatible unless the issue requires a migration.
- Claim-bearing outputs, run artifacts, JSON schemas, hashes, filenames, and
  provenance semantics are stable or migrated with explicit tests and docs.
- Shared logic is centralized only where it clarifies ownership or reduces real
  duplication.
- Tests directly cover new Module or Interface behavior and retain minimal
  Adapter/entrypoint coverage.
- New follow-ups are filed or linked, not hidden in final-summary prose.
- No silent deferrals: any CodeRabbit or reviewer finding triaged as DEFER at
  Warning/Major severity or higher is converted to, or linked with, a GitHub
  issue and mentioned in the originating review thread with rationale.

## Validation Loop

1. Run focused tests for changed behavior first.
2. Add or update targeted tests for new Module behavior, migration behavior,
   compatibility guards, or regression surfaces.
3. Run broader validation before claiming done:
   - `uv run pytest`
   - `ruff check scripts tests`
   - `ruff format scripts tests --check` or run format, then recheck
   - `ty check`
4. If the issue touches frontend, site, data exports, long-running jobs, shell
   infra, or another scoped area, also run the checks named by the relevant
   `AGENTS.md`.
5. If staging data or output paths, run
   `uv run python -m scripts.lib.pipeline active-run-status` before staging.
6. If a check fails, classify it as an introduced regression, related existing
   failure, unrelated existing failure, or external unavailability. Fix
   introduced and related failures. Do not mutate unrelated areas just to make a
   broad suite green; document them and keep issue-specific validation strong.
7. If a required check cannot run, record the exact command, failure reason,
   next-best evidence, and whether this blocks merge.

## Independent Review

After local validation passes for the changed surface, run the
`$coderabbit-review` workflow. The `$` denotes the repo helper wrapper described
in `.agents/skills/coderabbit-review/SKILL.md`; prefer that wrapper over direct
`coderabbit` invocation. Use the direct CLI only to discover or debug the
underlying binary:

- `coderabbit --version`
- `coderabbit review --help`

Then invoke the wrapper with the selected review flags:

```bash
uv run python scripts/infra/coderabbit_review_watch.py -- --type uncommitted [other-flags]
```

Run a bounded review over the changed code. Triage findings as:

- MUST_FIX: Critical findings and high-confidence in-scope Major findings
- DEFER: Minor, Trivial, Info, low-confidence Major, stylistic, out-of-scope,
  risky, or decision-requiring findings
- IGNORE: duplicates or clear false positives

Fix MUST_FIX items, rerun relevant validation, and re-review within a bounded
loop. Do not chase purely stylistic or out-of-scope comments. Do not claim
completion while any unresolved Critical or in-scope high-confidence Major
remains.

Also perform a local self-review of the final diff: search for stale duplicate
helpers, dead imports, missing docs references, brittle tests, and accidental
schema or artifact changes.

## Commit, PR, and Merge

- Inspect `git status --short` before staging.
- Stage only files changed for this issue.
- Create atomic conventional commits with WHY-focused bodies.
- If repo policy and credentials allow it, open or update the PR, wait for
  required checks, resolve review feedback, and merge.
- If merge is not permitted from this environment, leave the branch merge-ready,
  update the issue or PR with validation and blocker details, and name the exact
  blocker in the final response.

## Stop Rules

Stop only when:

- GitHub issue #[ISSUE_NUMBER] is fully addressed according to the issue,
  comments, and success criteria above.
- Local validation is green for the changed surface and any unavailable,
  unrelated, or external check is precisely documented.
- CodeRabbit has no unresolved MUST_FIX findings.
- Changes are committed atomically.
- The work is merged, or merge-ready with a documented external blocker.
- The final response includes changed files, commits, validation
  commands/results, CodeRabbit triage, PR/merge status, follow-up issues, and
  residual risks.

If this workflow is being run under a multi-issue `/goal`, return a clear
COMPLETED, BLOCKED, or DEFERRED status for this issue. A blocked issue should
not stall the whole run after all useful in-scope work has been completed and
the blocker is documented in GitHub.
