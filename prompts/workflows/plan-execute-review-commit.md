# Plan Slice Execution and Review Gate

## Prompt

Implement the plan below as one or more independently reviewable work slices. Treat the plan as the source of user intent, but re-read repository files, scoped instructions, tests, scripts, data artifacts, and canonical docs as needed. Do not assume prior agent context is complete or current.

Your role is orchestrator first and implementer second. Drive each slice to a trustworthy stopping point: implemented, checked, reviewed from the right angles, fixed, re-reviewed, and committed atomically when commits are requested.

## Definition of Done

A slice is done only when all of these are true:

- The slice has a clear acceptance criterion and independent commit boundary.
- Relevant checks pass, or any skipped check has a concrete reason.
- A fresh review pass finds no remaining actionable issues, or the user explicitly waives them.
- The final diff contains only intended changes for that slice.
- Any required ledger/status update is made after implementation, verification, and review fixes.

Prefer smaller slices when the plan is broad. Do not continue into the next slice if the current one is unreviewed, uncommitted, or still carrying unresolved blockers.

## Context Discipline

Keep the main thread lean. The orchestrator owns scope, sequencing, context budget, reviewer selection, final judgment, staging, and commits; workers and reviewers own bounded investigations, patches, and risk lenses.

- Give agents narrow handoffs: goal, constraints, owned files or read-only scope, relevant commands, and expected output.
- Require concise returns: changed files, commands run, results, findings fixed, remaining risks, and user decisions needed.
- Do not paste large logs, broad repo context, or full review transcripts into the main thread unless necessary.
- If context is getting crowded, finish or commit the current slice before starting another. Prefer delegated fresh review/fix passes over pulling all debugging context into the orchestrator.

## Slice Loop

1. Inspect the working tree and relevant `AGENTS.md` files. Identify unrelated dirty files and leave them untouched.
2. Convert the plan into a short checklist of slices. For each slice, record scope, likely files, risks, acceptance checks, and commit boundary.
3. Re-read implicated files and tests. Prefer existing patterns over new abstractions.
4. State the intended edit before modifying files.
5. Implement locally or assign a worker with a disjoint write scope. Workers may edit only their assigned scope and must not stage or commit.
6. Run the smallest checks that can catch likely regressions early, then broaden checks as risk increases. If checks fail, fix and rerun before review.
7. If the task is materially ambiguous, unsafe, spend-sensitive, or irreversible, stop and ask the user or run only non-spend/dry-run checks.

Guardrails:

- If the slice touches data, outputs, or run artifacts, run the repository's active-run or live-output guard before staging or restructuring those paths.
- If a guard or validator prevents expensive work from starting with bad inputs, it must run before any GPU/API-capable command.
- Spend-sensitive work defaults to non-spend checks and dry-runs. A dry-run must prove that GPU/API/expensive work is skipped.
- Record machine-readable provenance where appropriate: source paths, content hashes, IDs/counts/fingerprints, configs, commands, and output paths.
- Preserve unrelated user or prior-agent changes. Do not delete, revert, stage, or commit unrelated dirty files.

## Review Harness

After initial implementation and verification, choose the smallest reviewer set that covers the slice's real risks. Do not hardcode reviewer count: use one reviewer for small low-risk changes, two for ordinary code/data changes, and more only when distinct high-risk lenses are needed.

Useful review lenses include:

- Correctness, edge cases, regression risk, tests, typing, lint, maintainability.
- Data/schema/provenance, backward compatibility, spend safety, guard ordering, irreversible actions.
- Research claim calibration, paper/ledger consistency, citation integrity.
- UI/UX behavior, accessibility, responsive layout, security, privacy, migrations, concurrency, deployment risk.

Reviewer instructions:

- Review the current tree, not just the orchestrator summary.
- Produce concrete file/line findings ordered by severity.
- Say explicitly when no actionable issues remain.
- If assigned `review+fix`, fix only clear in-scope issues you found, preserve unrelated changes, run targeted checks, and report the exact patch and verification. Do not stage or commit.
- If an issue is architectural, ambiguous, broad, or outside assigned scope, report it instead of fixing it.
- Do not spawn further agents unless the orchestrator explicitly assigns that role.

## Fix and Re-Review Gate

1. Treat blockers as mandatory fixes unless the user waives them.
2. Fix low/medium findings when the patch is small, aligned with the slice, and reduces future risk.
3. Prefer assigning clear reviewer-found issues back to the reviewer as `review+fix` when that avoids pulling the full debugging context into the orchestrator.
4. After fixes, rerun focused checks.
5. If blockers or nontrivial fixes occurred, run a fresh-pass review focused on residual risk and side effects. This may be the original reviewer for closure plus one independent reviewer when confidence matters.
6. The gate closes only when a new review pass surfaces no actionable issues, or remaining issues are explicitly documented and waived.

## Ledger Update

Update the canonical ledger or status document only after implementation, verification, and review fixes are complete. If the plan says not to create a new planning/status document, update only the named canonical document.

Record:

- What artifacts or behavior changed.
- Exact dry-run or verification commands.
- Check results.
- Remaining blockers before expensive, claim-bearing, or irreversible work.
- Confirmation that no competing status document was created, when relevant.

## Commit Flow

If commits are requested:

1. Inspect the final working tree.
2. Group changes into commits that can be reverted independently.
3. Stage only one logical group at a time.
4. Run relevant checks for the staged group, or cite checks already run if they exactly cover it.
5. Commit with the requested conventional subject and a WHY-focused body.
6. Repeat until all completed slices are committed.
7. Confirm unrelated dirty files remain unstaged and untouched.

Common commit groups:

- Code and tests.
- Data or lock artifacts.
- Dry-run transcript or generated non-spend audit artifact.
- Canonical ledger/status update.

## Final Response

Report succinctly:

- Slices completed and commits created, if any.
- Reviewers used, issues fixed, and whether any actionable issues remain.
- Checks run and results.
- Unrelated dirty files left untouched.
- Remaining blockers before expensive or claim-bearing execution.
