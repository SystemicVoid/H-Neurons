---
name: exec-review
description: Use when explicitly invoked as exec-review, /exec-review, or $exec-review to execute one bounded work slice from a filename, path, plan item, or direct task through implementation, verification, independent review/fix, and atomic commit.
---

# Exec Review

Execute exactly one bounded slice of work to reviewed atomic commit. The invocation argument is the scope boundary: a filename, directory, plan item, issue, or direct task. Do not expand beyond that boundary unless required to satisfy the slice safely.

## Intake

1. Treat the user's invocation argument as the slice scope. Resolve paths relative to the current workspace.
2. If the argument is missing or too broad to act on safely, ask one concise clarification before editing.
3. Inspect `git status` and relevant local instructions.
4. Identify unrelated dirty files and leave them unstaged, unmodified, and uncommitted.
5. Convert the scope into a short checklist: acceptance criteria, likely files, risks, checks, delegation plan, and commit boundary.

## Orchestration

The main thread is the orchestrator, not the scratchpad. It owns scope, sequencing, context budget, reviewer selection, final acceptance, staging, and commits.

- Keep noisy work off the main thread: exploration notes, stack traces, failed attempts, and raw logs stay inside worker/reviewer threads unless needed for a decision.
- Delegate when it reduces main-thread context or wall time. Prefer parallel agents for read-heavy exploration, test/log triage, summarization, and independent reviews.
- Be careful with parallel writers. Use one writer at a time unless file ownership is disjoint and explicit.
- Give each agent a narrow handoff: goal, constraints, owned files or read-only scope, commands to run, and concise return format.
- Require summaries, not transcripts: changed files, checks run, findings fixed, remaining risks, and user decisions needed.
- If main-thread context is getting crowded, finish, review, and commit the current slice before starting another.
- Do not let subagents spawn further agents unless explicitly assigned that role.

## Implementation Loop

1. Re-read implicated files, tests, scripts, docs, and existing patterns.
2. Implement locally when the edit is small, decision-heavy, tightly coupled, or on the critical path. Assign a worker when work is bounded, context-heavy or mechanical, and file ownership is clear. Workers must not stage or commit.
3. Run focused checks early, then broaden checks as risk increases. Fix failures and rerun relevant checks before review.
4. For data, output, spend-sensitive, GPU/API-capable, irreversible, or claim-bearing work, run the repo's guardrails and dry-runs before risky commands or staging.
5. Record provenance when relevant: source paths, hashes, IDs/counts/fingerprints, configs, commands, and output paths.

## Review+Fix Harness

After initial verification, choose as many reviewers as the slice warrants. Do not hardcode a fixed panel.

- Use one reviewer for simple contained docs, config, or small code changes.
- Use two reviewers for ordinary code, data, ops, or user-facing changes.
- Add more only for distinct risk lenses: correctness, tests/types/lint, data/schema/provenance, spend/ops safety, research claim calibration, UI/UX, security, migrations, concurrency, deployment.

Default reviewer mode is `review+fix`: reviewers fix clear in-scope issues they find, run targeted checks, and return only the patch summary and residual risks. Use `review-only` when a fix needs orchestrator/user judgment, may conflict with another writer, crosses the slice boundary, or touches risky operations.

For multiple reviewers:

- Run read-only reviews in parallel when lenses overlap heavily.
- Run `review+fix` reviewers sequentially, or in parallel only with disjoint file ownership.
- Prefer assigning reviewer-found fixes back to a reviewer instead of pulling debugging context into the main thread.

Reviewer instructions:

- Inspect the actual current tree, not just the orchestrator summary.
- Fix only clear in-scope issues when assigned `review+fix`.
- Preserve unrelated changes and slice boundaries.
- Run targeted checks for fixes.
- Report concise file/line findings, fixes made, commands run, and remaining risks.
- Say explicitly when no actionable issues remain.
- Do not stage or commit.
- Report ambiguous, architectural, broad, or out-of-scope issues instead of fixing them.

## Acceptance Gate

1. Treat blockers as mandatory fixes unless the user waives them.
2. Fix low/medium findings when small, aligned with scope, and useful.
3. Prefer assigning clear reviewer-found issues back as `review+fix` when it avoids pulling full debugging context into the main thread.
4. After fixes, rerun focused checks.
5. If blockers or nontrivial fixes occurred, run a fresh-pass review for residual risk and side effects, preferably by a fresh reviewer or a reviewer who did not make the fix.
6. The slice is done only when a fresh review pass surfaces no actionable issues, or remaining issues are explicitly documented and waived.

## Commit Rule

If files changed, create atomic conventional commits for completed slices unless the user explicitly says not to commit or blockers remain.

1. Inspect the final working tree.
2. Stage only the files for the completed slice.
3. Run relevant staged or already-equivalent checks.
4. Commit with a conventional subject and a WHY-focused body.
5. Confirm unrelated dirty files remain unstaged and untouched.
