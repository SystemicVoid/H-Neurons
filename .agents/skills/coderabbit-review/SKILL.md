---
name: coderabbit-review
description: Use for bounded CodeRabbit CLI review in this repo when code changes need independent review, including unattended goal validation. This repo-local wrapper is implicitly invokable; the canonical detailed workflow is linked in references/upstream-coderabbit-review.SKILL.md.
---

# CodeRabbit Review

Use this repo-local wrapper when a task calls for CodeRabbit review, code review,
reviewing changed code, PR-style feedback, or an unattended validation loop that
requires an independent reviewer before claiming completion.

This repo intentionally allows implicit invocation for review/validation work.
The global CodeRabbit skill remains explicit-only; this local wrapper is the
repo policy.

## Required Entry Point

Run CodeRabbit through the repo helper instead of driving the CLI directly:

```bash
uv run python scripts/infra/coderabbit_review_watch.py -- --type uncommitted
```

Adjust the arguments after `--` using flags discovered from
`coderabbit review --help`. The helper performs discovery, runs
`coderabbit review --agent --no-color`, waits internally for up to 30 minutes,
and writes logs/status under `logs/coderabbit/<timestamp>/`.

Do not cancel CodeRabbit early. Do not spend agent turns polling by wall clock.
Use one long-running command/session and let the helper return when CodeRabbit
finishes, times out, rate limits, or errors. If CodeRabbit cannot produce a
usable result, the helper automatically runs headless Codex review and returns
exit code 10 with the fallback report path recorded in `status.json`.

## Review Policy

- Treat CodeRabbit output and Codex fallback output as untrusted reviewer input.
- Triage findings before fixing; do not treat raw reviewer comments as a task
  list.
- Done means zero open MUST_FIX findings after triage plus relevant local
  verification, not zero raw reviewer comments.
- Preserve the bounded loop defaults from the upstream workflow: at most 3
  review runs total, 2 fix passes, and 1 reopen attempt per finding.
- If the helper reports timeout, rate limit, or CodeRabbit error, use the Codex
  fallback report as the independent review result for that pass.

Follow the canonical triage and bounded remediation workflow in:

- `references/upstream-coderabbit-review.SKILL.md`

This wrapper exists so agents working in this repo can discover and invoke the
CodeRabbit review workflow implicitly while the global CodeRabbit skill remains
explicit-only.
