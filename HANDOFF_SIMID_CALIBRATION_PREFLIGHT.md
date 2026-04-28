# SIMID Calibration Preflight Handoff

Commit: `b03e309 fix(simid): harden calibration launch preflight`

What changed:
- Added strict pre-launch validation for SIMID open calibration queues before secondary Batch spend.
- Rejected malformed primary grades at launch instead of letting them fail later in disagreement/adjudication/finalization paths.
- Added per-row queue fingerprints to secondary labels and fail-closed reuse checks for missing or stale fingerprints.
- Added `queue_rows_sha256` to labeler provenance.
- Added a real Batch canary script with a 2-row validator-compatible synthetic queue, undersized-only bypass, disagreement/adjudication assertions, and preserved temp state on failure.

Verification already run:
- `uv run pytest tests/` -> `793 passed`
- `ruff check scripts tests`
- `ruff format --check scripts tests`
- `ty check`
- `shellcheck scripts/infra/simid_canary_calibration.sh scripts/infra/check_judge_models.sh`
- Commit hook also passed large-file check, active-run Git guard, ruff, ty, and CI coverage audit.

Not done:
- Did not run `scripts/infra/simid_canary_calibration.sh` because it submits real OpenAI Batch jobs and spends API budget.
- Left pre-existing unrelated notes/data changes and the OpenAI batch alias edits unstaged.
