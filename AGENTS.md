# Repository Guidelines

## Always-On Defaults

- Use the literature when building experiments based on related work; start with `papers/INDEX.md`.
- When the user asks for a handoff prompt for another agent, output only the prompt body. Do not prepend framing or wrap it in explanatory prose.
- Follow existing conventional commits pattern.
- For independent code review, use the repo-local `coderabbit-review` skill and `python .agents/skills/coderabbit-review/scripts/coderabbit_review_watch.py -- <coderabbit review args>`; let it wait up to 30 minutes and use its Codex fallback only on CodeRabbit timeout, rate limit, or error.
- For run-output commit safety, check `.pre-commit-config.yaml`'s `active-run-git-guard`; run `uv run python -m scripts.lib.pipeline active-run-status` before staging data or output paths.

## Scoped Guidance

Read the narrower instruction file before working in these areas:

- `scripts/AGENTS.md` for Python scripts, evaluation helpers, and experiment entrypoints.
- `scripts/infra/AGENTS.md` for long GPU jobs, pipeline wrappers, tmux/systemd-inhibit orchestration, and remote-run scripts.
- `scripts/lib/AGENTS.md` for pipeline guard-library changes.
- `data/AGENTS.md` for committed run outputs, provenance sidecars, and run-directory layout.
- `site/AGENTS.md` for the static site, site data exports, and deployment.
- `notes/AGENTS.md` for notes and research-log hygiene. 

## Build, Test, and Development Commands

- Use `uv add <package>` for Python dependencies; regenerate `requirements.txt` with `uv export --no-hashes --frozen --no-emit-project > requirements.txt` in the same change.
- `ruff check scripts` and `ruff format scripts` lint and format Python before review.
- `ty check` type-checks `scripts/` and must pass with zero diagnostics before committing.
- Tests live in `tests/` and run via `uv run pytest`.
- Core evaluation helpers (`normalize_answer`, `extract_mc_answer`) have unit tests in `tests/test_utils.py`.
- `ruff`, `ty`, and `prek` are global tools on PATH.

## Quantitative Reporting Standards

For claim-bearing quantitative reporting, read `docs/quantitative-reporting-standards.md`.
