# Repository Guidelines

## Always-On Defaults

- Use the literature when building experiments based on related work; start with `papers/INDEX.md`.
- **Gate on diagnosticity, not cost — a non-diagnostic null is not a falsification.**
  Two habits, each sensible alone, compound into our costliest recurring error across
  projects: staging work behind the *cheapest* test, then reading its null as "claim
  false." The cheapest surface (smallest model, fewest samples, most saturated readout)
  is exactly the one most likely to return a non-diagnostic null — and staging then
  silently discards everything behind it. For any spec that gates one experiment behind
  another:
  - **Cheap licenses de-risking, not deciding.** A cheap test may inform, prioritize,
    and refine freely. It may kill or defer a claim only when it is *diagnostic* for it:
    sited where the effect must appear per the effect's own theory (surface, model
    capability, N, unsaturated readout), and shown able to detect an effect of the
    claim's expected size — via a positive control / manipulation check on that harness,
    or a prior artifact establishing its sensitivity. "The harness runs" is not
    sensitivity, and spend buys no license either: an expensive non-diagnostic null is
    equally void.
  - **Name the discard.** Every kill/defer-on-null states in one sentence which
    experiment it forecloses and why a null there would be diagnostic rather than a
    power/validity failure. If that sentence can't be written, the boundary is a
    checkpoint, not a gate — stage on cost freely, but kill-authority must be earned.
    The pre-run review gate checks this clause.
  - **Cap non-diagnostic verdicts in both directions.** A null from a non-diagnostic
    test records as harness_inadequate and the claim stays open: either strengthen the
    test, or park the claim as an explicit resourcing decision — parking on
    cost/value-of-information grounds is always legitimate; relabeling it "falsified"
    never is. A positive from the same surface caps at pilot-suggestive. Either way the
    result's first-class product is design information (siting, variance, catch rates,
    powered N) for the test that will settle the claim.
  - **Don't overcorrect.** When a cheap test is diagnostic, use it and let it kill —
    the license is diagnosticity, not expense. Size effort to the claim's value, not to
    a reflex in either direction.
- Follow existing conventional commits pattern.
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
