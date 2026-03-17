# Roadmap

Infrastructure improvements deferred for future consideration.

## Structured Logging

Replace ad-hoc `print()` calls across pipeline scripts with Python `logging` at
INFO/DEBUG levels. Long-running intervention and activation-extraction jobs
benefit from timestamped, level-filtered output -- especially with the
suspend/resume workflow on Pop!_OS where post-wake debugging currently relies on
scrollback grep.

## AGENTS.md Freshness Validation

`AGENTS.md` and `CLAUDE.md` are currently identical copies. Symlink one to the
other (or consolidate into a single canonical file) to eliminate drift risk when
guidelines are updated.
