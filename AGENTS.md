# Repository Guidelines

If anything about this repository is surprising, brittle, or easy to get wrong, add a short, specific note to this file when you discover it. Capture recurring quirks, unexpected workflows, hidden dependencies, stale references, or repeating errors in agent workflows so future sessions do not waste time rediscovering them.

## Project Structure & Module Organization
Core code lives in `scripts/`, with one CLI per pipeline stage: response collection, answer-token extraction, balanced ID sampling, activation extraction, classifier training, and intervention helpers. Put new research utilities in `scripts/` unless they justify a reusable package.

`data/` holds local datasets, sampled JSONL outputs, and review artifacts. Commit intermediate pipeline outputs (JSONL, JSON) to git — the original H-Neurons repo does this for all `data/examples/` files (up to ~800KB each) and raw TriviaQA parquets. Keep model weights and ephemeral logs out (covered by `.gitignore`). `docs/` contains research notes and deep-research writeups; the root `README.md` is the canonical pipeline overview.

`scripts/collect_responses.py` imports heavyweight runtime dependencies (`torch`, `transformers`, `openai`) at module import time, so lightweight analysis utilities should not import it just to reuse `normalize_answer`; copy the function verbatim instead.

`scripts/lambda-bootstrap.sh` and `scripts/lambda-AGENTS.md` are currently tuned for A100-40GB + Mistral-7B defaults; when running on GH200/H100 (ARM64) or switching to larger models (24B/27B), update model IDs, output filenames, and any architecture assumptions before launching long jobs.

`scripts/extract_activations.py` needs the same `apply_chat_template()` tensor-vs-`BatchEncoding` guard as `collect_responses.py`; without it, newer `transformers` can fail only when the long Step 4 job starts.

For zero-cost runs without an OpenAI key, use `scripts/extract_answer_tokens.py --strategy synthetic-output` and pair it with `scripts/extract_activations.py --locations output`; this preserves resume behavior without the manual JSONL hack from the README.

Shared agent skills should live in `~/.config/forge/agents/.agents/skills-store/` and be symlinked into repo-local `.agents/skills/` entries instead of copied into the repo. This keeps multi-repo skill updates centralized.

## Build, Test, and Development Commands
Use `uv` for Python environment management.

- `uv sync` installs the project from `pyproject.toml`.
- `uv add <package>` adds a dependency; keep `requirements.txt` aligned in the same change because the repo still documents runtime deps there.
- `uv run python scripts/collect_responses.py --help` checks a script entrypoint and available flags.
- `uv run python scripts/classifier.py --model_path /path/to/model ...` runs the training stage.
- `uv run ruff check scripts` and `uv run ruff format scripts` lint and format Python before review.
- `uv run ty check` performs a lightweight type pass if you introduce nontrivial new logic.

## Coding Style & Naming Conventions
Follow existing Python conventions and modern best practices.

## Testing Guidelines
There is no formal `tests/` suite yet. Use judgment and follow best practices.

## Commit & Pull Request Guidelines
Recent history uses Conventional Commit-style subjects.
