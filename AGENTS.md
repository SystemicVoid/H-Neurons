# Repository Guidelines

If anything about this repository is surprising, brittle, or easy to get wrong, add a short, specific note to this file when you discover it. Capture recurring quirks, unexpected workflows, hidden dependencies, stale references, or repeating errors in agent workflows so future sessions do not waste time rediscovering them.

## Project Structure & Module Organization
Core code lives in `scripts/`, with one CLI per pipeline stage: response collection, answer-token extraction, balanced ID sampling, activation extraction, classifier training, and intervention helpers. Put new research utilities in `scripts/` unless they justify a reusable package.

`data/` holds local datasets, sampled JSONL outputs, and review artifacts. Commit intermediate pipeline outputs (JSONL, JSON) to git — the original H-Neurons repo does this for all `data/examples/` files (up to ~800KB each) and raw TriviaQA parquets. Keep model weights and ephemeral logs out (covered by `.gitignore`). `docs/` contains research notes and deep-research writeups; the root `README.md` is the canonical pipeline overview.

`scripts/collect_responses.py` imports heavyweight runtime dependencies (`torch`, `transformers`, `openai`) at module import time, so lightweight analysis utilities should not import it just to reuse `normalize_answer`; copy the function verbatim instead.

`scripts/lambda-bootstrap.sh` and `scripts/lambda-AGENTS.md` are currently tuned for A100-40GB + Mistral-7B defaults; when running on GH200/H100 (ARM64) or switching to larger models (24B/27B), update model IDs, output filenames, and any architecture assumptions before launching long jobs.

`scripts/extract_activations.py` needs the same `apply_chat_template()` tensor-vs-`BatchEncoding` guard as `collect_responses.py`; without it, newer `transformers` can fail only when the long Step 4 job starts.

For zero-cost runs without an OpenAI key, use `scripts/extract_answer_tokens.py --strategy synthetic-output` and pair it with `scripts/extract_activations.py --locations output`; this preserves resume behavior without the manual JSONL hack from the README.

For repo hygiene, keep compact experiment artifacts such as benchmark CSVs, consistency-sample JSONL files, answer-token JSONL files, balanced ID JSONs, and compact intervention outputs visible in git. Keep heavy activation dumps, scratch investigation folders, and local sync state ignored. If a local GH200 sync is partial, treat the remote copy as canonical until file counts and presence of the classifier match.

The fetched remote `mistral24b_classifier.pkl` currently emits scikit-learn's `InconsistentVersionWarning` when loaded locally (`0.23.2` pickle vs newer local sklearn). It still loads, but treat cross-machine probe pickles as version-sensitive artifacts and prefer recording metrics JSON alongside them.

The GH200 remote run now has a local cron-based wakeup path via `scripts/cron_gh200_wakeup.sh`; it watches the remote `logs/gh200_pipeline.{status,done,failed}` files, pings the `agents` tmux session on state changes, and can launch a one-shot `codex exec` takeover tmux session when the pipeline finishes or fails.

On a single GH200 96GB GPU, `Mistral-Small-24B-Instruct-2501` should be launched with an explicit GPU placement like `--device_map cuda:0`; `device_map=auto` can spill part of the model to CPU/Grace memory and run far slower. By contrast, `Llama-3.3-70B-Instruct` in bf16 does not fit fully on that GPU, so a pure-GPU run is not possible without changing the model format (for example quantization) or hardware setup.

The GH200 tmux orchestrator must not assume a running Step 1 pane shows `uv` as `pane_current_command`. After switching to the CUDA-capable `.venv-gpu` path, healthy runs show `python`, so `scripts/gh200_sequential_pipeline.sh` should treat `uv`, `python`, and `python3` as active runners or it will restart a good job by mistake.

On `transformers` 5.3.0, loading `Mistral-Small-24B-Instruct-2501` emits a tokenizer warning about an incorrect regex pattern and recommends `fix_mistral_regex=True`. That warning did not block execution, but it is a real reproducibility risk to track before comparing token-level outputs or answer-token matches across reruns.

For paper-faithful H-Neuron replication, note that the local `scripts/classifier.py` can sweep `C` using held-out probe metrics (`accuracy`/`f1`/`auroc` etc.), but it does **not** implement the paper's full selection rule that also scores TriviaQA behavior after suppressing the selected neurons. Treat the current sweep as a good detector-selection baseline, not the final paper-equivalent intervention-selection criterion.

For FaithEval intervention runs, `scripts/run_intervention.py` defaults to `--prompt_style anti_compliance`, but its own prompt builder notes that `--prompt_style standard` matches the official Salesforce framing and presumed paper usage. Use `standard` when the goal is paper-faithful replication rather than stress-testing resistance to misleading context.

`scripts/extract_answer_tokens.py` is brittle to malformed JSON from the judge model on short quoted answers. In a 100-sample Mistral canary, one row (`tc_115`, Walter Cronkite's "And that's the way it is") failed repeatedly for that reason even though the other 99 extracted spans all matched `extract_activations.py`'s current tokenizer/span logic.

For this project, the operational policy is now stricter than "it runs somehow": on rented hardware, only run models whose full-precision weights and activation workflow fit entirely on the GPU being rented. A single GH200 96GB proved viable for the 24B class, but `Llama-3.3-70B-Instruct` violated that rule and was intentionally stopped. Treat roughly the 24B class as the safe upper bound on this box unless a larger bf16 model is explicitly demonstrated to stay GPU-resident throughout the relevant stages.

Pricing check on 2026-03-15: Runpod is cheaper for smaller 80GB-class iteration (for example A100 80GB / H100 80GB), but Lambda's GH200 96GB was listed at `$1.99/hr`, which is cheaper than Runpod's nearest 94GB-class single-GPU option (`H100 NVL` at `$2.59/hr`). For this repo's 24B activation-faithful runs, that means "Runpod is cheaper" is only true for smaller jobs; for the validated ~96GB single-GPU path, Lambda can actually be the lower-cost option.

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

## Quantitative Reporting Standards
Every quantitative claim in presentation materials (site, slides, reports) must include uncertainty estimates. Use bootstrap 95% confidence intervals where sample sizes allow (n > 30). For classifier metrics, report ± from stratified bootstrap over test samples. For intervention compliance rates, report ± from binomial proportion CIs. If uncertainty cannot yet be computed, flag the number explicitly as "no CI" so reviewers know it is a gap, not an oversight.

## Testing Guidelines
There is no formal `tests/` suite yet. Use judgment and follow best practices.

## Long-Running Jobs & System Suspend
On Pop!_OS / COSMIC DE, system auto-suspend will kill tmux jobs mid-run. Always hold a `systemd-inhibit` lock for jobs longer than ~20 minutes.

**At launch** — wrap the command:
```bash
systemd-inhibit --what=sleep:idle --who="<job-name>" --why="<description>" \
    uv run python scripts/run_intervention.py ...
```

**For already-running jobs** — spin up a watcher in a dedicated tmux window named `inhibit`:
```bash
systemd-inhibit --what=sleep:idle --who="<job-name>" --why="<description>" \
    bash -c 'while tmux list-windows | grep -q "<job-window-name>"; do sleep 30; done'
```
This auto-releases the lock when the job window closes — no manual cleanup needed. Check active locks with `systemd-inhibit --list`.

**Post-suspend recovery** — if a job survived suspend but is running slowly (CPU fallback, low GPU utilization/wattage), Ctrl+C may not land because the terminal is in raw mode. Use `kill -9` directly on the python PID: `pgrep -a python | grep <script>` to find it, then `kill -9 <pid>`. The script's resume logic will pick up from the last written line on restart.

## Commit & Pull Request Guidelines
Recent history uses Conventional Commit-style subjects.
