# Pipeline & Workflow Notes

## Shared Utilities

- `scripts/utils.py` contains shared lightweight helpers (`normalize_answer`, `extract_mc_answer`). Import from there — never duplicate these functions into individual scripts.
- `scripts/collect_responses.py` imports `torch`/`transformers`/`openai` at module level — don't import it from lightweight scripts for utility functions.
- `scripts/evaluate_intervention.py` loads the OpenAI key via `python-dotenv` from the repo-root `.env`.

## Zero-Cost Runs

For runs without an OpenAI key: `--strategy synthetic-output` on `extract_answer_tokens.py` paired with `--locations output` on `extract_activations.py`.

## Template Guard

`extract_activations.py` needs the same `apply_chat_template()` tensor-vs-`BatchEncoding` guard as `collect_responses.py`.

## Profiling

Local Gemma profiling with `scalene` should start with `python -m scalene run --off` and only enable profiling after model load. Profiling the load phase directly can OOM or swap-thrash even when the unprofiled run fits.

## BioASQ OOD Probing

Use the official BioASQ Task B JSON (question `body` + `type` + `exact_answer`) rather than HF mirrors like `kroshan/BioASQ`, which flatten answer/context into CSV text and do not match the original task schema.

## Disjoint Classifier Evaluation

`data/gemma3_4b/pipeline/test_qids_disjoint.json` contains 782 sampled IDs, but the current disjoint classifier evaluation covers 780 because two IDs are missing activation files. Use the CI-bearing summary JSON as the reporting source of truth.

## Tailscale-First Cloud Access

`scripts/infra/lambda-bootstrap.sh` supports Tailscale-first access. Pass `TAILSCALE_AUTH_KEY` to auto-enroll/tag the instance; SSH is only locked to the Tailscale address after enrollment succeeds, so a missing/bad key will not cut off the public bootstrap session.

## SAE Investigation

Full plan, status tracker, and technical details (hook points, dimensions, release names): `docs/archive/sae_investigation_plan.md`.

SAE scripts: `extract_sae_activations.py`, `classifier_sae.py`, `intervene_sae.py`, `spike_sae_feasibility.py`.

SAE feature data goes to `data/gemma3_4b/pipeline/activations_sae/` (gitignored).
