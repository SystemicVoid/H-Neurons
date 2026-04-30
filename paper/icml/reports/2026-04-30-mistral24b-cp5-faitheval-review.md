# Mistral 24B CP5 FaithEval Review

> Verdict: CP5 completed, the H/control artifact contract passed, and parser
> failures were zero throughout. It does **not** pass the Mistral intervention
> replication gate. The apparent H-neuron slope is a slope-only signal; paired
> endpoint effects are null.

## Run Identity

| Field | Value |
| --- | --- |
| Wrapper command | `CP4_REVIEWED=1 STAGES=faitheval,faitheval_controls TMUX_WRAPPED=1 INHIBIT_WRAPPED=1 PROJECT_DIR=/workspace/02-h-neurons bash scripts/infra/mistral24b_replication.sh` |
| Pod | RunPod `yl8yx0vpo69eww`, H100 80GB HBM3, `US-CA-2`, retained volume `eitc8vwogm` |
| Wrapper log | `logs/mistral24b_replication_20260430T112112Z.log` |
| H artifacts | `data/mistral24b/intervention/faitheval/experiment/` |
| Control artifacts | `data/mistral24b/intervention/faitheval/control/` |
| Environment | `data/mistral24b/pipeline/environment_20260430T112112Z.txt` |
| Completion | `2026-04-30T12:18:52Z`; pod deleted after final sync |

## Contract Checks

Local post-sync checks passed:

```bash
uv run python -m scripts.lib.pipeline check-stage --output-dir data/mistral24b/intervention/faitheval/experiment --manifest data/manifests/faitheval_seed42_n200_mistral24b.lock.json --alphas 0.0 0.5 1.0 1.5 2.0 2.5 3.0
```

and the same `check-stage` command passed for all eight control directories:
five `seed_*_unconstrained` and three `seed_*_layer_matched`.

The H/control run contracts share:

- model key/path: `mistral_small_24b_instruct_2501` / `mistralai/Mistral-Small-24B-Instruct-2501`;
- benchmark and prompt style: `faitheval`, `standard`;
- sample manifest fingerprint: `a912a5bb29e4ab65`, path `data/manifests/faitheval_seed42_n200_mistral24b.lock.json`;
- classifier path/hash: `models/mistral24b_classifier_canonical.pkl`, `597435a84b19e68151f3f6903fd5ff1f12c61647440b07392dc34fdb8bea919d`;
- alpha grid: `0.0 0.5 1.0 1.5 2.0 2.5 3.0`;
- exact H-baseline binding: control config hashes match `intervention_run_config.json` and `results.20260430_112305.json`.

Parse failures are reported for every H alpha and every control seed/alpha; all
counts are `0/200`.

## Rates

| Series | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H-neurons | 53.0 | 49.5 | 52.0 | 52.0 | 54.0 | 54.0 | 53.0 |
| Unconstrained mean | 51.5 | 51.6 | 52.0 | 51.5 | 51.6 | 51.5 | 51.6 |
| Layer-matched mean | 51.7 | 51.5 | 52.0 | 51.8 | 51.7 | 51.7 | 51.5 |

`comparison_summary.json` reports H slope `+0.79 pp/alpha`, unconstrained
random mean slope `-0.02 pp/alpha`, and an unconstrained random empirical
95% slope interval `[-0.103, +0.063]`. The raw triage field therefore says
`specificity_supported`.

That triage is rejected for the CP5 claim gate because it is slope-only:

- H paired endpoint alpha `0.0 -> 3.0`: `0.0 pp`, 95% paired bootstrap CI
  `[-4.01, +4.00]`.
- H paired no-op-to-max alpha `1.0 -> 3.0`: `+1.0 pp`, 95% paired bootstrap CI
  `[-2.50, +4.50]`.
- H item flips alpha `0.0 -> 3.0`: `9` true-to-false, `9` false-to-true,
  `97` stayed true, `85` stayed false.
- H item flips alpha `1.0 -> 3.0`: `5` true-to-false, `7` false-to-true.
- H alpha `0.0` is already above the all-control mean by `2.875/200` items,
  and H alpha `3.0` is above the all-control mean by the same `2.875/200`
  items. The endpoint offset is not an alpha-induced effect.

## Interpretation

CP5 closes the prompt/parser/control readiness concern for full FaithEval, but
it does not show that the selected positive Mistral H-neurons reproduce the
Gemma-style intervention effect. The only positive-looking number is the
least interpretable one for this failure mode: a slope contrast against nearly
flat controls. The paired endpoint, no-op-to-max endpoint, item-flip balance,
and baseline offset audit all say not to claim an intervention replication.

Do not launch downstream Mistral SAE, bridge, or manuscript-upgrade work on the
assumption that CP5 passed. Any further Mistral FaithEval work should be framed
as a null classification or measurement-stability audit, not as the next step
after a successful H-neuron intervention gate.
