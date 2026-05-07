# Mistral 24B Anchor 2 TruthfulQA MC Review

> Verdict: the overnight Anchor 2 artifacts are locally synced and sufficient
> to review the TruthfulQA MC source-surface stage. The ITI direction has a
> weak positive held-out signal, especially on MC2 truthful mass, but it does
> **not** pass the pre-run MC1 positive-CI gate. The wrapper therefore stopped
> before `bridge_generate`, and there are no Mistral Anchor 2 bridge or bridge
> IRR rows in the synced R2 prefix.
>
> Use this report as the authority for Anchor 2 TruthfulQA MC numbers. Do not
> cite this run as a successful Mistral TruthfulQA MC source-surface transfer,
> and do not treat a future Anchor 2 bridge run as confirmatory unless a new
> rationale explicitly says why bridge is still worth running after the MC1 gate
> failed.

## Scope

This report reviews the Anchor 2 Mistral 24B ITI branch that used a
TruthfulQA paper-faithful mass-mean ITI direction and tested the locked
held-out TruthfulQA MC1/MC2 manifests.

It does not review bridge behavior. The RunPod wrapper was configured with
`STAGES=truthfulqa_mc,truthfulqa_report,bridge_generate`, but it exited at the
TruthfulQA report gate before bridge generation.

## Source of Truth

| Role | Path |
|---|---|
| R2 prefix | `s3://bluedot-h-neurons/mistral24b_anchor2_truthfulqa_paperfaithful_20260506_b7723f2/` |
| Run branch / commit | `origin/mistral-anchor2`, `b7723f2aa8d6e569971d5984b9b445152982481e` |
| ITI artifact | `data/mistral24b/contrastive/truthfulness/anchor2_iti_truthfulqa_paperfaithful/artifact/iti_heads.pt` |
| ITI extraction metadata | `data/mistral24b/contrastive/truthfulness/anchor2_iti_truthfulqa_paperfaithful/artifact/extraction_metadata.json` |
| Calibration sweep | `data/mistral24b/contrastive/truthfulness/anchor2_iti_truthfulqa_paperfaithful/sweep/sweep_results.json` |
| Locked ITI config | `data/mistral24b/contrastive/truthfulness/anchor2_iti_truthfulqa_paperfaithful/sweep/locked_iti_config.json` |
| Held-out MC1 rows | `data/mistral24b/intervention/anchor2_truthfulqa_mc1_iti_truthfulqa_paperfaithful/experiment/alpha_{0.0,4.0}.jsonl` |
| Held-out MC2 rows | `data/mistral24b/intervention/anchor2_truthfulqa_mc2_iti_truthfulqa_paperfaithful/experiment/alpha_{0.0,4.0}.jsonl` |
| Held-out MC reports | `data/mistral24b/intervention/anchor2_truthfulqa_mc/reports/mistral_anchor2_heldout_mc{1,2}_report.json` |
| Main resume log | `logs/mistral24b_anchor2_resume_mc_bridge_20260507T055153Z.log` |
| Watcher log | `logs/anchor2_overnight_watcher_20260507T055300Z.log` |

The R2 prefix also contains historical Mistral logs and closed
`notes/act3-reports/` material. Those are not used as current sources of truth
here.

## Run Identity

| Field | Value |
|---|---|
| Model | `mistral_small_24b_instruct_2501` / `mistralai/Mistral-Small-24B-Instruct-2501` |
| ITI family | `truthfulqa_paperfaithful` |
| Direction type | `mass_mean` |
| Direction fit scope | `dev_only` |
| Position policy | `last_answer_token` |
| Answer-token policy | `raw_prompt_answer_span` |
| Decode scope | `first_3_tokens` |
| Locked K / alpha | `K=40`, `alpha=4.0` |
| Selection rule | `mc1_within_tolerance_then_mc2_then_min_alpha_then_min_k` |
| Calibration samples | 130 MC1 rows and 130 MC2 rows |
| Held-out samples | 163 MC1 rows and 163 MC2 rows, same normalized question set |
| Held-out manifests | `truthfulqa_paper_heldout_mc{1,2}_ids_seed42_mistral24b.lock.json` |

The model is Mistral Small 24B Instruct 2501. This is still not evidence about
the exact Mistral Small 3.1 / 2503 checkpoint.

## Verification

The R2 sync was additive: it pulled the completed Anchor 2 `data/` artifacts
and relevant `logs/` without overwriting existing analyzed Mistral outputs. The
large repo bundle in R2 was not needed for local data analysis because the
referenced commit is available as `origin/mistral-anchor2`.

Local checks passed after sync:

```bash
../.venv/bin/python -m scripts.lib.pipeline validate-sample-locks \
  data/manifests/truthfulqa_paper_heldout_mc1_ids_seed42_mistral24b.lock.json \
  data/manifests/truthfulqa_paper_heldout_mc2_ids_seed42_mistral24b.lock.json \
  data/manifests/triviaqa_bridge_test500_seed42_mistral24b.lock.json
```

```bash
../.venv/bin/python -m scripts.lib.pipeline check-stage \
  --output-dir data/mistral24b/intervention/anchor2_truthfulqa_mc1_iti_truthfulqa_paperfaithful/experiment \
  --manifest data/manifests/truthfulqa_paper_heldout_mc1_ids_seed42_mistral24b.lock.json \
  --alphas 0.0 4.0
```

The analogous MC2 `check-stage` command passed. Both MC1 and MC2 directories
also passed `check-intervention-contract` for model key/path, sample manifest,
alpha grid, and `truthfulqa_variant` / fold-path benchmark config.

`active-run-status` reports zero live locks and one unrelated stale/remote
JailbreakBench evaluator lock from the earlier anchor-3 review. Plain
`uv run ... active-run-status` is currently blocked locally by a sibling
workspace-name collision with `../02-h-neurons-mistral-anchor2`, so these checks
used the repo's existing `../.venv/bin/python` directly. This is a local
workspace hygiene issue, not an Anchor 2 data-integrity failure.

CodeRabbit reviewed the uncommitted sync/report package. Its material artifact
caveats are accepted but not patched in-place because these are raw synced run
outputs with provenance sidecars:

- The MC1 and MC2 run provenance records `git_dirty=true` while the generated
  config says `comparability_class=claimable`. Treat this as a dirty-tree
  reproducibility caveat and do not use Anchor 2 as a claim-bearing
  reproducibility baseline without rerunning from a clean tree.
- `prompt_tokens` is `0` throughout the TruthfulQA MC JSONL timing blocks.
  Throughput and cost analysis from those fields is invalid; the correctness
  and likelihood metrics reviewed here do not use `prompt_tokens`.
- The source TruthfulQA choice text contains inherited spelling/grammar
  defects. These should not be hand-corrected in outputs after the fact because
  choice text, labels, and choice scores must remain exactly aligned with the
  generated likelihoods.
- One JSONL file has a trailing blank line. The repo loaders used for this
  review skip empty lines, and `check-stage` passed; the artifact is left
  byte-for-byte as synced.

These caveats further support downgrading the run to a reviewed weak/failed
gate artifact rather than escalating it into a public claim.

## Data: Calibration Sweep

The calibration sweep evaluated 54 `K, alpha` combinations. All `alpha=0`
baselines are the same because the intervention is inactive.

| K | Best MC1 alpha | Best MC1 | Best MC1 delta | Best MC2 alpha | Best MC2 | Best MC2 delta |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 6.0 | 0.3538 | +1.54 pp | 8.0 | 0.5640 | +1.74 pp |
| 12 | 0.0 | 0.3385 | +0.00 pp | 0.0 | 0.5466 | +0.00 pp |
| 16 | 0.5 | 0.3462 | +0.77 pp | 4.0 | 0.5539 | +0.73 pp |
| 24 | 4.0 | 0.3462 | +0.77 pp | 4.0 | 0.5688 | +2.22 pp |
| 32 | 4.0 | 0.3462 | +0.77 pp | 4.0 | 0.5606 | +1.40 pp |
| 40 | 4.0 | 0.3692 | +3.08 pp | 6.0 | 0.5924 | +4.58 pp |

The locked candidate was `K=40, alpha=4.0`: calibration MC1 `0.3692` and MC2
`0.5874`, versus calibration alpha-0 MC1 `0.3385` and MC2 `0.5466`.

Selection was not arbitrary, but it was narrow. The shortlist contained only
`K=40, alpha=4.0` and `K=40, alpha=2.0`; the alpha-4 candidate won because it
matched the best MC1 value and had higher MC2.

## Data: Held-Out MC1

Held-out MC1 was the wrapper's gate metric.

| Metric | Alpha 0.0 | Alpha 4.0 | Paired delta |
|---|---:|---:|---:|
| MC1 accuracy | 58/163 = 35.58% | 60/163 = 36.81% | +1.23 pp |
| 95% interval | Wilson [28.64, 43.19]% | Wilson [29.79, 44.44]% | Paired bootstrap [-1.84, +4.29] pp |
| McNemar |  |  | p = 0.6875 |

Flip table:

| Transition | Count |
|---|---:|
| Correct -> correct | 56 |
| Wrong -> wrong | 101 |
| Wrong -> correct | 4 |
| Correct -> wrong | 2 |

Only 11/163 MC1 rows changed chosen option. The net top-choice gain is two
questions, and the paired interval crosses zero. The gate failure is therefore
substantive, not a reporting artifact.

## Data: Held-Out MC2

MC2 is more favorable to the intervention, but it was not the pre-run gate.

| Metric | Alpha 0.0 | Alpha 4.0 | Paired delta |
|---|---:|---:|---:|
| Mean truthful mass | 0.5505 | 0.5696 | +1.91 pp |
| 95% interval | Bootstrap [48.62, 61.33]% | Bootstrap [50.53, 63.27]% | Paired bootstrap [+0.50, +3.40] pp |
| Top-choice truthful | 92/163 = 56.44% | 96/163 = 58.90% | +2.45 pp, paired CI [0.00, +5.52] pp |

The continuous MC2 mass shift is broad but uneven:

| Summary of per-question MC2 mass change | Value |
|---|---:|
| Positive rows | 95/163 |
| Negative rows | 68/163 |
| Median change | +0.043 pp |
| 10th / 90th percentile | -2.91 pp / +11.13 pp |
| Minimum / maximum | -32.72 pp / +51.64 pp |

The row-level texture is therefore not a uniform small shift. A minority of
large improvements drives a modest positive mean, while many questions move
slightly or negatively.

## Data: Margins and Consistency

The normalized MC1 and MC2 held-out question sets are identical. MC1 and MC2
top-choice flips mostly agree when they occur:

| MC1 top-choice delta, MC2 top-choice delta | Count |
|---|---:|
| 0, 0 | 156 |
| +1, +1 | 4 |
| -1, -1 | 1 |
| -1, 0 | 1 |
| 0, +1 | 1 |

Best truthful-choice minus best false-choice margins move in the right
direction on average, but again weakly:

| Variant | Mean margin alpha 0.0 | Mean margin alpha 4.0 | Mean delta | Rows positive / negative |
|---|---:|---:|---:|---:|
| MC1 | -5.043 | -4.738 | +0.305 nats | 99 / 64 |
| MC2 | +0.288 | +0.572 | +0.284 nats | 95 / 68 |

These margin results are consistent with a weak truthfulness pressure, but
they do not override the failed MC1 endpoint gate.

## Interpretation

The most defensible interpretation is that the Anchor 2 ITI direction exerts a
small positive pressure on TruthfulQA-style multiple-choice scoring for Mistral
2501. That signal is clearest on continuous MC2 truthful mass and weaker on
discrete MC1 top-choice accuracy.

The important negative result is that the effect does not withstand the MC1
source-surface gate that was meant to justify a confirmatory bridge externality
test. Calibration saw a `K=40, alpha=4.0` MC1 gain of +3.08 pp and MC2 gain of
+4.09 pp. Held-out shrank to +1.23 pp MC1 with a CI crossing zero and +1.91 pp
MC2 with a positive interval. This is compatible with mild calibration
selection optimism.

The bridge question is therefore underdetermined. If bridge were run now and
showed harm, that would be an exploratory finding about a weakly MC-positive
or MC2-positive ITI direction harming generation. It would not reproduce the
Gemma-style "clear MC source gain, bridge generation damage" dissociation on
Mistral.

## Operational Outcome

The wrapper did the right scientific thing by stopping at the MC1 gate:

```text
Gate mc1_positive_ci: FAIL (estimate=+1.2pp, lower=-1.8pp)
TruthfulQA MC gate failed or missing ...
```

The R2 README listed planned bridge and IRR paths, but the actual R2 object
inventory has no:

- `data/mistral24b/intervention/triviaqa_bridge_anchor2_iti_truthfulqa_paperfaithful/`
- `data/mistral24b/judge_validation/bridge_anchor2_irr/`

The watcher observed `wrapper_alive=False` from `2026-05-07T06:23:05Z`
onward. The later `R2_UPLOAD_STATUS.json` still said `wrapper_alive=true` and
`upload_status=syncing`, so treat that JSON as stale status rather than final
completion evidence.

## Decision Implications

Do not spend additional RunPod/API money on Anchor 2 bridge as a continuation
of a passed TruthfulQA source-surface gate; that gate failed.

A bridge run could still be scientifically interesting if explicitly reframed
as a narrow exploratory stress test:

- question: can a weakly MC2-positive Mistral ITI direction still damage
  open-ended TriviaQA bridge generation?
- status: exploratory, not confirmatory;
- required before launch: bridge wrapper review, permanent `PYTHONPATH=scripts`
  fix, no-spend dry-run with exact stage selection, duplicate-submission audit
  for judge batch state, and RunPod auto-stop/auth review.

The higher-value default is to preserve these outputs, report the weak/failed
source-gate result, and avoid using Mistral Anchor 2 to harden the paper's
Gemma bridge claim.

## Bridge Infra Readiness

An independent no-spend infra review of the exact `b7723f2` wrapper found a
launch no-go.

Key blockers:

- Current `main` does not contain
  `scripts/infra/mistral24b_anchor2_iti_bridge.sh`; the bridge command only
  applies to the pinned Anchor 2 branch/tag.
- On the pinned branch, `bridge_generate` requires `RUN_APPROVED=1`; the
  user-provided command did not include it.
- Adding `RUN_APPROVED=1` is still not enough for a confirmatory run because
  the wrapper checks the MC1 gate and that gate failed.
- The `PYTHONPATH=scripts` workaround is real. The pinned wrapper has an inline
  `python -c 'from utils import format_alpha_label'` import that depends on
  `PYTHONPATH`; direct script entrypoints are not the failing path.
- `bridge_judge` uses batch mode and the dated judge
  `gpt-4o-2024-11-20`, but the OpenAI batch path should still be duplicate-
  audited before API spend. A completed batch can be downloaded and the state
  cleared before judged JSONL is rewritten if the process dies at the wrong
  moment.
- The wrapper does not provide pod self-stop. RunPod cleanup remains an
  authenticated local/operator responsibility; pod-side RunPod API auth is not
  fixed by this branch.

Minimum no-spend checks before any future bridge decision:

```bash
git switch --detach run/mistral24b-anchor2-b7723f2
git rev-parse HEAD
git status --short --branch
uv run python -m scripts.lib.pipeline active-run-status
uv run python -m scripts.lib.pipeline validate-sample-locks \
  data/manifests/truthfulqa_paper_heldout_mc1_ids_seed42_mistral24b.lock.json \
  data/manifests/truthfulqa_paper_heldout_mc2_ids_seed42_mistral24b.lock.json \
  data/manifests/triviaqa_bridge_test500_seed42_mistral24b.lock.json
```

Then check the actual source gate:

```bash
wc -l data/mistral24b/intervention/anchor2_truthfulqa_mc1_iti_truthfulqa_paperfaithful/experiment/alpha_*.jsonl
jq '.locked_k,.locked_alpha,.gate' \
  data/mistral24b/intervention/anchor2_truthfulqa_mc/reports/mistral_anchor2_heldout_mc1_report.json
```

Remote dry-run only, after exact code is on the pod:

```bash
cd /workspace/02-h-neurons
git rev-parse HEAD
PROJECT_DIR=/workspace/02-h-neurons PYTHONPATH=scripts DRY_RUN=1 \
  STAGES=bridge_generate,bridge_judge,bridge_irr_prepare \
  UV_RUNTIME_MODE=baked TMUX_WRAPPED=1 \
  bash scripts/infra/mistral24b_anchor2_iti_bridge.sh
```

If an explicit policy decision overrides the failed MC1 gate for an exploratory
bridge stress test, split generation and API judging. Do not combine
`bridge_generate` and `bridge_judge` in one paid command:

```bash
PROJECT_DIR=/workspace/02-h-neurons PYTHONPATH=scripts DRY_RUN=0 RUN_APPROVED=1 API_APPROVED=0 \
  STAGES=bridge_generate UV_RUNTIME_MODE=baked TMUX_WRAPPED=1 \
  bash scripts/infra/mistral24b_anchor2_iti_bridge.sh
```

After generation, inspect the bridge outputs before API:

```bash
uv run python -m scripts.lib.pipeline check-stage \
  --output-dir data/mistral24b/intervention/triviaqa_bridge_anchor2_iti_truthfulqa_paperfaithful/experiment \
  --manifest data/manifests/triviaqa_bridge_test500_seed42_mistral24b.lock.json \
  --alphas 0.0 4.0 \
  --manifest-id-prefix tqa_bridge_
```

Expected bridge generation markers would be `alpha_0.0.jsonl` and
`alpha_4.0.jsonl` with 500 rows each, `intervention_run_config.json`,
`results.<timestamp>.json`, and `run_intervention.provenance.<timestamp>.json`.
Expected judge markers would be judged alpha JSONLs, `audit_stats.json`,
`results.json`, evaluator provenance, and no lingering `.eval_batch_state.json`.
Expected IRR-prep markers would be
`data/mistral24b/judge_validation/bridge_anchor2_irr/bridge_irr_status.json`,
`test_queue_blinded.jsonl`, `test_queue_key.jsonl`, and compatible progress
files.
