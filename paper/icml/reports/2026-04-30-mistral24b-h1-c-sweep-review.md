# Mistral 24B H1 Intervention-Aware C-Sweep Review

> Verdict: the H1 follow-up tested the strongest cheap explanation for the
> CP5 null. The intervention-aware selector did choose a different C
> (`0.75`, 9 positive H-neurons) than the original AUROC-selected CP3/CP5
> classifier (`1.0`, 10 positive H-neurons), but the locked n=200 FaithEval
> follow-up stayed flat. This weakens H1 as a sufficient explanation for the
> Mistral 2501 FaithEval null. It does not rescue a Mistral H-neuron
> intervention-replication claim.

Companion context:

- CP5 numerical authority:
  [2026-04-30 Mistral 24B CP5 FaithEval Review](2026-04-30-mistral24b-cp5-faitheval-review.md).
- CP5 adversarial pipeline audit:
  [2026-04-30 Mistral 24B CP5 Pipeline Audit](../reviews/2026-04-30-mistral24b-cp5-pipeline-audit.md).
- H1 hypothesis memo, now superseded for the H1 run decision:
  [Why CP5 went null on Mistral 2501](../reviews/2026-04-30-mistral24b-cp5-null-causes-and-2501-vs-2503.md).
- Mistral execution ledger:
  [Mistral 24B L1 Mitigation Strategy](../../../notes/icml/mistral24b/2026-04-28-5.5-pro-l1-mitigation-strategy.md).

## Scope and Claim Status

This report is a post-CP5 null-classification analysis. It asks whether the
main code-level protocol divergence identified after CP5, namely
intervention-aware C selection, changes the Mistral 2501 FaithEval conclusion.

It is not a fresh confirmatory final-test evaluation. The FaithEval follow-up
reuses the same locked n=200 Mistral FaithEval sample as CP5 after CP5 had
already been inspected. Therefore:

- It can falsify the narrow hypothesis that the CP5 null was mainly caused by
  AUROC-only C selection.
- It cannot by itself establish a new positive intervention claim without a
  fresh pre-registered holdout and matched controls.
- Because the result is null, the absence of fresh random controls is not a
  major blocker for the negative conclusion. If it had been positive, controls
  would have been mandatory before claim escalation.

## Data Authority

| Artifact | Path |
|---|---|
| Wrapper | `scripts/infra/mistral24b_h1_intervention_aware_c.sh` |
| Selector | `scripts/select_intervention_aware_c.py` |
| Classifier sweep metrics | `data/mistral24b/pipeline/h1_intervention_aware_c/classifier_c_sweep_metrics.json` |
| Candidate checkpoints | `data/mistral24b/pipeline/h1_intervention_aware_c/candidate_models/` |
| Selection summary | `data/mistral24b/pipeline/h1_intervention_aware_c/selection_summary.json` |
| Selected classifier | `models/mistral24b_classifier_h1_intervention_aware.pkl` |
| TriviaQA per-C outputs | `data/mistral24b/intervention/triviaqa_bridge_h1_intervention_aware_c/` |
| FaithEval follow-up outputs | `data/mistral24b/intervention/faitheval_h1_intervention_aware_c/experiment/` |
| Completed classifier provenance | `data/mistral24b/pipeline/h1_intervention_aware_c/classifier_c_sweep_metrics.json.provenance.20260430_185115.json` |
| Selection provenance | `data/mistral24b/pipeline/h1_intervention_aware_c/selection_summary.json.provenance.20260430_193425.json` |
| FaithEval provenance | `data/mistral24b/intervention/faitheval_h1_intervention_aware_c/experiment/run_intervention.provenance.20260430_193526.json` |
| Wrapper logs | `logs/mistral24b_h1_intervention_aware_c_20260430T185059Z.log`, `logs/mistral24b_h1_intervention_aware_c_20260430T192053Z.log`, `logs/mistral24b_h1_intervention_aware_c_20260430T193423Z.log`, `logs/mistral24b_h1_intervention_aware_c_20260430T193520Z.log` |

The stale `classifier_c_sweep_metrics.json.provenance.20260430_183455.json`
sidecar records the failed low-memory CPU attempt and has status `running`.
It is not the authority for the completed sweep. The completed provenance
records `status=completed`, git SHA `3a091fb3fab1aa79ec00b43aa44b8aac179f02ce`,
and a clean remote git state.

The run used `mistralai/Mistral-Small-24B-Instruct-2501`, not the paper's exact
Mistral-Small-3.1/2503 checkpoint.

## Method Facts

The wrapper splits the post-CP5 H1 test into four stages:

| Stage | Role | Paid GPU? | Primary guard |
|---|---|---:|---|
| `classifier_c_sweep` | Train L1/liblinear candidate classifiers over C grid on existing CP2 activations | No model GPU load; CPU/RAM heavy | activation validator and active-run guard |
| `triviaqa_c_sweep` | For each candidate, run TriviaQA Bridge at `alpha=0.0` to score model performance under H-neuron suppression | Yes | H100/A100 hardware guard, per-C checkpoint guard, per-C stage/contract skip checks |
| `select_c` | Choose C by `heldout_dev_accuracy + triviaqa_alpha0_deterministic_accuracy_rate` and copy selected model | No | candidate model hash and per-C result hash validation |
| `faitheval` | Run selected model on locked FaithEval n=200, `standard` prompt, alpha grid `0.0..3.0` | Yes | H100/A100 hardware guard, sample lock, stage/contract checks |

The C grid was fixed at:

`0.05, 0.1, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0`.

The selector follows the protocol divergence highlighted from the H-Neurons
paper: use held-out classifier accuracy plus TriviaQA performance when
suppressing selected positive H-neurons. It validates that each per-C TriviaQA
summary is bound to the exact candidate classifier hash before scoring.

## Verification Performed

Local post-sync checks passed:

```bash
uv run python -m scripts.lib.pipeline check-stage \
  --output-dir data/mistral24b/intervention/faitheval_h1_intervention_aware_c/experiment \
  --manifest data/manifests/faitheval_seed42_n200_mistral24b.lock.json \
  --alphas 0.0 0.5 1.0 1.5 2.0 2.5 3.0
```

```bash
uv run python -m scripts.lib.pipeline check-intervention-contract \
  --output-dir data/mistral24b/intervention/faitheval_h1_intervention_aware_c/experiment \
  --benchmark faitheval \
  --model-key mistral_small_24b_instruct_2501 \
  --model-path mistralai/Mistral-Small-24B-Instruct-2501 \
  --classifier-path models/mistral24b_classifier_h1_intervention_aware.pkl \
  --sample-manifest data/manifests/faitheval_seed42_n200_mistral24b.lock.json \
  --alphas 0.0 0.5 1.0 1.5 2.0 2.5 3.0 \
  --benchmark-config prompt_style=standard
```

All ten TriviaQA per-C directories passed `check-stage` with manifest
`data/manifests/triviaqa_bridge_reserve200_seed42.json` and
`--manifest-id-prefix tqa_bridge_`, and all ten passed
`check-intervention-contract` against their exact candidate checkpoint.

Additional checks:

| Check | Result |
|---|---|
| FaithEval sample lock validation | passed |
| Activation-input validator | `accepted=true`; train answer 720/720, train other 720/720, dev answer 200/200, shape errors 0 |
| Focused selector/classifier tests | `uv run pytest tests/test_intervention_aware_c_selector.py tests/test_classifier.py -q` passed: 19 tests |
| Shell wrapper lint | `shellcheck scripts/infra/mistral24b_h1_intervention_aware_c.sh` passed |
| Local active-run status | one unrelated live Gemma SIMID lock; no Mistral H1 live/stale locks copied locally |
| RunPod status after artifact sync | zero pods; retained network volume `eitc8vwogm` only |

## Data

### C Sweep and Selection

The classifier sweep evaluated candidates on the CP2 dev answer-token split.
The final intervention-aware selection score is:

`heldout_dev_accuracy + TriviaQA alpha=0 deterministic accuracy under H-neuron suppression`.

| C | Positive H-neurons | Dev accuracy | TriviaQA suppress accuracy | Selection score |
|---:|---:|---:|---:|---:|
| 0.05 | 0 | 0.500 | 0.675 | 1.175 |
| 0.10 | 1 | 0.500 | 0.680 | 1.180 |
| 0.30 | 5 | 0.695 | 0.660 | 1.355 |
| 0.50 | 7 | 0.715 | 0.665 | 1.380 |
| **0.75** | **9** | **0.735** | **0.660** | **1.395** |
| 1.00 | 10 | 0.745 | 0.630 | 1.375 |
| 1.25 | 12 | 0.740 | 0.620 | 1.360 |
| 1.50 | 13 | 0.745 | 0.605 | 1.350 |
| 2.00 | 28 | 0.750 | 0.365 | 1.115 |
| 3.00 | 80 | 0.780 | 0.240 | 1.020 |

The selected model hash is
`9ffa2f1a13a12b5742590c18c44b8997bf67597ae9b6b544f1762a1f87ca9671`.

The selected positive coefficients are:

| Rank | Layer | Neuron | Coef |
|---:|---:|---:|---:|
| 1 | 16 | 21660 | 6.5966 |
| 2 | 20 | 24498 | 5.1745 |
| 3 | 36 | 19342 | 2.9037 |
| 4 | 36 | 11109 | 2.4005 |
| 5 | 19 | 6029 | 1.5191 |
| 6 | 17 | 5215 | 1.5134 |
| 7 | 15 | 23885 | 0.4260 |
| 8 | 17 | 19878 | 0.3234 |
| 9 | 18 | 30843 | 0.2578 |

Compared with the original CP3/CP5 `C=1.0` classifier, the selected H1 model
removes one positive target and improves TriviaQA suppress accuracy by 3.0 pp
on the reserve-200 bridge set, at a 1.0 pp dev-accuracy cost.

### Selection Robustness

I recomputed candidate dev predictions from the saved candidate checkpoints and
ran a paired bootstrap over the dev items and TriviaQA reserve items
independently, 20,000 resamples, seed 42. The exact C=0.75 win is not stable
against nearby candidates:

| Comparison | Point score diff: C=0.75 minus candidate | 95% bootstrap interval |
|---|---:|---:|
| vs C=0.30 | +0.040 | [-0.010, +0.090] |
| vs C=0.50 | +0.015 | [-0.025, +0.055] |
| vs C=1.00 | +0.020 | [-0.015, +0.055] |
| vs C=1.25 | +0.035 | [-0.015, +0.085] |
| vs C=1.50 | +0.045 | [-0.010, +0.100] |
| vs C=2.00 | +0.280 | [+0.200, +0.365] |
| vs C=3.00 | +0.375 | [+0.285, +0.465] |

So the sweep strongly rejects broad/high-C candidates under the coupled score,
but it does not uniquely identify C=0.75 over the local plateau around
`0.5 <= C <= 1.5`.

### TriviaQA Per-C Paired Texture

Against C=0.75 on the same 200 TriviaQA Bridge reserve items:

| Candidate | C=0.75 correct, candidate wrong | C=0.75 wrong, candidate correct | C=0.75 minus candidate |
|---:|---:|---:|---:|
| 0.05 | 5 | 8 | -1.5 pp |
| 0.10 | 4 | 8 | -2.0 pp |
| 0.30 | 2 | 2 | 0.0 pp |
| 0.50 | 1 | 2 | -0.5 pp |
| 1.00 | 6 | 0 | +3.0 pp |
| 1.25 | 10 | 2 | +4.0 pp |
| 1.50 | 14 | 3 | +5.5 pp |
| 2.00 | 63 | 4 | +29.5 pp |
| 3.00 | 86 | 2 | +42.0 pp |

The high-C candidates are not merely lower by independent-rate noise; they
cause many item-level right-to-wrong losses under suppression.

### FaithEval Follow-Up

The selected C=0.75 classifier was then run on the locked n=200 Mistral
FaithEval manifest with `prompt_style=standard`.

| Alpha | Compliance | Rate | Parse failures |
|---:|---:|---:|---:|
| 0.0 | 103/200 | 0.515 | 0 |
| 0.5 | 104/200 | 0.520 | 0 |
| 1.0 | 103/200 | 0.515 | 0 |
| 1.5 | 103/200 | 0.515 | 0 |
| 2.0 | 104/200 | 0.520 | 0 |
| 2.5 | 104/200 | 0.520 | 0 |
| 3.0 | 104/200 | 0.520 | 0 |

Paired item effects:

| Contrast | Delta | 95% paired bootstrap interval | Flips |
|---|---:|---:|---|
| alpha 0.0 -> 3.0 | +0.5 pp | [-3.0, +4.0] | 7 false-to-true, 6 true-to-false |
| alpha 1.0 -> 3.0 | +0.5 pp | [-2.0, +3.0] | 4 false-to-true, 3 true-to-false |
| alpha 0.0 -> 1.0 | 0.0 pp | [-3.0, +3.0] | 5 false-to-true, 5 true-to-false |

Linear slope over the seven alpha means is +0.143 pp/alpha. Spearman
rho(alpha, compliance rate) is 0.577, but that monotonic-looking rank statistic
is carried by one-item plateaus and has no meaningful endpoint effect.

### Comparison to CP5 Canonical Classifier

On the same FaithEval item manifest, H1 does not improve over the original CP5
H-neuron run:

| Alpha | CP5 canonical C=1.0 | H1 C=0.75 | H1 minus CP5 |
|---:|---:|---:|---:|
| 0.0 | 0.530 | 0.515 | -1.5 pp |
| 0.5 | 0.495 | 0.520 | +2.5 pp |
| 1.0 | 0.520 | 0.515 | -0.5 pp |
| 1.5 | 0.520 | 0.515 | -0.5 pp |
| 2.0 | 0.540 | 0.520 | -2.0 pp |
| 2.5 | 0.540 | 0.520 | -2.0 pp |
| 3.0 | 0.530 | 0.520 | -1.0 pp |

The comparison is descriptive because both curves were observed on the same
locked post-hoc sample; it is still useful as a sanity check. The new selector
does not reveal a hidden endpoint effect.

## Pipeline Review

### What Withstands Scrutiny

The H1 wrapper had the right shape for a minimal-GPU follow-up. It split CPU
and GPU stages, required `H1_REVIEWED=1` before GPU stages, ran the H100/A100
hardware guard before model-loading stages, and did not rerun activations. The
per-C TriviaQA stage refused missing checkpoints before loading the model, and
the selector checked that each per-C TriviaQA summary was hash-bound to the
candidate checkpoint it was scoring.

The selector implements the intended coupled score and makes the selected model
traceable. The chosen checkpoint hash in `selection_summary.json` matches the
copied `models/mistral24b_classifier_h1_intervention_aware.pkl`, and the same
hash appears in the FaithEval `intervention_run_config.json`.

The result is scientifically interpretable as a null-classification follow-up:
the strongest cheap post-CP5 protocol mismatch was tested directly; the
output stayed null. That is materially better evidence than leaving H1 as an
untested speculation.

The high-C collapse on TriviaQA is also a real finding. It shows why the paper
couples detection with suppression utility: detector accuracy keeps improving
through C=3.0, while suppression utility falls off a cliff. For this Mistral
2501 setup, selecting solely on dev detection accuracy would choose an
80-neuron model that is destructive on the TriviaQA suppression criterion.

### What Does Not Withstand Scrutiny

The exact C=0.75 winner should not be overread. The point estimate is the
winner under the implemented score, but bootstrap intervals against nearby
candidates cross zero. A future report should say "the coupled criterion moves
the selected model into the local C=0.5-1.0/1.5 plateau" rather than "C=0.75
is uniquely optimal."

The H1 run does not support any positive Mistral intervention claim. The
FaithEval endpoint and no-op-to-max effects are effectively zero, item flips
are nearly balanced, and the rate curve moves by at most one item over 200.
This is even flatter than CP5.

The H1 run also does not prove that no Mistral checkpoint can reproduce the
paper. It only tests the 2501 causal-LM anchor, the existing CP2 split family,
the standard FaithEval prompt, the multiplicative positive-weight FFN scaling
operator, and the reserve-200 TriviaQA C-selection proxy.

The C-sweep is post-hoc after CP5. It is appropriate as a falsification of the
H1 explanation, not as a new model-selection path that can be used to make a
claim on the same FaithEval lock.

### Pipeline Gaps Worth Fixing

These are not reasons to rerun H1, but they would improve future safety:

1. The wrapper default `STAGES=all` is operationally dangerous for a
   minimal-GPU workflow. The `H1_REVIEWED=1` guard prevents accidental GPU
   launch, but defaulting a paid wrapper to all stages invites mistakes.
   Prefer an explicit stage requirement for future high-cost wrappers.
2. The selector allows zero-positive-neuron candidates. C=0.05 did not win
   here because dev accuracy was 0.5, but a minimum-target guard would avoid
   meaningless "no intervention" candidates in future coupled selectors.
3. The selected C is reported without uncertainty. The report-level bootstrap
   above should either be automated into future selection summaries or the
   summary should explicitly mark the point-estimate winner as heuristic.
4. The completed local artifact set includes the failed CPU-attempt provenance
   sidecar with `status=running`. It is preserved correctly, but any commit or
   archive should document that `20260430_185115` is the completed classifier
   provenance authority.

## Interpretation

The H1 hypothesis was plausible before this run because the H-Neurons paper
explicitly selects C using a coupled detection-plus-suppression criterion, and
our CP3/CP5 classifier had been selected by detector quality alone. That was a
real protocol mismatch, not a post-hoc rationalization.

After the H1 run, that mismatch is no longer a sufficient explanation for the
Mistral 2501 FaithEval null. The coupled criterion did what it was supposed to
do on the selection proxy: it avoided high-C models that damaged TriviaQA under
suppression. But the selected 9-neuron model still produced no FaithEval
dose-response. The most conservative reading is:

> Mistral 2501 has a sparse TriviaQA-derived hallucination readout, and the
> coupled C-selection rule can avoid obvious TriviaQA utility damage, but under
> the current multiplicative positive-FFN intervention it still does not move
> FaithEval compliance on the locked n=200 standard-prompt sample.

This pushes explanatory weight away from "we picked the wrong C" and toward
one or more of:

- the 2501-vs-2503 checkpoint/post-training difference;
- benchmark/surface specificity: TriviaQA suppression utility is not a proxy
  for FaithEval context-compliance steering on this model;
- operator specificity: positive-weight multiplicative FFN scaling may not be
  the causal handle for the Mistral 2501 readout;
- sample/prompt limitations that can hide small effects but cannot explain the
  complete lack of endpoint movement by themselves.

The high-C TriviaQA collapse is the most interesting positive insight. It
supports the paper's methodological concern that better detector accuracy can
select too many functionally important neurons and degrade utility. In this
run, however, that insight is about avoiding damage, not about finding a
FaithEval steering lever.

## Uncertainty Register

| Uncertainty | Level | Why it remains |
|---|---|---|
| Exact best C among 0.5, 0.75, 1.0, 1.25, 1.5 | High | Point differences are 1.5-4.5 pp on the summed score; bootstrap intervals cross zero |
| Whether any rejected high-C candidate would move FaithEval | Medium | Not run; high-C candidates are rejected by the paper-style utility criterion because TriviaQA suppression accuracy collapses |
| Whether a fresh larger FaithEval sample would reveal a tiny positive effect | Medium | n=200 paired CI is still several pp wide; observed endpoint is only +0.5 pp |
| Whether Mistral 2503 would behave differently | Medium-high | 2501 is a same-family anchor, not the exact paper checkpoint; 2503 migration was not done |
| Whether a different operator/basis could steer FaithEval | Medium | This run tests positive-weight multiplicative FFN scaling only |
| Whether prompt style suppresses an effect | Medium | Only `standard` prompt was used for the H1 follow-up |
| Whether the C-selection proxy transfers from TriviaQA to FaithEval | High | The null suggests weak transfer, but no alternative proxy was evaluated |

## Research Guidance

Use this result to close the H1 follow-up, not to launch another local C tweak.
The paper-facing statement should be:

> On the Mistral-Small-24B-Instruct-2501 anchor, the H-neuron readout remains
> detectable and the intervention-aware C sweep avoids TriviaQA utility damage,
> but neither AUROC-selected nor intervention-aware-selected positive FFN
> scaling produces a FaithEval dose-response on the locked n=200 standard
> prompt.

Avoid:

> The C sweep found the right Mistral H-neurons.

and:

> Mistral fails because we used 2501 instead of 2503.

The first is contradicted by the FaithEval null. The second remains plausible
but untested.

## Most Valuable Next Steps

1. Treat the Mistral 2501 FaithEval H-neuron branch as closed for the current
   ICML limitation-response path. CP5 and H1 together are enough: readout yes,
   FaithEval intervention no.
2. Update manuscript and reviewer framing to present the Mistral result as a
   useful audit failure, not as a hidden replication. The stronger story is
   that the audit scaffold catches a readout-to-control break across models.
3. If new Mistral GPU is authorized, require a new pre-registered question.
   The most defensible options are a prompt/measurement-stability audit or a
   real 2503 migration, not another 2501 C-grid tweak.
4. Add a non-spend code hardening item: either disallow zero-H-neuron
   candidates in coupled selectors or explicitly classify them as no-op
   baselines, and add selection-uncertainty output to future C-selection
   summaries.
5. Preserve the copied local artifacts before deleting the retained RunPod
   volume. If committing run outputs, re-run `active-run-status` first; a
   local unrelated Gemma SIMID lock was live during this review.
