# Mistral 24B CP2/CP3 Pipeline Review (2026-04-29)

> Canonical review for the completed Mistral 24B CP2/CP3 split, activation, and
> sparse-classifier gate under `data/mistral24b/pipeline/`. This supersedes the
> older GH200-era Mistral pipeline report for held-out detector status only. It
> does not supersede the Mistral execution ledger in
> [`notes/icml/mistral24b/2026-04-28-5.5-pro-l1-mitigation-strategy.md`](../../../notes/icml/mistral24b/2026-04-28-5.5-pro-l1-mitigation-strategy.md),
> which remains the progress tracker and source for downstream CP4+ decisions.

## Bottom Line

CP2/CP3 passed as a held-out detector-identification gate. The run produced
strictly disjoint train/dev/test splits, complete canonical activation arrays
for all required split/location pairs, and a saved L1/liblinear classifier whose
held-out test performance is above chance with recorded 95% stratified
bootstrap intervals: accuracy 0.7750 [0.715, 0.830], F1 0.7783 [0.7184,
0.8349], AUROC 0.8711 [0.8185, 0.9172].

The result is scientifically useful but narrower than a paper-facing replication
claim. It supports: "the Mistral 2501 same-family anchor has a sparse,
held-out, answer-token hallucination readout under the current pipeline." It
does not yet support: "H-neuron interventions replicate on Mistral", "the
result is causal", "the exact Mistral-Small-3.1/2503 checkpoint replicates", or
"the selected neurons are stable across split seeds."

Two review findings should be kept explicit. First, `scripts/classifier.py`
counts positive logistic-regression coefficients as selected H-neurons, matching
the H-Neurons paper's perturbation target definition; the saved model also has
10 negative nonzero coefficients, which are useful to the classifier but should
not be called intervention targets without a new rule. Second, the usable false
class is almost exhausted: `answer_tokens_llm.jsonl` has 565 false rows, and
train/dev/test consume 560 of them. That is acceptable for this frozen gate but
limits fresh same-source holdout or split-stability work unless more false rows
are generated or repaired.

## Data Authority

| Artifact | Path |
|---|---|
| Source answer tokens | `data/mistral24b/answer_tokens_llm.jsonl` |
| Train/dev/test split IDs | `data/mistral24b/pipeline/{train,dev,test}_qids_llm.json` |
| Activation root | `data/mistral24b/pipeline/activations_llm_canonical/` |
| Activation summaries | `data/mistral24b/pipeline/activations_llm_canonical/{train,dev,test}/summary.json` |
| Classifier model | `models/mistral24b_classifier_canonical.pkl` |
| Dev metrics | `data/mistral24b/pipeline/classifier_canonical_dev_metrics.json` |
| Test metrics | `data/mistral24b/pipeline/classifier_canonical_test_metrics.json` |
| Run log | `logs/mistral24b_replication_20260429T162910Z.log` |
| Validation script | `scripts/validate_mistral24b_cp23.py` |
| Execution ledger | `notes/icml/mistral24b/2026-04-28-5.5-pro-l1-mitigation-strategy.md` |

Operational cleanup is also closed: pod `kcp3vktzc5mdru` was deleted and
`cloudctl status` reported zero pods. Network volume `eitc8vwogm` remains in
`US-CA-2`, so storage-only spend remains until that volume is deleted.

## Data

### Source Pool and Splits

The source file has 2,539 extracted answer-token rows:

| Label | Source rows | Used in train/dev/test | Remaining |
|---|---:|---:|---:|
| True | 1,974 | 560 | 1,414 |
| False | 565 | 560 | 5 |

The split sampler used seeds 42/43/44 with strict per-class counts and explicit
exclusion paths for dev and test. Split counts and overlap checks are:

| Split | False | True | Total | Overlap with other splits |
|---|---:|---:|---:|---:|
| Train | 360 | 360 | 720 | 0 |
| Dev | 100 | 100 | 200 | 0 |
| Test | 100 | 100 | 200 | 0 |

All 1,120 split IDs are present in `answer_tokens_llm.jsonl`. The false-class
near-exhaustion is not leakage, but it is a real sampling constraint.

### Activation Extraction

All required activation locations completed with zero missing regions:

| Split/location | Files | Shape | Dtype | Size contribution |
|---|---:|---|---|---:|
| `train/answer_tokens` | 720 | `(40, 32768)` | `float32` | 3.6G |
| `train/all_except_answer_tokens` | 720 | `(40, 32768)` | `float32` | 3.6G |
| `dev/answer_tokens` | 200 | `(40, 32768)` | `float32` | 1001M |
| `test/answer_tokens` | 200 | `(40, 32768)` | `float32` | 1001M |
| Total | 1,840 | `(40, 32768)` | `float32` | 9.0G |

The flattened feature width is 1,310,720, matching 40 layers x 32,768 FFN
neurons. `metadata.json` records `aggregation_method="mean"`,
`use_abs=true`, `use_mag=true`, `device_map="cuda:0"` in provenance, model key
`mistral_small_24b_instruct_2501`, and tokenizer kwargs
`fix_mistral_regex=true` via the registry metadata.

Answer-token spans are short, which is expected for TriviaQA answer-only
responses: train median 3 tokens, dev median 3, test median 3.

### Classifier Training and Evaluation

Training used the paper-style asymmetric 3-vs-1 setup: false answer tokens are
positive; true answer tokens plus true/false non-answer tokens are negative.
Thus the train matrix has 1,440 examples: 360 positive, 1,080 negative. Dev and
test evaluation are balanced 1-vs-1 answer-token sets with 100 positive and 100
negative examples each.

The C sweep selected `C=1.0` by dev AUROC:

| C | Positive selected neurons | Dev accuracy | Dev F1 | Dev AUROC |
|---:|---:|---:|---:|---:|
| 0.001 | 0 | 0.500 | 0.000 | 0.500 |
| 0.005 | 0 | 0.500 | 0.000 | 0.500 |
| 0.010 | 0 | 0.500 | 0.000 | 0.500 |
| 0.050 | 0 | 0.500 | 0.000 | 0.410 |
| 0.100 | 1 | 0.500 | 0.000 | 0.636 |
| 0.500 | 7 | 0.715 | 0.716 | 0.797 |
| 1.000 | 10 | 0.745 | 0.756 | 0.847 |

Held-out metrics for the selected and saved model:

| Split | Accuracy | Precision | Recall | F1 | AUROC | Confusion |
|---|---:|---:|---:|---:|---:|---|
| Dev | 0.745 [0.685, 0.805] | 0.7248 [0.6636, 0.7895] | 0.790 [0.710, 0.870] | 0.7560 [0.6970, 0.8113] | 0.8474 [0.7941, 0.8954] | TP 79, TN 70, FP 30, FN 21 |
| Test | 0.775 [0.715, 0.830] | 0.7670 [0.7027, 0.8333] | 0.790 [0.710, 0.870] | 0.7783 [0.7184, 0.8349] | 0.8711 [0.8185, 0.9172] | TP 79, TN 76, FP 24, FN 21 |

Intervals are 10,000-resample, seed-42, stratified-by-label percentile
bootstraps. They describe uncertainty over the frozen balanced evaluation
items, not architecture-level or split-seed uncertainty.

The saved classifier has 20 nonzero coefficients: 10 positive and 10 negative.
The pipeline's `selected_h_neurons` field counts only positive coefficients.

| Positive target | Layer | Neuron | Flat index | Coefficient |
|---:|---:|---:|---:|---:|
| 1 | 15 | 23885 | 515405 | 0.7650 |
| 2 | 15 | 26931 | 518451 | 0.2192 |
| 3 | 16 | 21660 | 545948 | 6.4416 |
| 4 | 17 | 5215 | 562271 | 2.4688 |
| 5 | 17 | 19878 | 576934 | 1.0926 |
| 6 | 18 | 30843 | 620667 | 0.4471 |
| 7 | 19 | 6029 | 628621 | 1.1727 |
| 8 | 20 | 24498 | 679858 | 4.9085 |
| 9 | 36 | 11109 | 1190757 | 2.9282 |
| 10 | 36 | 19342 | 1198990 | 3.7420 |

Layer distribution for positive targets is mostly middle layers with two late
layer-36 neurons: layers 15, 17, and 36 each contribute two targets; layers 16,
18, 19, and 20 contribute one each.

## Verification Performed

I reran the project validator:

```bash
uv run python scripts/validate_mistral24b_cp23.py
```

It returned `accepted=true`, `status=ok`, with all split, activation, model-key,
model-path, classifier-width, metrics, and provenance checks passing.

I also independently checked:

| Check | Result |
|---|---|
| `.npy` file count | 1,840 |
| Activation root size | 9.0G |
| Shape scan over all `.npy` files | 1,840/1,840 are `(40, 32768)` |
| Dtype scan over all `.npy` files | 1,840/1,840 are `float32` |
| Split IDs missing from source file | 0 |
| Train/dev/test pairwise overlaps | 0 / 0 / 0 |
| Source false rows remaining after split use | 5 |
| Focused classifier test | `uv run pytest tests/test_classifier.py -q` passed: 7 tests |
| Reporting coverage audit | `uv run python scripts/audit_ci_coverage.py` passed |
| Active run guard | `uv run python -m scripts.lib.pipeline active-run-status` reported 0 live/stale/malformed locks |

## Interpretation

### What Withstands Scrutiny

The strongest conclusion is a detector-readout conclusion. This run materially
improves the Mistral state from "operational artifacts and training-only
metrics" to "held-out sparse detector on disjoint canonical splits." The test
AUROC interval is well above 0.5, and test accuracy/F1 are close to dev despite
selection on dev AUROC. The feature width, layer count, model key, and tokenizer
kwargs all match the registry-defined Mistral 2501 target, so this is not a
Gemma-shaped artifact accidentally routed through Mistral filenames.

The sparsity is also in the expected order of magnitude. Ten positive targets
out of 1,310,720 FFN neurons is 0.00763 per mille. The H-Neurons paper reports
Mistral-Small-3.1-24B in roughly the 0.01 per mille regime and defines
positive-weight neurons as the perturbation candidate set
([H-Neurons](../../../papers/h-neurons-hallucination-correlated.md)). This is
not exact replication, but it is directionally consistent.

The run also closes an important engineering risk: the canonical answer-token
activation path works on a 24B-class Mistral model with the current tokenizer
settings and produces complete downstream-ready arrays. That matters because
Mistral-specific tokenizer and checkpoint routing were major pre-run risks.

### What Does Not Yet Follow

Classification is not causality. The H-Neurons paper itself separates Q1
identification from Q2 perturbation, and states that causal claims require
controlled interventions. This run only completes Q1 for the same-family
Mistral anchor. CP4/CP5 are still required before any Mistral intervention or
over-compliance claim is defensible.

Held-out performance is not enough to infer a mechanism. Probe and steering
literature gives the right skeptical frame: high probe accuracy can reflect
features that are predictive under the sampled distribution rather than cleanly
causal or invariant, and steering results can depend heavily on prompt format,
baseline choice, and evaluation context
([Control Tasks for Probes](../../../papers/Designing%20and%20Interpreting%20Probes%20with%20Control%20Tasks-D19-1275.md),
[Probes Are Unreliable](../../../papers/probes-unreliable-2207.04153.md),
[Reliable Steering Evaluation](../../../papers/reliable-steering-eval-2410.17245.md)).
Therefore, the report should not be used as a shortcut around matched controls,
prompt/parser audits, or FaithEval pilot smoke.

The exact-checkpoint caveat remains. The model is
`mistralai/Mistral-Small-24B-Instruct-2501`, not the paper's
Mistral-Small-3.1/2503-style checkpoint. This is a credible same-family
limitation-response anchor, not an exact paper checkpoint reproduction.

### Uncertainty and Risk Register

| Risk | Severity | Current evidence | Consequence |
|---|---|---|---|
| False-class near-exhaustion | Medium | 560/565 false source rows consumed | Fresh same-source holdout and split-seed replication require new/ repaired false rows |
| Split-seed instability unknown | Medium | One train/dev/test split family only | Selected-neuron identity and metrics may be sample-specific |
| Detector null missing | Medium | No Mistral matched-random-neuron detector baseline in this CP3 package | Do not claim H-neurons beat random features on Mistral yet |
| Intervention causality absent | High | No CP4/CP5 intervention outputs | No Mistral behavioral-control claim |
| Exact-checkpoint mismatch | Medium | 2501 causal-LM path by design | Wording must say same-family 2501, not Mistral Small 3.1/2503 |
| Label/judge artifact debt | Medium | Source file inherits LLM answer-token extraction and 63 skipped malformed rows from prior Step 2 | Detector may reflect extraction/judge-filtered subset; report the source-pool construction |
| Negative coefficients easy to overread | Low/Medium | Saved model has 10 negative nonzero weights | Only positive weights are H-neuron intervention targets under current method |

## Scientific Position After CP3

The Mistral branch is now worth continuing, but only as a gated audit. It has
earned CP4 FaithEval pilot/control smoke; it has not earned manuscript language
claiming cross-model intervention replication. The most balanced wording is:

> On a same-family Mistral 24B 2501 anchor, the paper-style sparse detector gate
> replicated at the readout level: 10 positive FFN targets achieved held-out
> test AUROC 0.871 [0.818, 0.917] on disjoint TriviaQA-derived answer-token
> splits. Behavioral intervention replication remains untested.

Avoid:

> Mistral replicates the H-neuron effect.

That sentence conflates held-out readout with controlled perturbation and hides
the exact-checkpoint caveat.

## Most Valuable Next Steps

1. Run CP4 FaithEval pilot/control smoke before any full FaithEval claim run.
   The pilot must include the H-neuron path, at least one unconstrained random
   control, one layer-matched control, shared manifest, shared prompt style,
   shared evaluator/parser, and explicit prompt/parser audit rows.
2. Before CP5, freeze the evaluator/rubric IDs and resolve whether full
   controls are 5 unconstrained + 3 layer-matched or exactly 3 + 3. Do this
   before seeing full FaithEval outcomes.
3. Add a Mistral detector null if the paper will compare Mistral readout
   quality to the H-Neurons paper's random-neuron baseline. A matched-count
   random-neuron classifier on the same train/dev/test split would separate
   "sparse signal exists" from "selected sparse signal is unusually good."
4. Record split-stability only if the false pool is expanded. With only five
   unused false rows, repeated same-source splits are mostly resamples of the
   same minority pool and should not be oversold.
5. Keep the network-volume decision explicit before the next paid run:
   `eitc8vwogm` remains useful if CP4/CP5 run soon in `US-CA-2`; otherwise its
   storage spend should be deliberately retained or deleted.
