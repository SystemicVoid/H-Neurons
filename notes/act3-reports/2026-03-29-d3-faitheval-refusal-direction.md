# D3 FaithEval Refusal-Direction Closeout

## Question

Does all-layer refusal-direction ablation on FaithEval behave like a robust analogue of the H-neuron intervention curve, or does it only provide a narrow diagnostic signal about refusal overlap?

## Artifacts Used

- `data/gemma3_4b/intervention/faitheval_direction_ablate_d3_calibrated/experiment/results.20260329_192019.json`
- `data/gemma3_4b/intervention/faitheval_direction_ablate_d3_calibrated/experiment/run_intervention.provenance.20260329_192019.json`
- `data/gemma3_4b/intervention/faitheval_direction_ablate_d3_calibrated/experiment/alpha_0.0.jsonl`
- `data/gemma3_4b/intervention/faitheval_direction_ablate_d3_calibrated/experiment/alpha_0.02.jsonl`
- `data/gemma3_4b/intervention/faitheval_direction_ablate_d3_calibrated/experiment/alpha_0.03.jsonl`
- Exploratory-only context: `data/gemma3_4b/intervention/faitheval_direction_ablate_all-layers_refusal-directions_c892775a83/`, `data/gemma3_4b/intervention/faitheval_direction_ablate_calibration/`, `data/gemma3_4b/intervention/faitheval_direction_ablate_microbeta_calibration/`

## Clean Headline Result

The clean full run is the decision-relevant D3 artifact. It was executed from a clean worktree (`git_dirty: false`) at commit `9e960722bf4760a8679068314019daee39713a9a`, with all-layer direction ablation on FaithEval at β ∈ {0.00, 0.02, 0.03}.

Headline compliance:

| β | Compliance |
|---|---|
| 0.00 | 66.0% (660/1000) |
| 0.02 | 70.2% (702/1000) |
| 0.03 | 51.1% (511/1000) |

Parse failures are 0/1000 at all three settings. That removes parser-failure ambiguity, but it does not make the curve clean.

The inherited aggregate `delta_0_to_max_pp` and slope fields in `results.20260329_192019.json` are not the right headline because the curve is non-monotonic. The decision-relevant evidence is the per-β row behavior.

## Row-Level Audit

### Paired Transition Counts

| Transition | Better | Worse | Net |
|---|---|---|---|
| β=0.00 → β=0.02 | 105 False→True | 63 True→False | +42 |
| β=0.02 → β=0.03 | 45 False→True | 236 True→False | -191 |

β=0.02 is a real but modest paired lift. The β=0.03 collapse is large and asymmetric.

### Mean Generated Tokens

| β | Mean generated tokens |
|---|---|
| 0.00 | 2.157 |
| 0.02 | 2.407 |
| 0.03 | 3.506 |

The token increase at β=0.03 is real, but the main failure mode in the clean run is not parser collapse. It is answer-distribution distortion.

### Response-Format Drift

| β | Bare letter | `A)` style | Extra-text / numeric label |
|---|---|---|---|
| 0.00 | 981 (98.1%) | 0 (0.0%) | 19 (1.9%) |
| 0.02 | 760 (76.0%) | 221 (22.1%) | 19 (1.9%) |
| 0.03 | 526 (52.6%) | 455 (45.5%) | 19 (1.9%) |

Format drift is mostly from bare letters toward option-marker answers such as `A)` and `B)`. The extra-text bucket stays flat and is mostly numeric option labels like `2)` or `3`, not a large wave of long-form free text in this clean run.

### Chosen-Option Distribution

| β | A | B | C | D | Numeric labels (`1`-`4`) |
|---|---|---|---|---|---|
| 0.00 | 222 (22.2%) | 266 (26.6%) | 245 (24.5%) | 248 (24.8%) | 19 (1.9%) |
| 0.02 | 254 (25.4%) | 282 (28.2%) | 231 (23.1%) | 214 (21.4%) | 19 (1.9%) |
| 0.03 | 240 (24.0%) | 581 (58.1%) | 69 (6.9%) | 91 (9.1%) | 19 (1.9%) |

This is the important refinement. At β=0.03 the model does not simply stop answering. It heavily over-selects `B`. The largest answer transitions from β=0.02 to β=0.03 are `C→B` (142 rows) and `D→B` (100 rows).

### Per-Correct-Option Compliance

| Correct option | β=0.00 | β=0.02 | β=0.03 |
|---|---|---|---|
| A | 160/263 = 60.8% | 194/263 = 73.8% | 173/263 = 65.8% |
| B | 167/237 = 70.5% | 178/237 = 75.1% | 207/237 = 87.3% |
| C | 151/236 = 64.0% | 163/236 = 69.1% | 53/236 = 22.5% |
| D | 182/264 = 68.9% | 167/264 = 63.3% | 78/264 = 29.5% |

If β=0.03 were only a parser problem, these buckets would fail more uniformly. Instead, the model becomes unusually strong on correct-`B` items and very weak on correct-`C`/`D` items. That is consistent with answer-option bias on an MCQ benchmark.

## Why This Is Not A Robust Baseline B Curve

- There is a narrow usable point at β=0.02, not a broad stable intervention band.
- β=0.03 is already a cliff.
- The clean run removes parser-failure ambiguity, but row-level audit shows answer-option bias / output-distribution distortion at β=0.03.
- This is not a robust refusal-ablation analogue of the H-neuron FaithEval curve.
- Baseline B is informative diagnostically, but it is not currently a strong candidate for wider D3 expansion.

## Interpretation

Baseline B may have a narrow usable β=0.02 point, but it is not a robust refusal-ablation analogue of the H-neuron FaithEval curve.

The clean run removes parser-failure ambiguity, but row-level audit suggests answer-option bias / output-distribution distortion at β=0.03. That makes FaithEval interpretable only with MCQ-specific diagnostics, not with compliance and parse-failure summaries alone.

The scientific value of D3 is therefore diagnostic. It says the refusal-direction family may touch part of the same geometry as the H-neuron intervention, but the intervention is fragile enough that it should not be framed as a promising mitigation branch on current evidence.

## Decision

D3 is decision-complete on FaithEval.

- Do not broaden D3.
- If Baseline B is carried forward at all, carry β=0.02 only as one scoped externality-check row in D5.
- Otherwise move directly to D3.5.

## Remaining Caveats

- This is still a single-benchmark readout. There is no cross-benchmark robustness claim here.
- The clean run removes one ambiguity, not all ambiguity. Zero parse failures does not certify clean steering behavior on MCQ tasks.
- The row-level audit is benchmark-specific. It sharpens the FaithEval interpretation; it does not by itself say what will happen on jailbreak or capability evals.

## Recommended Next Action

Prioritize D3.5 refusal-overlap analysis. D3 was valuable as a diagnostic probe of the refusal-overlap story; it is not currently a promising mitigation branch.

## Doc Hygiene

- Raw artifact filenames still use `alpha_*` because they come from a shared intervention harness. In the narrative for direction steering, the scientific variable is β.
- The `n_h_neurons` field in `results.20260329_192019.json` is a stale schema artifact from the shared harness and should not be interpreted scientifically for direction runs.
