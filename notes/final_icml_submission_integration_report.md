# Final ICML Submission Integration Report

Date: 2026-05-07

## Files Changed

- `paper/icml/main.tex`
- `paper/icml/main.pdf`
- `paper/icml/figures/fig3_bridge.py`
- `paper/icml/figures/fig3_bridge.pdf`
- `scripts/check_icml_prose.py`
- `notes/final_icml_submission_integration_report.md`

## Build Command

- `make` from `paper/icml/`
- Output PDF: `paper/icml/main.pdf`

## Validation Results

- `make`: pass. PDF built with BibTeX and final LaTeX passes.
- `rg -n "(LaTeX Warning:|Citation.*undefined|Reference.*undefined|There were undefined|Overfull|Float too large|^!)" paper/icml/main.log`: pass, no hits.
- `uv run python scripts/check_icml_prose.py paper/icml/main.tex`: pass with 0 failures and 7 warnings. Warnings are ordinary related-work/discussion uses of "steering" without nearby model/operator words; no banned contrast pattern, em dash, or forbidden overclaim remains.
- `uv run python paper/icml/figures/fig3_bridge.py`: pass. Figure 3 regenerated from the adjudicated bridge IRR summary.
- Forbidden-claim grep over `paper/icml/main.tex`: pass, no hits for Mistral replication, Mistral L1 closure, SIMID truthfulness improvement, SAE failure, TruthfulQA transfer, settled jailbreak measurement, wrong-entity mechanism, or readout-identifies-steering-target language.
- Banned prose-pattern grep over `paper/icml/main.tex`, `paper/icml/figures/fig3_bridge.py`, and this report: pass, no hits.
- `ruff format --check scripts paper/icml/figures/fig3_bridge.py`: pass, 118 files already formatted.
- `ruff check scripts paper/icml/figures/fig3_bridge.py`: pass.
- `ty check`: pass.
- `git diff --check`: pass.
- Figure existence check: pass for `fig1_scaffold.pdf`, `fig2_matched.pdf`, and `fig3_bridge.pdf`.
- `coderabbit review --agent --base-commit a04d2e2 --no-color`: completed with 1 minor finding in `scripts/check_icml_prose.py`; fixed by adding `verbatim*` environment handling.

## Headline Claim Ledger Summary

- Gemma FaithEval readout-to-control: H-neurons and tested Gemma Scope SAE features have comparable readout AUROC (0.843 vs. 0.848), while only H-neuron scaling passes the FaithEval compliance control gate (+2.09 pp/alpha [1.38, 2.83]; neuron-minus-SAE +1.93 pp/alpha [+0.94, +2.92]).
- Within-SAE selector audit: readout, prompt-end utility, and answer-span selectors inside the same 509-feature Gemma SAE pool produce metric-specific margin effects, but all miss the compliance endpoint.
- Externality: Gemma ITI improves TruthfulQA MC answer selection (MC1 +6.3 pp [3.7, 8.9]; MC2 +7.49 pp [5.28, 9.82]) while reducing open-ended TriviaQA bridge accuracy (-5.8 pp [-8.8, -3.0]); wrong-entity substitution is retained as behavioral taxonomy only.
- Measurement: Gemma jailbreak conclusions depend on scoring granularity and evaluator construct; output-length exposure remains a reporting risk. CSV-v3 is retained for taxonomy with no ground-truth superiority claim.
- Stress tests: Mistral 2501 readout passes, FaithEval and TruthfulQA MC1 gates fail, JailbreakBench curve is large but uncontrolled; SIMID remains diagnostic-only after failed calibration/effect gates.

## Evidence Conflicts Resolved

- Mistral strategy prose was treated as non-authoritative where later direct reports existed. The manuscript uses CP3 readout, CP5/H1 FaithEval, Anchor 2 TruthfulQA MC, and Anchor 3 JailbreakBench numbers from their direct reports.
- SIMID historical summaries were superseded by the 2026-04-28 calibration review and 2026-05-03 partial prospective review. The paper now includes SIMID only as failed-gate diagnostic evidence.
- Mistral Anchor 3 shows a large JBB alpha curve, but the lack of matched random/layer controls keeps it in an appendix stress-test row.
- CodeRabbit suggested moving this report under `notes/icml/reviews/`; the file remains at `notes/final_icml_submission_integration_report.md` because the user requested that exact output path.
- Truncation wording was narrowed after the evidence audit: the current ground-truth metric layer primarily defends scoring-granularity and evaluator-construct claims for Gemma JailbreakBench. The archived Gemma 256-token run is treated as a superseded measurement-risk note, and the Mistral 2501 row records that its 256-token diagnostic tracks full-output scoring in that stress test.
- Figure 3 now fails fast if the adjudicated bridge IRR summary is missing or has a non-adjudicated status, preventing fallback taxonomy counts from drifting away from the canonical source.

## Unresolved Caveats

- Primary claims remain Gemma-local.
- SAE conclusions are limited to the tested Gemma SAE pool, extraction layers, selectors, and operators.
- TruthfulQA MC remains constrained answer selection; it does not measure open-generation factuality.
- Bridge wrong-entity substitution is behavioral coding only; no localized mechanism claim is made.
- Gemma JailbreakBench specificity remains single-seed.
- Mistral stress tests are failed or incomplete gates; no replication claim is made.
- SIMID is diagnostic-only.

## Remaining Submission Risks

- The PDF is 16 pages including appendices; confirm the workshop appendix/page policy before upload.
- The prose checker intentionally leaves 7 non-failing "steering" warnings in related-work/discussion contexts.
