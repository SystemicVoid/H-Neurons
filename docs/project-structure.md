# Project Structure & Module Organization

Core pipeline code lives in `scripts/` (flat — sibling imports like `from intervene_model import …` require it). Cloud/remote orchestration scripts live in `scripts/infra/`. Put new research utilities in `scripts/` unless they justify a reusable package.

`data/` is organized by model: `data/gemma3_4b/` (primary, local runs) and `data/mistral24b/` (secondary, cloud GPU run). Shared datasets live at `data/benchmarks/` and `data/TriviaQA/`. Original paper reference outputs are in `data/original_paper_examples/`. Commit compact experiment artifacts (JSONL, JSON) to git; keep heavy activations and investigation dumps gitignored (`data/**/activations/`, `data/**/investigation_*/`).

Within `data/gemma3_4b/`, artifacts are grouped by pipeline stage:
- `pipeline/` — TriviaQA probe-training outputs (consistency_samples, answer_tokens, train/test qids, classifier summaries, pipeline_report)
- `probing/bioasq13b_factoid/` — BioASQ OOD probe transfer (samples, answer_tokens, classifier_summary, logs/)
- `intervention/<benchmark>/experiment/` — H-neuron intervention results (alpha_*.jsonl, results.json)
- `intervention/<benchmark>/control/` — negative control results (seed_* dirs, comparison_summary)
- `swing_characterization/` — swing-sample analysis

`site/index.html` is a hand-maintained presentation deck: narrative copy, chart arrays, and intervention numbers are hardcoded. When results change, update both the prose and the embedded JS arrays together or the site drifts.

`data/gemma3_4b/intervention/faitheval_standard/experiment/results.json` only stores raw compliance totals. If you need parse-failure counts, parseable-subset rates, or format-sensitive site exports, derive them from the committed `alpha_*.jsonl` rows instead of assuming `results.json` is sufficient.

Research-output scripts now write adjacent `*.provenance.json` sidecars. When you need to answer "what produced this artifact?", check the sidecar next to the file or in the output directory before inferring from directory names or shell history.
