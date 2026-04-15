# Detection Is Not Enough: Gemma 3 4B Intervention Audit Workspace

This repository began as a fork of THUNLP's H-Neurons project and still retains the paper-faithful H-neuron pipeline, original reference materials, and example artifacts. Its current center of gravity is broader: a single-model Gemma-3-4B-IT case study on when strong internal readouts do and do not become useful steering targets. The working framework throughout the repo is `measurement`, `localization`, `control`, and `externality`.

Primary current narratives:

- Site: [`site/`](site/)
- Current paper draft: [`paper/draft/full_paper.md`](paper/draft/full_paper.md)
- Measurement contract: [`notes/measurement-blueprint.md`](notes/measurement-blueprint.md)

## Overview

This is a research workspace for mechanistic intervention experiments in `google/gemma-3-4b-it`. It contains the original H-neuron replication funnel, intervention and negative-control runs across multiple benchmarks, evaluator and reporting audits, paper assembly tooling, and site-export paths for presenting current results. H-neurons remain the paper-faithful baseline, but not the whole thesis.

## Current Status

- H-neuron replication remains the first anchor: the paper-faithful sparse detector keeps a held-out signal on the clean split (`76.5%` accuracy, `95% CI [73.6, 79.5]`) and the committed FaithEval intervention path still shows a no-op-to-max compliance gain (`+4.5 pp`, `95% CI [2.9, 6.1]`). See [`data/gemma3_4b/pipeline/pipeline_report.md`](data/gemma3_4b/pipeline/pipeline_report.md) and [`data/gemma3_4b/intervention_findings.md`](data/gemma3_4b/intervention_findings.md).
- Matched readout quality did not guarantee useful steering: in the current FaithEval comparison, SAE features matched H-neuron detection quality within uncertainty but failed to produce a useful control signal under the committed SAE intervention setup. The broader project claim is therefore not "detectors are fake," but "good detectors are not automatically good levers."
- Transfer and evaluation remain fragile: the ITI bridge path improves some constrained answer-selection surfaces while hurting open-ended factual generation on the locked TriviaQA bridge test, and the jailbreak work is now partly a measurement case study where evaluator choice changes the conclusion. The current public framing lives in [`site/story.html`](site/story.html) and [`site/methods.html`](site/methods.html).

## Quick Start

This repo uses `uv` for Python environments and repo-local execution.

```bash
uv sync --dev
uv run pytest
ruff check scripts tests
ty check
```

Practical prerequisites:

- Many workflows require local access to a target model and benchmark data.
- Judge-based evaluation paths require API credentials in the environment or `.env`.
- Large artifacts already live under `data/`; this repo is not organized around a single end-to-end bootstrap command.

## Core Workflows

Paper-faithful H-neuron pipeline:

- `scripts/collect_responses.py` collects consistent TriviaQA-style generations.
- `scripts/extract_answer_tokens.py` tags answer spans for the original CETT-style pipeline.
- `scripts/sample_balanced_ids.py` creates train/test manifests.
- `scripts/extract_activations.py` materializes activation features.
- `scripts/classifier.py` trains the sparse detector used by the H-neuron baseline.

Intervention and evaluation:

- `scripts/run_intervention.py` is the main experiment entrypoint for neuron, SAE, direction, and ITI-head interventions.
- `scripts/evaluate_intervention.py` runs GPT-judge evaluation for supported benchmark families.
- `scripts/run_negative_control.py` executes specificity checks rather than relying on headline effects alone.

Canonical shell wrappers:

- `scripts/infra/` contains the runnable wrappers used for claim-bearing experiments.
- Representative entrypoints include the ITI sweep and evaluation wrappers, TriviaQA bridge wrappers, the canonical `5000`-token jailbreak pipeline, and the D7 pilot and full-500 scripts.

Paper and site outputs:

- `uv run python scripts/build_full_paper.py` assembles the current paper draft.
- `uv run python scripts/export_site_data.py` exports site-facing JSON summaries from committed outputs.
- `scripts/infra/publish.sh site --slug aware-fresco-4a2q --client amp` publishes the site to its canonical URL.

If you need the original forked-paper examples, start with [`data/original_paper_examples/`](data/original_paper_examples/). If you need the current project argument, start with [`site/index.html`](site/index.html) or [`paper/draft/full_paper.md`](paper/draft/full_paper.md).

## Repository Map

- [`scripts/`](scripts/) - experiment entrypoints, analysis scripts, and shared utilities
- [`scripts/infra/`](scripts/infra/) - canonical orchestration wrappers for larger runs
- [`data/`](data/) - committed experiment outputs, semantic run directories, controls, and audits
- [`notes/`](notes/) - planning documents, measurement contracts, and sprint notes
- [`paper/`](paper/) - source shards, reviews, citations, and assembled draft for the current paper
- [`paper/draft/`](paper/draft/) - source shards and assembled draft for the current paper
- [`site/`](site/) - public presentation site and consumers of exported result payloads
- [`tests/`](tests/) - regression coverage for pipeline guards, reporting, evaluation, and exports
- [`papers/`](papers/) - local paper corpus, H-neurons reference materials, and literature notes

For the current committed data layout, also see [`docs/archive/project-structure.md`](docs/archive/project-structure.md).

## Measurement and Reproducibility

This repo treats evaluation as part of the scientific object, not an afterthought.

- Quantitative claims in project-facing materials are expected to include uncertainty.
- Claim-relevant artifacts should carry adjacent `*.provenance.json` sidecars.
- Negative controls are expected for new intervention claims.
- Evaluator choice can materially change conclusions, especially on jailbreak-style safety runs.
- `notes/runs_to_analyse.md` is part of the run lifecycle, not optional bookkeeping.

The canonical contract for Act 3 claims lives in [`notes/measurement-blueprint.md`](notes/measurement-blueprint.md). Quantitative reporting coverage is tracked in [`docs/ci_manifest.json`](docs/ci_manifest.json).

## Lineage and Further Reading

This repo still contains the upstream H-neurons lineage:

- Upstream framing and paper materials: [`papers/h-neurons-hallucination-correlated.md`](papers/h-neurons-hallucination-correlated.md)
- Original converted paper text: [`papers/original-paper-markdown-converted.md`](papers/original-paper-markdown-converted.md)
- Original paper-faithful example outputs: [`data/original_paper_examples/`](data/original_paper_examples/)

Current project-facing reading order:

- [`site/index.html`](site/index.html)
- [`site/story.html`](site/story.html)
- [`site/methods.html`](site/methods.html)
- [`data/gemma3_4b/pipeline/pipeline_report.md`](data/gemma3_4b/pipeline/pipeline_report.md)
- [`data/gemma3_4b/intervention_findings.md`](data/gemma3_4b/intervention_findings.md)
- [`paper/draft/full_paper.md`](paper/draft/full_paper.md)
