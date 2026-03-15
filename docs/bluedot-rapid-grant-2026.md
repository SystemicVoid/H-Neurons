# BlueDot Rapid Grant Draft (Compute)

Updated: 2026-03-15

This document turns the current repo state, measured runtimes, and current public pricing into a ready-to-paste BlueDot rapid grant application.

The main recommendation is to ask for **$500**.

Why this number, not more and not less:

- It is large enough to fund a serious replication plus one meaningful extension.
- It is small enough to read as disciplined rather than padded.
- It matches the actual bottleneck in this repo: cloud compute for larger fully GPU-resident runs, not basic feasibility.

## Recommended Ask

**Funding request:** `$500`

**Grant posture:** credible lean

**Scope this funds:**

- finish a clean replication on the already-working small-model pipeline
- validate the method on one larger open model that stays fully GPU-resident
- run one serious extension, preferably intervention experiments first and OOD transfer second

**Scope this does not promise:**

- full six-model paper replication
- any single-GPU 70B run

The 70B exclusion matters. The repo already contains evidence that a single-GPU Llama-70B route is both slow and methodologically wrong for this project, like trying to move a couch through a door it technically fits through only if you scrape both walls. It might move, but it is not a clean experiment.

## Budget

Pricing below was checked on 2026-03-15 from public provider pricing pages:

- Lambda GH200 96GB: `$1.99/hr`
- Runpod H100 PCIe 80GB: `$1.99/hr`
- Runpod H100 NVL 94GB: `$2.59/hr`
- Runpod A100 PCIe 80GB: `$1.19/hr`
- OpenAI `gpt-4o-mini`: `$0.15 / 1M` input tokens and `$0.60 / 1M` output tokens

Recommended budget breakdown:

| Item | Amount | Why |
| --- | ---: | --- |
| Lambda GH200 credits | $250 | About 125 GPU-hours for one validated 24B-scale run plus rerun buffer |
| Runpod credits | $200 | Cheap iteration and ablations on 80GB-class GPUs without burning premium 96GB time |
| OpenAI API credits | $50 | Answer-token extraction and benchmark judging buffer |
| **Total** | **$500** | |

Why split Lambda and Runpod instead of choosing one provider:

- Use **Runpod** where sticker price wins.
- Use **Lambda GH200** where the workflow actually needs a validated ~96GB single-GPU path.
- For this repo, "cheapest per hour" and "cheapest usable experiment hour" are not always the same thing.

## Evidence Behind The Ask

### What is already working

The local Gemma-3-4B replication already completed end-to-end.

From [data/gemma3_4b_pipeline_report.md](../data/gemma3_4b_pipeline_report.md):

- Step 1 response collection: about 4 hours GPU time
- Step 2 answer-token extraction: about 33 minutes and about `$3.50` API cost
- Step 4 activation extraction: about 86 seconds
- Total runtime: about 4.5 hours

That report also documents:

- 3,500 TriviaQA questions processed
- 2,997 answer-token records after filtering
- 38 identified H-neurons
- 77.7% accuracy on the initial split
- 76.5% accuracy on a disjoint split

This matters for the grant because it shows the project is already past the "does this work at all?" stage.

### What has already been built for larger runs

The repo already contains larger-model artifacts and extension scaffolding:

- [data/mistral24b_TriviaQA_consistency_samples.jsonl](../data/mistral24b_TriviaQA_consistency_samples.jsonl)
- [data/mistral24b_answer_tokens.jsonl](../data/mistral24b_answer_tokens.jsonl)
- [models/mistral24b_classifier.pkl](../models/mistral24b_classifier.pkl)
- [scripts/run_intervention.py](../scripts/run_intervention.py)
- [scripts/evaluate_intervention.py](../scripts/evaluate_intervention.py)
- [data/intervention/faitheval/results.json](../data/intervention/faitheval/results.json)

So the main remaining constraint is compute budget and rerun headroom, not missing pipeline code.

### Why 70B is out of scope

From [docs/gh200-research-log-2026-03-15.md](./gh200-research-log-2026-03-15.md):

- the attempted Llama-3.3-70B Step 1 run reached only about `22 / 3500` questions in `49m 38s`
- that implies about `26.6` questions per hour
- at that speed, Step 1 alone would take about `131.6` hours
- at `$2/hr`, that is about `$263` for Step 1 alone

That run was intentionally stopped because it was not scientifically acceptable under the repo's "full precision and fully GPU-resident" rule.

That is the core budgeting lesson: spending less on the wrong run is not frugality, it is noise.

## Form Draft

### How much funding are you requesting?

`$500`

### What have you already done on this project?

I have already completed an end-to-end local replication of the H-Neurons pipeline on Gemma-3-4B: response collection, answer-token extraction, activation extraction, and sparse classifier training. That run processed 3,500 TriviaQA questions, identified 38 candidate H-neurons, and reached 76.5% accuracy on a disjoint test split, which is close to the paper's reported small-model performance. I have also automated the larger-model cloud pipeline, produced Mistral-24B response and answer-token artifacts, trained a 24B classifier artifact, and built intervention scripts plus preliminary benchmark outputs.

### What specifically would this grant fund?

`$250` Lambda GH200 credits for fully GPU-resident 24B-scale replication and extension runs (`$1.99/hr`, about 125 GPU-hours).

`$200` Runpod credits for cheaper iteration, ablations, and reruns on 80GB-class GPUs (`$1.19/hr` A100 80GB or `$1.99/hr` H100 80GB).

`$50` OpenAI API credits for answer-token extraction and benchmark judging.

Pricing checked on 2026-03-15 from public provider pricing pages.

### How does this project reduce catastrophic risk from AI / contribute to AI going well for humanity?

This project studies a mechanistic handle on hallucination and over-compliance: a sparse set of neurons that appear to predict, and potentially causally influence, when a model answers confidently instead of refusing or expressing uncertainty. That matters for catastrophic-risk reduction because advanced models are dangerous when they confidently act on false premises, hide uncertainty, or become easier to steer into unsafe behavior by compliance pressure. Independent replication is especially valuable here because mechanistic claims can sound precise without being robust; extending the work to interventions and OOD evaluation tests whether these neurons are a real control point for safer and more truthful behavior.

### What would you do without this grant?

Without the grant, I would continue with local small-model work and very limited cloud pilots, but I would likely delay or narrow the larger fully GPU-resident replication and the intervention sweep. In practice that would turn the project into a partial verification rather than a strong independent replication of the paper's larger-model claims. The work would still move forward, but more slowly and with less evidential weight.

### What makes you think this project will be successful? Why you? Why now?

The strongest reason to expect success is that the hard part is no longer "can this pipeline be made to work?" but "can I afford enough clean runs to validate and extend it?" I already have working code, a completed local replication, larger-model artifacts, and benchmark/intervention tooling in the repo. I am also already tracking methodological failure modes such as tokenization brittleness, answer-token filtering effects, and the need to keep models fully GPU-resident, which lowers the risk of wasting compute on bad runs. The timing is good because the paper is recent enough that independent replication is still scarce, but the tooling and open models are now good enough for an outsider to do a serious check rather than just comment on the idea from a distance.

## Alternative Ask Sizes

### If you want the smallest credible ask

Ask for **$250**.

Use this if you want to optimize almost entirely for approval odds.

- `$125` Lambda GH200 credits
- `$75` Runpod credits
- `$50` API credits

Tradeoff:

- enough for one narrower replication-plus-extension path
- much less slack for reruns or debugging

### If you want a broader ask

Ask for **$1,000**.

Use this only if you want to explicitly pitch a broader multi-model extension.

- `$500` Lambda GH200 credits
- `$400` Runpod credits
- `$100` API credits

Tradeoff:

- materially more ambitious
- higher chance the request looks large relative to a rapid small grant

## Source Links

- Lambda pricing: <https://lambda.ai/pricing>
- Runpod GPU pricing: <https://www.runpod.io/gpu-pricing>
- OpenAI pricing: <https://platform.openai.com/docs/pricing>
