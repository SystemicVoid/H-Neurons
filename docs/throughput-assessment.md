# Throughput Assessment — Local H-Neuron Intervention Runs

_Created 2026-03-20 by splitting performance analysis out of `docs/tooling-assessment.md`._

## Scope

This note is about **making local intervention runs faster without losing scientific cleanliness**. It covers the jailbreak path first, because that is where we now have the best evidence.

Primary evidence bundle:

- `/tmp/hneurons-scalene/jb-realcli-20260320T112747Z`

Most important artifacts:

- `/tmp/hneurons-scalene/jb-realcli-20260320T112747Z/analysis-metrics.txt`
- `/tmp/hneurons-scalene/jb-realcli-20260320T112747Z/commands.txt`
- `/tmp/hneurons-scalene/jb-realcli-20260320T112747Z/baseline/run.log`
- `/tmp/hneurons-scalene/jb-realcli-20260320T112747Z/baseline/time.txt`
- `/tmp/hneurons-scalene/jb-realcli-20260320T112747Z/baseline/baseline-nvitop.log`
- `/tmp/hneurons-scalene/jb-realcli-20260320T112747Z/profiled/run.log`
- `/tmp/hneurons-scalene/jb-realcli-20260320T112747Z/profiled/profiled-cpu-summary.log`
- `/tmp/hneurons-scalene/jb-realcli-20260320T112747Z/profiled/profiled-cpu-threads.log`
- `/tmp/hneurons-scalene/jb-realcli-20260320T112747Z/profiled/profiled-nvitop.log`

Source files inspected for the current recommendations:

- `scripts/run_intervention.py`
- `AGENTS.md`
- `docs/tooling-assessment.md`

## Executive Read

The real jailbreak path is a **mixed CPU + GPU workload**.

What we know with good confidence:

- The unprofiled real-path baseline kept the GPU busy: mean GPU util `91.05%`, mean GPU power `95.26W`, mean GPU memory `9427.8 MiB`.
- The near-real profiled run preserved that picture only in the early steady-state window: mean GPU util `81.5%`, hot worker CPU mean `98.16%`, about `+14.9%` slowdown by sample `64`.
- Late in the profiled run, Scalene stopped being a useful observer and became part of the problem: by sample `70` slowdown was `+191.9%`, GPU util had collapsed, and the host entered memory/swap thrash.

So the practical decision is:

- **Do not spend more time debugging Scalene for full runs.**
- **Do optimize the real path directly.**

Scalene was useful like a short stress test on a race car: enough to show where the engine and transmission both matter, but not trustworthy as something to leave bolted on for the whole race.

## What the Evidence Says

### Real-path baseline

Command and exact setup:

- `/tmp/hneurons-scalene/jb-realcli-20260320T112747Z/commands.txt`

Observed on the real CLI path in repo env:

- `scripts/run_intervention.py`
- `--benchmark jailbreak`
- `--jailbreak_source jailbreakbench`
- `--n_templates 5`
- `--alphas 3.0`
- `--max_samples 100`

Measured baseline:

- wall time `12:03.34`
- job CPU `100%`
- max RSS `5999812 kB`
- mean GPU util `91.05%`
- mean GPU power `95.26W`
- mean GPU memory `9427.8 MiB`

Evidence:

- `/tmp/hneurons-scalene/jb-realcli-20260320T112747Z/baseline/run.log`
- `/tmp/hneurons-scalene/jb-realcli-20260320T112747Z/baseline/time.txt`
- `/tmp/hneurons-scalene/jb-realcli-20260320T112747Z/baseline/baseline-nvitop.log`

Interpretation:

- The real path is not “GPU mostly waiting around.”
- There is VRAM headroom on a `16 GB` card.

Confidence: High.

### Near-real profiling result

Early steady-state window:

- profiled sample `64` elapsed `532s` vs baseline `463s`
- slowdown `+14.9%`
- mean GPU util `81.5%`
- mean GPU power `86.84W`
- mean GPU memory `9393.9 MiB`
- hot worker CPU mean `98.16%`

Late distortion window:

- profiled sample `69` elapsed `706s` vs baseline `498s`
- profiled sample `70` elapsed `1474s` vs baseline `505s`
- slowdown `+191.9%`
- late GPU util `5.44%`
- late GPU power `24.37W`
- hot worker CPU mean `51.06%`
- repeated `D<l` states in thread logs
- host memory and swap pegged in `nvitop`

Evidence:

- `/tmp/hneurons-scalene/jb-realcli-20260320T112747Z/profiled/run.log`
- `/tmp/hneurons-scalene/jb-realcli-20260320T112747Z/profiled/profiled-cpu-summary.log`
- `/tmp/hneurons-scalene/jb-realcli-20260320T112747Z/profiled/profiled-cpu-threads.log`
- `/tmp/hneurons-scalene/jb-realcli-20260320T112747Z/profiled/profiled-nvitop.log`
- `/tmp/hneurons-scalene/jb-realcli-20260320T112747Z/analysis-metrics.txt`

Interpretation:

- Scalene is usable only for a short post-load window on this path.
- The closer-to-real run overturns the earlier “GPU clearly not the limiter” stance.
- The run is mixed enough that both a hot CPU core and a heavily used GPU matter.

Confidence: High for the workload classification. Medium for any finer Scalene attribution claims.

### Code-path implications

The hot loop in `scripts/run_intervention.py` is simple:

- `tokenizer.apply_chat_template(...)`
- `unwrap_chat_template_output(...).to(model.device)`
- `model.generate(...)`
- `tokenizer.decode(...)`

For jailbreak, this runs one sample at a time in `run_jailbreak()`.

The intervention is implemented with Python forward pre-hooks in `HNeuronScaler`, one callback per hooked layer, mutating `x[:, :, idx]` inside the callback.

Why this matters:

- It is a plausible explanation for the “one hot core” symptom.
- It is also a plausible source of CPU-side gaps and orchestration tax between GPU kernels.

Evidence:

- `scripts/run_intervention.py`

Confidence: Medium-high. This is a reasoned code-path inference, not a direct timing breakdown.

## Ranked Throughput Paths

Ranked by expected throughput upside on the current local jailbreak path, not by implementation ease.

### 1. Reduce Python hook overhead by redesigning the intervention path

Potential throughput upside: High.

Scientific safety: Medium-high if the numerical operation stays identical.

Why it ranks first:

- The current intervention happens through Python forward pre-hooks across `23` layers and `38` neurons in the real run.
- That is exactly the sort of repeated Python callback cost that can pin one core while also fragmenting GPU work submission.
- Unlike tokenization, this overhead happens during generation itself, so it can matter every decode step.

Evidence:

- `scripts/run_intervention.py`
- `/tmp/hneurons-scalene/jb-realcli-20260320T112747Z/profiled/profiled-cpu-summary.log`
- `/tmp/hneurons-scalene/jb-realcli-20260320T112747Z/analysis-metrics.txt`

Uncertainty:

- We have not yet measured how much of the hot-core cost is hook callback overhead versus tokenization, decode, or HF generate orchestration.

Confidence: Medium.

How to ground it:

- Add low-overhead timers around hookless vs hooked runs on a fixed 10-20 sample slice.
- Compare three variants:
  - current pre-hook implementation
  - no-op hook scaffold
  - semantically equivalent module-wrapper or in-forward implementation
- Use wall time plus CUDA events around `model.generate()`.

### 2. Batch multiple samples per `model.generate()` call, within one alpha value

Potential throughput upside: High.

Scientific safety: Medium.

Why it ranks second:

- Real GPU memory stayed around `9.4 GB`, leaving real headroom on a `16 GB` card.
- One-at-a-time generation is the current structure in `run_jailbreak()`.
- Batching can amortize host overhead and keep the GPU fed more efficiently.

Evidence:

- `/tmp/hneurons-scalene/jb-realcli-20260320T112747Z/baseline/baseline-nvitop.log`
- `/tmp/hneurons-scalene/jb-realcli-20260320T112747Z/analysis-metrics.txt`
- `scripts/run_intervention.py`

Why not rank it first:

- Sampled generation plus variable prompt lengths make batching less “free” than it sounds.
- Padding and decode divergence can reduce gains.
- It is a bigger behavior-shape change than caching tokenization.

Confidence: Medium-high that some gain exists. Medium on magnitude.

How to ground it:

- Benchmark batch sizes `2`, `4`, `8` on a fixed 20-sample slice for one alpha.
- Bucket prompts by length or template to reduce padding waste.
- Compare:
  - samples/sec
  - mean GPU util
  - VRAM
  - response-format parity checks on a held-out sample set

### 3. Pre-tokenize prompts and cache chat-template outputs across alpha sweeps

Potential throughput upside: Medium.

Scientific safety: High.

Why it ranks third:

- Jailbreak prompts are deterministic and reused across alphas.
- Current code tokenizes in every `generate_response()` call.
- This is behavior-preserving: same text in, same token IDs out.

Evidence:

- `scripts/run_intervention.py`
- `/tmp/hneurons-scalene/jb-realcli-20260320T112747Z/profiled/profiled-cpu-summary.log`

Why not higher:

- The GPU is already heavily used, so tokenization alone is unlikely to be the whole answer.
- Hook overhead may dominate the hot-core symptom.

Confidence: High that it is safe. Medium that the payoff is material.

How to ground it:

- Time prompt preparation separately from `model.generate()`.
- Compare cached-token vs uncached-token runs on identical prompts for one alpha and for a multi-alpha sweep.
- Record share of wall time saved.

### 4. Length-bucketed batching

Potential throughput upside: Medium.

Scientific safety: Medium.

Why it is distinct from plain batching:

- Plain batching can lose efficiency if one long sequence drags shorter ones with it.
- Grouping similar prompt lengths is a standard way to recover batch efficiency without changing semantics much.

Evidence:

- `scripts/run_intervention.py`
- the workload uses variable prompt lengths across jailbreak templates and goals

Confidence: Medium.

How to ground it:

- Compute token-length histogram for the 500 jailbreak prompts.
- Compare naive batch order vs length-bucketed batch order on the same slice.

### 5. Use `torch.inference_mode()` in generation

Potential throughput upside: Low to medium.

Scientific safety: High.

Why it is worth noting:

- The current code uses `torch.no_grad()`.
- `torch.inference_mode()` can remove some autograd bookkeeping more aggressively.

Evidence:

- `scripts/run_intervention.py`

Uncertainty:

- The real gain may be small relative to decode and hook overhead.

Confidence: Medium-high that it is safe. Low-medium on payoff.

How to ground it:

- A/B benchmark on the same 20-sample slice with everything else fixed.

### 6. Overlap CPU prep for sample `n+1` with GPU generation for sample `n`

Potential throughput upside: Low to medium.

Scientific safety: Medium-high.

Why it is interesting:

- The system appears mixed rather than purely GPU-bound.
- If tokenization and prompt prep remain material after hook cleanup, overlap can hide some host cost.

Why it ranks lower:

- It adds complexity.
- The current code is very simple and serial.
- It may not help much if the main tax is already inside `model.generate()` due to hooks.

Confidence: Low-medium.

How to ground it:

- First measure prompt-prep cost directly.
- Only prototype this if prompt prep is still a meaningful wall-time share after simpler changes.

### 7. Lower `max_new_tokens`, but only for short-answer benchmarks

Potential throughput upside: Benchmark-specific medium; for jailbreak low or invalid.

Scientific safety: High for `faitheval_standard`, low for jailbreak.

Why the ranking is low for the current question:

- On jailbreak, this would change the task rather than optimize the same task.
- The response-length audit already shows jailbreak and falseqa regularly hit the `256` cap.

Evidence:

- previous audit captured in the old tooling note split from this file

Confidence: High.

How to ground it:

- Keep a benchmark-specific token-length audit artifact.
- Only lower caps where the empirical max is comfortably below the proposed new cap.

### 8. `torch.compile()`

Potential throughput upside: Very uncertain, possibly medium.

Scientific safety: Medium-low until validated.

Why it is not a first move:

- Dynamic generation plus Python hooks is an awkward target for compiler wins.
- If hooks stay as Python callbacks, compile may buy little or create instability.

Confidence: Low.

How to ground it:

- Revisit only after hook-path cleanup.
- Run a small equivalence suite and timing suite on a bounded slice.

### 9. Quantization

Potential throughput upside: Medium in some setups, but uncertain here.

Scientific safety: Low for canonical comparability.

Why it ranks last:

- You are not currently starved for VRAM.
- Quantization changes the numerical regime of the activations you are explicitly intervening on.

Confidence: Medium that it can help speed in general. High that it weakens comparability.

How to ground it:

- Treat it as exploratory only.
- Compare bf16 vs quantized on a small slice and inspect both throughput and intervention-behavior drift.

## Measurement Plan That Is Worth Doing

If the goal is faster research throughput, spend engineering time on this instead of more Scalene debugging:

1. Add in-process timers to the real path.
2. Time these segments separately:
   - prompt construction
   - chat templating / tokenization
   - host-to-device transfer
   - `model.generate()`
   - decode
   - file write
3. Add a flag to benchmark a fixed slice cleanly, such as 20 samples on one alpha.
4. Run targeted A/B tests for:
   - hook implementation variants
   - cached tokenization
   - batch size and length bucketing
   - `inference_mode()`

This is the highest-signal next measurement loop because it measures the real pipeline without letting the profiler become the workload.

## What Not to Chase First

- More full-run Scalene debugging
- Disk I/O optimization
- Data-loading optimization
- Quantization for canonical runs
- Blanket lowering of `max_new_tokens`

## Current Recommendation

The current recommendation holds in spirit, but it should be sharpened.

Old framing:

- “Look for CPU/setup gaps and maybe batch.”

Better framing now:

- “Optimize a mixed workload where the most suspicious avoidable tax is Python-side intervention overhead, then batch to amortize the remaining host cost.”

That is the highest-confidence path to faster local research throughput while keeping the experiment scientifically recognizable.
