# Throughput Assessment — Local H-Neuron Intervention Runs

_Created 2026-03-20 by splitting performance analysis out of `docs/tooling-assessment.md`._
_Updated 2026-03-23 with deeper workload characterisation and concrete intervention alternatives._
_Updated 2026-03-24 with empirical results from canonical 5000-token run and sensor gap analysis._

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

## Executive Summary

**The GPU is doing real work, and one CPU core is doing suspicious amounts of paperwork.**

`max_new_tokens=1024` is now the dominant unavoidable cost; hook overhead is still the dominant avoidable cost. The reason hook overhead still matters is brutal and simple: you now pay it on ~4× more decode steps.

The real jailbreak path is a **mixed CPU + GPU workload**:

- The unprofiled real-path baseline kept the GPU busy: mean GPU util `91.05%`, mean GPU power `95.26W`, mean GPU memory `9427.8 MiB`.
- The near-real profiled run preserved that picture only in the early steady-state window: mean GPU util `81.5%`, hot worker CPU mean `98.16%`, about `+14.9%` slowdown by sample `64`.
- Late in the profiled run, Scalene stopped being a useful observer and became part of the problem: by sample `70` slowdown was `+191.9%`, GPU util had collapsed, and the host entered memory/swap thrash.

Practical decision:

- **Do not spend more time debugging Scalene for full runs.**
- **Do optimize the real path directly.**

GPU is continuously busy, but not expensively busy — single-stream long decode with too much host-side fuss around it.

## What the Evidence Says

### Real-path baseline

Command and exact setup: `/tmp/hneurons-scalene/jb-realcli-20260320T112747Z/commands.txt`

Observed on the real CLI path in repo env:

- `scripts/run_intervention.py`
- `--benchmark jailbreak --jailbreak_source jailbreakbench --n_templates 5 --alphas 3.0 --max_samples 100`

Measured baseline:

- wall time `12:03.34`
- job CPU `100%`
- max RSS `5999812 kB`
- mean GPU util `91.05%`
- mean GPU power `95.26W`
- mean GPU memory `9427.8 MiB`

Interpretation:

- The real path is not "GPU mostly waiting around."
- There is VRAM headroom on a `16 GB` card.

Confidence: High.

### Near-real profiling result

- profiled sample `64` elapsed `532s` vs baseline `463s`
- slowdown `+14.9%`
- mean GPU util `81.5%`
- mean GPU power `86.84W`
- mean GPU memory `9393.9 MiB`
- hot worker CPU mean `98.16%`

Interpretation:

- Scalene is usable only for a short post-load window on this path.
- The closer-to-real run overturns the earlier "GPU clearly not the limiter" stance.
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

- The hook fires **once per hooked layer per forward** — that is **23 Python callbacks per decode step**, plus prefill.
- This is exactly the kind of "death by a thousand tiny Python paper cuts" that keeps one host thread pinned while the GPU remains mostly busy.
- It is also a plausible source of CPU-side gaps and orchestration tax between GPU kernels.

Confidence: Medium-high. This is a reasoned code-path inference, not a direct timing breakdown.

### W&B GPU monitoring observations

Periodic dips in SM clock + memory-access % are visible in the W&B jailbreak long-responses run. These dips are worth correlating with sample boundaries:

- If roughly once per sample → evidence of host-side orchestration gaps (end of generation → decode → postprocess/write → next prompt prep → next prefill launch).
- If many times within a sample → decode-phase variability, termination checks, allocator/scheduler behaviour, or hook-related per-step fragmentation.

Worth correlating against: sample completion timestamps, generated token counts, prompt lengths, time in `model.generate()`, time outside `model.generate()`.

## Estimated Impact

| Optimisation | Expected gain | Confidence |
|---|---|---|
| Hook-path cleanup | 10–25% | Medium |
| Tokenization caching | Low single-digit % | High |
| `inference_mode()` | 0–3% | Medium-high |
| Batching (size 2–4) | Potentially larger than all above | Medium (risky for sampled jailbreak) |

More than 30% from hook cleanup alone would be surprising. The 4× longer decode (`max_new_tokens=1024`) is the floor — you are optimising the tax, not the floor.

## Ranked Throughput Paths

### Speed only (current jailbreak run)

1. Redesign/remove the Python intervention path
2. Try small-batch generation
3. Cache tokenization/chat-template outputs
4. `inference_mode()`
5. Everything else

### Speed × scientific cleanliness (canonical jailbreak run)

1. Intervention-path cleanup
2. Tokenization caching
3. `inference_mode()`
4. Batching — only after validation

## Intervention Architecture Alternatives

### A. Wrapped `down_proj` forward — best cleanliness-first option

Replace the `down_proj` modules with a small wrapper that stores target indices or a precomputed scale vector. On forward: if `alpha == 1.0`, call through; otherwise scale the **input activation** before `F.linear`.

A particularly clean version: precompute a full `scale_vector` of ones, set target dims to `alpha`, do `x = x * scale_vector`. This avoids fancy indexed assign/gather/scatter inside the hot path.

The current line `x[:, :, idx] = x[:, :, idx] * self._alpha` likely creates extra indexed gather/scatter work. A dense multiply by a prebuilt scale vector is often cleaner and may actually be faster, even though it touches more elements.

Why this ranks highest for conservative use:

- Preserves **activation-side** intervention semantics
- Removes `register_forward_pre_hook` callback overhead
- Keeps the code auditable
- Should be much closer to current numerical behaviour than weight scaling
- Works with the existing "scaler has `.alpha`" interface pattern

### B. Direct weight-column scaling — highest upside, validate carefully

Because `down_proj` is linear, scaling selected input neurons is mathematically equivalent to scaling the corresponding weight columns. For a fixed alpha over a whole run: restore original target columns, multiply target columns by `alpha`, then restore originals after the run.

This removes per-token intervention overhead almost entirely.

**Catch:** mathematically equivalent, but not guaranteed bitwise identical in `bf16`. Scaling the activation and scaling the weight are equivalent in exact arithmetic, but not necessarily numerically identical in `bfloat16` because the rounding happens on a different operand.

This does not mean it is bad. It means: **validate before calling it canonical**.

- **Best fast path:** weight-column scaling
- **Best conservative path:** wrapped forward with activation-side scaling

### C. "Fused single hook" — not worth it

The current design already has one hook per affected module, not one per neuron. The big cost is the hook mechanism firing across layers every step, not the number of closures.

### D. Custom C++ / Triton / autograd-free op — overkill

That is sports-car nonsense when the bicycle has a flat tire.

## Batching Feasibility for `run_jailbreak()`

### Technically feasible?

Yes.

### Safe in the strictest scientific sense?

Not for exact equivalence. That is the key distinction.

### Main risks

**Left padding / position handling:** For batched decoder-only generation, you need left padding (`tokenizer.padding_side = "left"` with a valid `pad_token_id`). Usually fine, but validate for Gemma chat formatting.

**Variable output lengths:** With `max_new_tokens=1024`, one long sample can drag shorter batch-mates around. Gains depend on both prompt-length spread and generated-length spread. Jailbreak responses are exactly the kind of thing that diverge a lot.

**Sampling non-equivalence:** With `do_sample=True`, batching changes RNG consumption and sometimes kernel shape / floating-point order. Even if logits are "basically the same," sampled continuations will not be exact serial replicas. Batched sampled generation is **not** a pure implementation refactor — it is a slightly different experimental environment.

**Hooks + batching:** The intervention itself is row-wise fine (`x[:, :, idx] *= alpha` applies independently per batch row). The intervention math does not break under batching. The **sampling trajectory** is what changes.

### Recommendation

For **canonical jailbreak runs**: do not make batching your first move.

For **deterministic benchmarks** (`do_sample=False`): batching is much cleaner — test it earlier. Batch `faitheval`, `falseqa`, `bioasq` first as they are much easier to validate.

For jailbreak, only adopt batching after defining success as **statistically indistinguishable judged outcomes**, not exact response equivalence. On a 16 GB card with 1024-token decode: test **batch size 2** first, maybe **4** if VRAM allows, **8** feels optimistic with KV cache growth.

## Tokenization Caching Across Alphas — ✅ Done

Implemented in `65dd1bd`. CPU-side tensors cached once per sample before the alpha
sweep, passed via `prompt_cache` dict. Sycophancy caches turn 1 only (turn 2 depends
on model response).

## `torch.inference_mode()` vs `torch.no_grad()`

Probably safe, but smoke-test it. The hooks do in-place mutation of inference tensors, which is usually okay inside `inference_mode()`, but because you are mutating tensors in hooks, do not flip this blind on a long run without a short parity test.

Expected speedup: tiny, maybe 0–3%, maybe nothing measurable. Good hygiene, not salvation.

## What the Original Assessment Missed

### 1. `alpha=1.0` bypass — ✅ Partial

`HNeuronScaler` hook now short-circuits with `if self._alpha == 1.0: return args`
(avoids indexed write). Full hook removal for `alpha=1.0` runs is still an open
improvement — the hooks still fire 23× per decode step, just returning early.

### 2. Verify stop behaviour under 1024

After increasing the cap, check the output-length histogram. If too many jailbreak responses land exactly at 1024, either the model genuinely wants to ramble forever, or the stop configuration is not doing what you think. Fixing a stop-token/config issue is a **scientifically clean** throughput win.

### 3. Current indexed assignment is a bad micro-kernel choice

`x[:, :, idx] = x[:, :, idx] * self._alpha` probably involves indexed gather → multiply → indexed scatter/writeback. A precomputed dense scale vector (`x = x * scale_vector`) gives the runtime a much simpler memory access pattern.

## Measurement Plan

If the goal is faster research throughput, spend engineering time on this instead of more Scalene debugging:

### Per-sample instrumentation — ✅ Done

`generate_response()` returns a `timings` dict per sample: `template_s`, `h2d_s`,
`generate_s`, `decode_s`, `total_s`, `prompt_tokens`, `generated_tokens`,
`hit_token_cap`. Implemented in `65dd1bd`.

### Per-alpha instrumentation

- average generated length
- tokens/sec
- samples/sec

### Targeted A/B comparisons (20–50 sample slice)

- current hook path
- no-op / `alpha=1.0` with hooks fully removed
- refactored non-hook intervention (wrapped forward or weight scaling)
- ~~cached tokenization~~ ✅ done
- `inference_mode()`

## Minimal Validation Plan

Before trusting any intervention refactor:

1. **Deterministic parity check:** one prompt, one forward/generate step. Compare logits between current hooks, wrapped forward, and weight scaling.
2. **Short deterministic benchmark check:** `do_sample=False`, exact output match rate.
3. **Short jailbreak check:** 50–100 samples, same seed policy. Compare judged compliance rates. Require CI on the difference to be near zero before promoting.

Don't assume the fast thing is faithful — force it to earn the right.

## What Not to Chase First

- More full-run Scalene debugging
- Disk I/O optimisation
- Data-loading optimisation
- Quantization for canonical runs
- Blanket lowering of `max_new_tokens`
- `torch.compile()` before hook-path cleanup
- Custom C++/Triton kernels

## Concrete Recommendation

### Tier 1 — ✅ Done

1. ~~**Bypass intervention completely for `alpha=1.0`**~~ — partial: hook short-circuits but still fires ([see §1 above](#1-alpha10-bypass--✅-partial))
2. ~~**Add cheap timers**~~ — `generate_response()` returns per-sample timings dict (`65dd1bd`)
3. ~~**Cache tokenized prompts across alphas**~~ — CPU-side cache, passed via `prompt_cache` (`65dd1bd`)

### Tier 2 — highest-value real optimisation

4. Replace hook-based neuron intervention with either:
   - **wrapped `down_proj` forward + activation-side scale vector** if you want the cleanest parity story
   - **weight-column scaling** if you want max speed and are willing to validate "statistically indistinguishable" rather than assume exactness

### Tier 3 — only after that

5. Try **batch size 2** on a fixed jailbreak slice
6. Keep batching off the canonical run unless judged outcomes look stable

**Bottom line:** the bottleneck is not "the GPU is too small." The bottleneck is that the experiment is paying a Python tax on every token, every layer, for a transformation that is structurally much simpler than the machinery used to express it. Fix the representation of the intervention first; then revisit batching.


## Empirical Results — Canonical 5000-Token Run (2026-03-24)

Data source: `data/gemma3_4b/intervention/jailbreak/experiment/alpha_0.0.jsonl` (327/500 samples at time of analysis, run still in progress).

Run config: `--benchmark jailbreak --alphas 0.0 1.5 3.0 --max_new_tokens 5000 --seed 42 --wandb`

### Per-sample timing (alpha=0.0, n=327)

| Metric | Value |
|---|---|
| `generate_s` mean | 36.0 s |
| `generate_s` median | 37.2 s |
| `generate_s` min / max | 11.4 s / 48.8 s |
| `generate_s` stdev | 6.1 s |
| `generate_s` p10 / p90 | 27.7 s / 42.2 s |
| `generated_tokens` mean | 1226 |
| `generated_tokens` median | 1265 |
| `generated_tokens` min / max | 410 / 1618 |
| `prompt_tokens` mean | 58 |
| `hit_token_cap` | **0 / 327 (0%)** |

### Token length distribution

```
[   0,  500):   1  ( 0.3%)
[ 500,  800):  11  ( 3.3%)
[ 800, 1000):  31  ( 9.4%)
[1000, 1200):  70  (21.3%)
[1200, 1400): 177  (53.8%)   <- majority
[1400, 1600):  37  (11.2%)
[1600, 2000):   2  ( 0.6%)
[2000, 5000):   0  ( 0.0%)
```

**The 5000-token cap has ~3x headroom over actual usage.** No samples are being truncated. A cap of 2000-2500 would be safe for future runs without changing any results, though it does not save wall time since the model stops naturally.

Contrast with the 1024-cap smoke test: **70-80% truncation** across all alphas. The earlier runs were systematically underestimating natural response length.

### Throughput stability

| Window | tok/s mean | tok/s stdev |
|---|---|---|
| First 50 samples | 34.0 | 0.78 |
| Last 50 samples | 34.1 | 0.73 |
| Drift | **+0.04** | -- |

**Zero drift** over 327 samples. No thermal throttling, no memory fragmentation, no progressive slowdown. The system is in steady state.

### Time breakdown

| Component | Total across 327 samples |
|---|---|
| `generate_s` | 11,776 s (100.00%) |
| `template_s` | 0.000 s |
| `h2d_s` | 0.031 s |
| `decode_s` | 0.165 s |

`model.generate()` is the entire cost. The host-side orchestration gaps the earlier Scalene analysis worried about are **invisible in practice** after the tokenization caching fix. The overhead is not zero -- it just rounds to zero compared to 36 s of generation per sample.

### Correlation: token count vs wall time

Pearson r = **0.9953**. Time is almost perfectly linear in generated tokens.

Implied cost: **29.4 ms/token**.

| Quartile | mean generate_s | mean tokens | tok/s |
|---|---|---|---|
| Shortest 25% responses | 27.5 s | 965 | 35.2 |
| Longest 25% responses | 42.4 s | 1419 | 33.5 |

The ~5% slowdown on long sequences is consistent with attention cost scaling with KV cache length. Hook overhead, if significant, would show as a constant per-token tax, not a length-dependent one.

### Projection

- Average per sample: 36.0 s
- **Estimated total for 3 alphas x 500 samples: ~15 hours**

### Smoke test cross-reference (1024-token cap, n=10 per alpha)

| Alpha | hit_cap | mean gen_tok | tok/s |
|---|---|---|---|
| 0.0 | 70% | 994 | 34.9 |
| 1.0 | 80% | 954 | 35.1 |
| 2.0 | 70% | 949 | 35.1 |
| 3.0 | 70% | 888 | 35.1 |

tok/s is consistent across caps and alphas -- the decode engine runs at the same rate regardless of intervention strength. This is expected: the hooks do negligible work per step relative to the GPU decode kernel.

### What the live data overturns

1. **"Hook overhead is the dominant avoidable cost"** -- still plausible in theory, but the per-sample data shows `model.generate()` accounts for 100% of measured time. The hook overhead is buried inside `generate_s` and is not separable with current sensors. It may be real but it is not large enough to create visible inter-sample gaps.

2. **"The GPU is not the limiter"** -- the Scalene-era conclusion was always shaky. The live data is more consistent with "GPU is the limiter, the rest rounds to zero." This does not mean hooks are free -- just that they are not the thing to chase before measuring them properly.

3. **"Tokenization/orchestration overhead matters"** -- after the caching fix, it does not. `template_s` is literally zero for cached prompts.

## Sensor Gap Analysis

### What we measure well (per-sample, stored in JSONL)

- `template_s`, `h2d_s`, `generate_s`, `decode_s`, `total_s`
- `prompt_tokens`, `generated_tokens`, `hit_token_cap`

### What we cannot measure and need

| Gap | Why it matters | Fix cost |
|---|---|---|
| **Inter-sample wall-clock gap** | Time between sample N finishing and sample N+1's `generate()` starting -- includes JSONL write, tqdm update, W&B overhead, GC, any allocator work. Currently invisible because `total_s` only covers inside `generate_response()`. | Cheap: add `time.time()` at loop iteration start/end, store as `wall_start_ts` / `wall_end_ts`. |
| **Per-alpha aggregate timing** | No per-alpha wall clock, tokens/sec, or samples/sec is computed or logged. The doc called for this but it was never implemented. | Cheap: accumulate in the alpha loop, log to W&B and summary. |
| **Real-time W&B streaming** | Per-sample timings go to JSONL but not to W&B during the run. Cannot monitor throughput live except by parsing JSONL externally. | Cheap: `wandb.log()` per sample with `generate_s`, `generated_tokens`, tok/s. |
| **Hook overhead per decode step** | The doc identifies 23 Python callbacks per decode step as the main avoidable cost, but there is no measurement. Cannot separate hook time from pure GPU decode time inside `generate_s`. | Medium: add cumulative timer inside `HNeuronScaler.__call__`, report total hook time per sample. |
| **GPU utilization per sample** | Only available from external `nvitop` snapshots, not correlated with individual samples. | Medium: `pynvml` query at sample boundaries, or correlate nvitop log timestamps with sample timestamps (requires wall-clock timestamps above). |
| **Memory allocator behaviour** | No visibility into CUDA allocator fragmentation, cache flushes, or GC pauses over time. | Low priority given zero drift observed. |

### Measurement status vs. the original plan

| Planned sensor | Status |
|---|---|
| Per-sample prompt token count | Done -- in JSONL |
| Per-sample generated token count | Done -- in JSONL |
| Per-sample time in `model.generate()` | Done -- `generate_s` |
| Per-sample time outside `model.generate()` | **Missing** -- need wall-clock timestamps |
| Per-alpha average generated length | **Missing** -- not computed |
| Per-alpha tokens/sec | **Missing** -- not computed |
| Per-alpha samples/sec | **Missing** -- not computed |
| A/B: current hooks vs no-op | **Not run** |
| A/B: current hooks vs refactored intervention | **Not run** |

## Revised Estimated Impact (post-empirical)

| Optimisation | Expected gain | Confidence | Notes |
|---|---|---|---|
| Hook-path cleanup | **Unknown -- need measurement first** | Low (was Medium) | Cannot attribute any portion of `generate_s` to hooks without per-step timing. |
| ~~Tokenization caching~~ | ~~Done~~ | -- | Measured effect: `template_s` went from nonzero to 0. |
| `inference_mode()` | 0-3% | Medium-high | Still untested. |
| Batching (size 2-4) | Potentially 30-80% for deterministic benchmarks | Medium | Jailbreak risk unchanged. |
| **Lower `max_new_tokens` to 2000-2500** | **0% wall-time gain** | High | Model stops naturally at ~1200-1600. Cap is not the bottleneck. |
| **Add missing sensors** | **0% direct gain, unblocks all other decisions** | High | Cannot prioritise any optimisation without measuring it first. |

## Revised Ranked Next Steps (by ROI)

### Tier 1 -- Instrument before optimising

These cost minutes of engineering and unblock every decision below.

1. **Add wall-clock timestamps per sample** -- `time.time()` at loop iteration start/end, stored alongside `timings`. Makes inter-sample gaps and per-alpha aggregates computable. Unblocks correlation with W&B GPU dips.

2. **Add per-alpha summary** -- after each alpha completes, compute and log: wall time, samples/sec, mean tok/s, mean generated tokens. Stream to W&B if `--wandb`.

3. **Stream per-sample metrics to W&B** -- `wandb.log()` with `generate_s`, `generated_tokens`, tok/s per sample. Enables live monitoring and post-hoc correlation with GPU traces.

### Tier 2 -- Measure the hook tax

Cannot justify Tier 3 without this data.

4. **Add cumulative hook timer in `HNeuronScaler`** -- `time.perf_counter()` around the hook body, accumulate total, report per sample. This finally answers "how much of `generate_s` is Python hook overhead?"

5. **Run the A/B comparison** -- 50-sample slice, same seed: (a) current hooks, (b) hooks fully removed (`alpha=1.0`, no hooks installed), (c) wrapped `down_proj` forward. Compare tok/s. This is the decisive experiment for whether hook-path cleanup is worth doing.

### Tier 3 -- Optimise based on data

6. **If hook tax > 5%**: implement wrapped `down_proj` forward with precomputed scale vector (activation-side, cleanest parity story).

7. **If hook tax < 5%**: skip intervention refactor entirely; move to batching on deterministic benchmarks (`faitheval`, `falseqa`, `bioasq`).

8. **Batching on jailbreak**: only after deterministic benchmarks validate it and only if statistically indistinguishable judged outcomes are demonstrated.

### Do not chase

- More Scalene debugging
- Lowering `max_new_tokens` (model stops naturally, cap is inert)
- Disk I/O, data loading, quantization
- `torch.compile()` before hook measurement
- Custom C++/Triton kernels
