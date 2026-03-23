# Throughput Assessment — Local H-Neuron Intervention Runs

_Created 2026-03-20 by splitting performance analysis out of `docs/tooling-assessment.md`._
_Updated 2026-03-23 with deeper workload characterisation and concrete intervention alternatives._

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

## Tokenization Caching Across Alphas

Safe: yes. Jailbreak prompts are deterministic and reused across all alphas. Caching chat-template output, tokenized `input_ids`, and prompt length is scientifically clean.

Expected payoff: low single-digit % overall. Generation at ~28 s/sample dominates. Still worth doing — the implementation cost is tiny and you reuse the same 500 prompts across 4 alphas.

Best version: cache CPU-side tensors once per sample, then move to device per call. No need to keep all cached prompts on GPU.

## `torch.inference_mode()` vs `torch.no_grad()`

Probably safe, but smoke-test it. The hooks do in-place mutation of inference tensors, which is usually okay inside `inference_mode()`, but because you are mutating tensors in hooks, do not flip this blind on a long run without a short parity test.

Expected speedup: tiny, maybe 0–3%, maybe nothing measurable. Good hygiene, not salvation.

## What the Original Assessment Missed

### 1. `alpha=1.0` should bypass the intervention path entirely

In `SAEFeatureScaler`, there is already a short-circuit: `if self._alpha == 1.0: return output`. In `HNeuronScaler`, there is not. So right now the `alpha=1.0` run still pays full hook + indexed-write overhead while doing a mathematical no-op.

For neuron mode: either add a fast-path branch in the hook, or better, for `alpha == 1.0` run with **no intervention installed at all**. This alone speeds up **25% of a 4-alpha sweep**.

### 2. Verify stop behaviour under 1024

After increasing the cap, check the output-length histogram. If too many jailbreak responses land exactly at 1024, either the model genuinely wants to ramble forever, or the stop configuration is not doing what you think. Fixing a stop-token/config issue is a **scientifically clean** throughput win.

### 3. Current indexed assignment is a bad micro-kernel choice

`x[:, :, idx] = x[:, :, idx] * self._alpha` probably involves indexed gather → multiply → indexed scatter/writeback. A precomputed dense scale vector (`x = x * scale_vector`) gives the runtime a much simpler memory access pattern.

## Measurement Plan

If the goal is faster research throughput, spend engineering time on this instead of more Scalene debugging:

### Per-sample instrumentation

- prompt token count
- generated token count
- time in `model.generate()`
- time outside `model.generate()`

### Per-alpha instrumentation

- average generated length
- tokens/sec
- samples/sec

### Targeted A/B comparisons (20–50 sample slice)

- current hook path
- no-op / `alpha=1.0` with hooks bypassed
- refactored non-hook intervention (wrapped forward or weight scaling)
- cached tokenization
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

### Tier 1 — do immediately

1. **Bypass intervention completely for `alpha=1.0`**
2. **Add cheap timers** around: chat templating/tokenization, H2D copy, `model.generate()`, decode
3. **Cache tokenized prompts across alphas**

### Tier 2 — highest-value real optimisation

4. Replace hook-based neuron intervention with either:
   - **wrapped `down_proj` forward + activation-side scale vector** if you want the cleanest parity story
   - **weight-column scaling** if you want max speed and are willing to validate "statistically indistinguishable" rather than assume exactness

### Tier 3 — only after that

5. Try **batch size 2** on a fixed jailbreak slice
6. Keep batching off the canonical run unless judged outcomes look stable

**Bottom line:** the bottleneck is not "the GPU is too small." The bottleneck is that the experiment is paying a Python tax on every token, every layer, for a transformation that is structurally much simpler than the machinery used to express it. Fix the representation of the intervention first; then revisit batching.
