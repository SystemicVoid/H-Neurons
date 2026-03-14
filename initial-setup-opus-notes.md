
  H-Neurons vs Your mech-interp Environment: Compatibility Analysis

  What H-Neurons Does
  It identifies hallucination-associated neurons in LLMs. The pipeline: sample responses via vLLM → extract answer tokens →
  extract down_proj activations (CETT metric) → train a sparse logistic regression classifier → intervene on identified
  neurons at inference time. It uses HuggingFace `transformers` directly (not TransformerLens).

  Dependency Overlap


  ┌─────────────────┬─────────────────┬────────────────┐
  │ H-Neurons needs │ mech-interp has │ Status         │
  ├─────────────────┼─────────────────┼────────────────┤
  │ torch         │ 2.9.1+cu130     │ ✅             │
  │ transformers  │ 4.57.6          │ ✅             │
  │ datasets      │ 4.5.0           │ ✅             │
  │ accelerate    │ 1.12.0          │ ✅             │
  │ numpy         │ 1.26.4          │ ✅             │
  │ tqdm          │ 4.67.1          │ ✅             │
  │ scikit-learn  │ —               │ ❌ easy add    │
  │ joblib        │ —               │ ❌ easy add    │
  │ openai        │ —               │ ❌ easy add    │
  │ vllm          │ —               │ ⛔ blocker │
  └─────────────────┴─────────────────┴────────────────┘


  The vLLM Problem — This is the key issue

  Your mech-interp env runs Python 3.13 with CUDA 13.0 (cu130) targeting your RTX 5060 Ti's sm_120 (Blackwell). vLLM:

   1. Doesn't officially support Python 3.13 — CI only tests up to 3.12
   2. Doesn't ship wheels for CUDA 13.0 / sm_120 — pre-built wheels target older SM archs (86, 89, 90)
   3. Would require building from source with custom CUDA flags, and even that is uncharted territory for Blackwell

  Recommendation: Separate project, shared workspace

   Learning/
   ├── mech-interp/          # TransformerLens-based, Python ≥3.13, cu130
   │   └── lab/
   │       └── pyproject.toml
   └── h-neurons/            # HF transformers + vLLM, Python 3.12, cu126 or cu124
       └── pyproject.toml

  Why separate:
   - vLLM forces Python ≤3.12 — your mech-interp env requires ≥3.12 and is running 3.13. These are incompatible constraints.
   - vLLM needs different CUDA wheels — it doesn't support cu130. You'd need cu126/cu124, but PyTorch cu130 is what gives you
     sm_120 support. Mixing could cause runtime crashes.
   - Different paradigms — mech-interp uses TransformerLens (hook-based introspection), H-Neurons uses raw transformers +
     manual down_proj hooking. No shared abstractions.

  What you CAN share:
   - Model weights on disk (both load HF models the same way)
   - Data files (TriviaQA, any datasets you prepare)
   - Analysis results (the .npy activation files, classifier outputs)

  Practical escape hatch for vLLM: The collect_responses.py script is the only file that needs vLLM — it's just batch
  inference. You could replace it with a simple transformers generation loop (slower but works in your existing env) or run it
  as a standalone step with a separate minimal venv. The rest of the pipeline (extract_activations.py, classifier.py,
  intervene_model.py) only needs transformers + scikit-learn and would work fine in your mech-interp env with 3 small uv add
  calls.
  --
  The paper's pipeline fundamentally needs two things from a model:

   1. Black-box inference (steps 1-2): "Give me text completions" → llama.cpp is perfect for this, and arguably better for
      your single-GPU setup since vLLM's overhead is designed for multi-GPU batched serving
   2. White-box introspection (steps 4, 6): "Let me peek inside your neurons" → requires full PyTorch weights, no way around
      it
---
 Can you skip vLLM and use llama-server? Yes — with nuance.

  What vLLM actually does in H-Neurons

  It's used in exactly one script: collect_responses.py. That script does:

   1. Load a model via vllm.LLM(model=path)
   2. Call self.sampling_llm.chat(messages, SamplingParams(temperature=1.0, top_p=0.9, top_k=50, max_tokens=50)) in a loop —
      10 times per question to sample diverse responses
   3. Judge each response as true/false against ground truth

  That's it — pure text generation with chat templates. Your llama-server at 127.0.0.1:8080 already exposes
  /v1/chat/completions which is exactly the same API shape. I just confirmed it responds fine.

  What you CAN'T do with llama.cpp

  The activation extraction step (`extract_activations.py`) is the real meat of the paper, and it:
   - Loads the model via AutoModelForCausalLM.from_pretrained() (full PyTorch)
   - Registers forward hooks on every `down_proj` layer to capture intermediate neuron activations
   - Computes the CETT metric (contribution-weighted activations normalized by output norms)

--
 Step               │ Script                     │ Can use    │ Notes                                              │
  │                    │                            │ llama.cpp? │                                                    │
  ├────────────────────┼────────────────────────────┼────────────┼────────────────────────────────────────────────────┤
  │ 1. Collect         │ collect_responses.py     │ ✅ Yes │ Replace vLLM with OpenAI client → 127.0.0.1:8080 │
  │ responses          │                            │            │                                                    │
  │ 2. Extract answer  │ extract_answer_tokens.py │ ⚠️ Partial │ Uses AutoTokenizer + OpenAI API for LLM judge.   │
  │ tokens             │                            │            │ Tokenizer must match the model                     │
  │ 3. Sample balanced │ sample_balanced_ids.py   │ N/A        │ Pure JSON/Python, no model needed                  │
  │ IDs                │                            │            │                                                    │
  │ 4. Extract         │ extract_activations.py   │ ❌ No  │ Requires PyTorch model + forward hooks         │
  │ activations        │                            │            │                                                    │
  │ 5. Train           │ classifier.py            │ N/A        │ Pure scikit-learn, no model needed                 │
  │ classifier         │                            │            │                                                    │
  │ 6. Intervene on    │ intervene_model.py       │ ❌ No  │ Modifies down_proj weights in-place              │
  │ model              │                            │            │

  ---


Start with `Gemma-3-4B-it`. Reasons:

   1. Comfortably fits — 8GB weights leaves ~8GB for activation hooks during CETT extraction
   2. Decent results in the paper — 76.9% TriviaQA, 70.7% NQ, 71.0% BioASQ
   3. They provide example data for Gemma already (gemma27b_TriviaQA_* files) — different size but same tokenizer family, so
      you can reference their data format
   4. GGUF versions exist for llama.cpp — so you can serve the quantized version for response collection (step 1), then load
       the full bf16 HF model for activation extraction (step 4)

   ---
   ## Setup Log (Amp, 2026-03-14)

   ### What was done

   1. **Cloned H-Neurons** into `~/Documents/Engineering/mech-interp/lab/02-h-neurons/` as a uv workspace member
   - Created `pyproject.toml` matching the `*-*` workspace pattern
   - Deps already satisfied by workspace root: `scikit-learn`, `joblib`, `openai` (added to root pyproject.toml)
   - vLLM was **never installed** — not needed

   2. **Patched `scripts/collect_responses.py`** — replaced vLLM with OpenAI client
   - Removed: `from vllm import LLM, SamplingParams`, `import torch`, `--gpu_util`, `--tp_size` args
   - Added: `--sampling_base_url` (default `http://127.0.0.1:8080/v1`), `--sampling_api_key` (default `not-needed`)
   - Sampling now uses `OpenAI(base_url=...).chat.completions.create(model="local", ...)` 
   - Params preserved: `temperature=1.0, top_p=0.9, top_k=50, max_tokens=50`
   - `top_k` is non-standard OpenAI but llama-server accepts it as an extra body param

   3. **Symlinked** from Bluedot project for convenience:
   ```
   ~/Documents/Learning/Bluedot Project Technical AI safety/h-neurons-extension
     → ~/Documents/Engineering/mech-interp/lab/02-h-neurons
   ```

   ### llama-server API shape (confirmed live)
   - Endpoint: `http://127.0.0.1:8080/v1/chat/completions`
   - Model name: `"local"` (returned by `/v1/models`)
   - Response: standard OpenAI shape — `completion.choices[0].message.content`
   - Extra fields like `timings.predicted_per_second` available but not needed
   - Server reports `owned_by: "llamacpp"`, format `"gguf"`

   ### Quirks / things to watch
   - **`--model_path` arg is still required** in `collect_responses.py` but no longer used for loading
   (vLLM used it to load weights). It's still referenced in the data loading path via `item.get(...)`.
   Keep passing it — it's harmless and other scripts (extract_activations, extract_answer_tokens) still
   need a HF model path for tokenizer/weights.
   - **`top_k` passthrough**: The `openai` Python client will silently pass `top_k` as an extra body param.
   llama-server accepts it. If you ever point this at actual OpenAI API, it'll be ignored (not error).
   - **Judge step**: `extract_answer_tokens.py` still calls an LLM judge (GPT-4o by default) via OpenAI API.
   You could point this at llama-server too, but the extraction quality matters — a capable model helps here.

   ### Key decision: one download vs two

   **Can we use a single unquantized model for everything?**
   Yes — if you skip llama-server and use `transformers` `model.generate()` for step 1 too. Download HF
   bf16 weights once (~8 GB), use them for both response collection and activation extraction. The catch:
   llama-server needs GGUF format, transformers needs HF safetensors — they're incompatible file formats.
   So "single download" means ditching llama-server for step 1.

   **How much slower is `transformers` vs llama.cpp for step 1?**
   Job: 10K questions × 10 samples × ~20 tokens avg = ~2M tokens.
   - llama.cpp (4B F16): ~100-120 tok/s → **~5 hours**
   - transformers bf16:   ~40-60 tok/s  → **~11 hours**
   Delta is ~6 hours. One's an afternoon, the other's overnight. Both fine for a one-shot run.

   **How much disk space does the full pipeline produce?**
   - Steps 1-3 (JSONL text data): **~15 MB** — negligible
   - Step 4 (activation .npy files): **~6-12 GB** — the bulk
     - Each file: [layers × neurons] × float32 ≈ 1.5 MB
     - 2,000 questions × 2 locations = 4,000 files ≈ 6 GB
     - All 4 locations doubles it to ~12 GB
   - Step 5 (classifier .pkl): kilobytes

   ### Next steps
   1. Download Gemma-3-4B-it HF weights (single download — used for both generation and activation extraction)
   2. Get TriviaQA parquet: check if `data/TriviaQA/rc.nocontext/` is included or needs download
   3. Run the pipeline: collect → extract tokens → balance → extract activations → train classifier

