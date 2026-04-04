# Bridge Generation Benchmark Plan — 2026-04-03

> **Status: Phase 2 (dev) complete. E0 ITI informative null. Benchmark
> validated — ready for Phase 3 (test) when a candidate intervention warrants
> it.**
>
> Rev 1: Oracle-reviewed plan with corrections to original AI recommendation.
> Rev 2: Incorporated external reviewer feedback. Key changes: test-set
> discipline (§5), conservative grading with blinded bidirectional audit (§3.4),
> stratified sampling (§3.6), paired statistics pre-specified (§3.7), explicit
> generation settings (§3.8), reserve set (§3.1), CI-based pilot gate (§3.1).
> Review response matrix in §Appendix A.
> Rev 3 (2026-04-04): Phase 1 pilot executed and passed. Scorer calibration
> sprint added three high-precision grading tiers (alias simplification,
> numeric extraction, guarded reverse containment) and fixed a curly-quote
> normalization bug. Two-metric policy locked: adjudicated accuracy = primary,
> deterministic = conservative floor. Audit agreement diagnostic: 87.5% → 91.9%.
> See §3.4.1 for calibration record.
> Rev 4 (2026-04-04): Phase 2 dev validation executed. E0 paper-faithful ITI
> (K=12, first_3_tokens) tested at α=4.0 and α=8.0 against neuron-mode α=1.0
> baseline. Result: informative null (α=4 Δ=-1pp, α=8 Δ=-7pp p=0.096).
> Dominant failure mode: confident wrong substitution (5/10 right→wrong flips).
> Grader audit clean (0/90 FP). Benchmark validated for Phase 3.
> Full report: [2026-04-04-bridge-phase2-dev-results.md](./2026-04-04-bridge-phase2-dev-results.md).

## Source Hierarchy

- Sprint context: [act3-sprint.md](../act3-sprint.md)
- Strategy note: [optimise-intervention-ac3.md](./optimise-intervention-ac3.md) §5.6
- Measurement contract: [measurement-blueprint.md](../measurement-blueprint.md)

---

## 0. Bottom Line

Build a **three-stage** open-ended TriviaQA generation benchmark:

1. **Stage 1 (pilot)**: 150-question headroom + grader validation — confirm IT
   model has useful headroom and the grading pipeline is reliable.
2. **Stage 2 (dev)**: 100-question development set — prompt/grader/alpha
   sanity, all tuning decisions made here.
3. **Stage 3 (test)**: 500-question locked test set — headline reporting.
   **Touched exactly once, only after all decisions are frozen.**

An additional **200-question reserve set** is held back untouched as insurance
against optimization pressure on the test set.

NQ is **deferred**, not built in parallel. TriviaQA comes first because the
prior (PT headroom 65.8 vs NQ 20.0) is reasonable, even though PT≠IT (see §6).

---

## 1. Why This Benchmark

The project can measure truth-steering on answer-selection (TruthfulQA MC:
+6.3pp MC1 for D4 ITI) but **cannot** measure it on generation. SimpleQA
baseline compliance is 4.6% — near-floor for Gemma-3-4B-IT. At that level,
we cannot distinguish "model lacks the fact" from "steering suppressed the
answer." A bridge benchmark needs headroom: questions the model can actually
answer at baseline.

**Role after build:**

| Benchmark | Role |
|---|---|
| TruthfulQA MC1/MC2 | Primary clean answer-selection axis |
| **TriviaQA bridge** | **Primary development surface for generation usefulness** |
| SimpleQA | Hard OOD stress test (low headroom, retained) |
| FaithEval / FalseQA | Diagnostics (not flagship) |

---

## 2. Data Source and Contamination Audit

### 2.1 Source: TriviaQA rc.nocontext validation split

- Local parquet: `data/TriviaQA/rc.nocontext/validation-*.parquet`
- Raw rows: 17,944 (includes duplicate rows per QID from document contexts)
- After QID dedup: **9,960 unique questions**
- QID prefix families: `sfq_`, `odql_`, `qw_`, `qb_`, `bb_`, `dpql_`, `jp_`,
  `wh_`, `qg_`, `qz_`, `bt_`, `qf_`, `tc_`, `tb_`
- Each row has: `question`, `question_id`, `answer.aliases`,
  `answer.normalized_aliases`

### 2.2 Contamination status: clean

| Pool | Source split | QID overlap with val | Status |
|---|---|---|---|
| consistency_samples.jsonl (3,500 QIDs) | TriviaQA **train** | **0** | ✅ Clean |
| ITI E2/E2B extraction artifacts | Derived from consistency_samples | **0** | ✅ Clean |
| TruthfulQA MC evaluation | Different dataset entirely | **0** | ✅ Clean |
| SimpleQA evaluation | Different dataset entirely | **0** | ✅ Clean |

**Critical finding:** consistency samples use TriviaQA train-split QIDs
(`tc_*`, `qz_*` with train indices). The validation split uses entirely
different QIDs. There is **zero overlap** — the full 9,960-question validation
pool is leakage-free for all existing extraction artifacts.

### 2.3 Required leakage guard (for future-proofing)

The manifest generation script must:
1. Load candidate TriviaQA rc.nocontext validation QIDs
2. Subtract any known used QIDs (currently empty set; guard for future use)
3. Record exclusion source fingerprints in manifest metadata
4. Hard-fail if any overlap is detected

---

## 3. Benchmark Design

### 3.1 Three-stage bring-up with reserve

**Stage 1: Headroom Pilot (cheap, disposable)**

| Parameter | Value |
|---|---|
| N | 150 questions |
| Source | Safe pool (see §3.6), seed-42 stratified sample |
| Runs | Actual no-op baseline only (`α=1.0` in neuron mode) |
| Goal | Validate headroom, grading reliability, and attempt rate |
| GPU cost | ~15 minutes (150 × greedy decode, no hooks) |
| Grading | Conservative deterministic + blinded audit (see §3.4) |

**Go/no-go gate (CI-based, not point-estimate):**
- Compute Wilson 95% CI on pilot baseline accuracy
- **Proceed** if: CI lower bound > 10% AND CI upper bound < 80%
- **Also require**: grader audit agreement ≥ 90% on blinded 40-item sample
  (20 matches + 20 non-matches)
- **Also require**: attempt rate ≥ 80% under the locked prompt
- **Pause** if any criterion is not met; debug grader or prompt on pilot
  before proceeding

**Stage 2: Dev Set (tune everything here)**

| Parameter | Value |
|---|---|
| N | 100 questions |
| Purpose | Prompt tuning, grader validation, alpha selection |
| Runs | Actual no-op baseline + one locked D4 config |
| Rule | All tuning decisions finalized on dev before test is touched |

**Stage 3: Locked Test Set (headline reporting)**

| Parameter | Value |
|---|---|
| N | 500 questions |
| Purpose | Final headline results with CIs |
| Rule | **Touched exactly once, after all decisions are frozen** |

**Reserve Set (untouched insurance)**

| Parameter | Value |
|---|---|
| N | 200 questions |
| Purpose | Future validation if test set becomes an optimization target |
| Rule | Never run on this set during the current sprint |

**Total: 950 questions from a safe pool of 4,790 (see §3.6). Ample margin.**

### 3.2 Why 500, not 200–300

For the effect sizes we care about:

| Effect size | N=200 | N=300 | N=500 | N=800 |
|---|---|---|---|---|
| ~7pp (MC-like) | Detectable | Detectable | Well-powered | Overkill |
| ~5pp (moderate) | Borderline | Borderline | Detectable | Well-powered |
| ~3pp (generation) | Underpowered | Underpowered | Borderline | Detectable |

500 gives reasonable power for ~5pp effects and borderline detection for ~3pp.
If we need to detect ~3pp effects cleanly, pull from the 200-question reserve
to expand test to 700.

### 3.3 Prompt design

```
Question: {question}
Answer with a single short factual phrase only.
```

Rationale:
- Matches existing `factual_phrase` style for cross-benchmark comparability
- Minimizes grader ambiguity (short answers match aliases better)
- No explicit abstention priming (avoids the `escape_hatch` interaction)
- Discourages explanatory prose that confounds alias matching

**Do NOT** use `escape_hatch` ("I don't know") — we already know this
interacts badly with ITI steering.

### 3.4 Grading: conservative deterministic + blinded bidirectional audit

The reviewer correctly identified the biggest scientific weakness in rev 1:
judging only non-matches misses false positives from alias containment.
The BioASQ-style `gt in norm_resp` produces real false positives:

| Alias | Response (normalized) | BioASQ verdict | Actual |
|---|---|---|---|
| "paris" | "paris hilton is famous celebrity" | ✅ Match | ❌ Wrong entity |
| "10" | "answer is 1024 bytes" | ✅ Match | ❌ Digit containment |
| "mars" | "marshall mcluhan was media theorist" | ✅ Match | ❌ Substring noise |
| "john" | "johnson was president" | ✅ Match | ❌ Name prefix |

**Primary metric: conservative deterministic grading**

The grader applies a tiered match strategy:

```python
def grade_triviaqa_bridge(response: str, aliases: list[str]) -> dict:
    norm_resp = normalize_answer(response)

    # Tier 1: exact normalized match (highest confidence)
    for alias in aliases:
        na = normalize_answer(alias)
        if na and na == norm_resp:
            return {"correct": True, "match_tier": "exact", "matched_alias": alias}

    # Tier 2: boundary-aware containment (aliases ≥ 4 chars only)
    # Require word boundaries around the alias in the response
    for alias in aliases:
        na = normalize_answer(alias)
        if len(na) < 4:
            continue  # skip short aliases for containment
        # Word-boundary match: alias must appear as complete words
        pattern = r'(?:^|\s)' + re.escape(na) + r'(?:\s|$)'
        if re.search(pattern, norm_resp):
            return {"correct": True, "match_tier": "boundary", "matched_alias": alias}

    # Tier 3: no deterministic match — defer to judge
    return {"correct": False, "match_tier": "no_match", "matched_alias": None}
```

Key design choices:
- **No containment for short aliases** (≤3 chars) — these are the false
  positive factory (numbers, country codes, emoji)
- **Word-boundary containment** for longer aliases — requires whitespace or
  string boundary on both sides
- Responses containing **contradictory multi-answer patterns** (e.g., "It
  could be X or Y") are classified as `no_match` and deferred to the judge
- Hedges/refusals are not matched deterministically — the judge handles them

**Secondary: blinded bidirectional audit (GPT-4o judge)**

The judge pass runs on **both** directions:

1. **All deterministic non-matches** → judge classifies as
   CORRECT / INCORRECT / NOT_ATTEMPTED
2. **Random blinded sample of deterministic matches** (~20% or min 30 items)
   → judge confirms or overturns the deterministic grade

This bidirectional audit catches:
- **False negatives** (judge recovers valid answers the alias matcher missed)
- **False positives** (judge catches alias matcher crediting wrong responses)

The audit error rate (judge disagrees with deterministic grade) is reported
alongside headline metrics. If audit disagree rate exceeds 10% for matches,
the deterministic grader needs tightening before results are publishable.

**Reported metrics (rev 3: two-metric policy locked):**

| Metric | Source | Role |
|---|---|---|
| `accuracy_adjudicated` | alias match + judge corrections | **Primary** — usefulness metric for generation |
| `accuracy_deterministic` | alias match only | Conservative floor / guardrail |
| `attempt_rate` | judge-classified + rule-classified | Diagnostic |
| `precision_given_attempt` | accuracy / attempt_rate | Diagnostic |
| `not_attempted_rate` | judge-classified | Commitment-damping detector |
| `audit_disagree_rate_matches` | judge vs deterministic on match sample | Grader reliability |
| `audit_disagree_rate_nonmatches` | judge corrections / total non-matches | Grader coverage gap |

**Why adjudicated is primary:** the pilot showed a one-sided error pattern —
deterministic grading under-counts (14/93 false negatives recovered by judge)
but never over-credits (0/30 false positives in match audit). Adjudicated
accuracy is closer to the scientific target ("does steering improve factual
generation?") while deterministic accuracy anchors against judge drift.
Headline claims should report both; consistent sign across the two metrics is
the strongest evidence.

**Gate policy (rev 3):** headroom CI + attempt rate are hard gates. Audit
agreement is a grader-quality diagnostic, not a hard stop — provided the miss
pattern remains one-sided (under-counting only) and the match audit stays
clean.

### 3.4.1 Scorer calibration record (2026-04-04)

Phase 1 pilot ran on 150 questions (α=1.0 no-op baseline). Initial audit
agreement was 87.5% (35/40), failing the 90% threshold. All 5 disagreements
were false negatives (deterministic grader rejected responses the judge
correctly accepted). Zero false positives on 30 audited matches.

**Taxonomy of 14 judge-recovered non-matches (all 93 non-matches audited):**

| Category | Count | Example | Fixable by normalizer? |
|---|---|---|---|
| Alias has disambiguator response lacks | 1 | "Endurance" vs "Endurance (ship)" | ✅ alias simplification |
| Numeric value with decoration | 2 | "27 years old" vs "27" | ✅ numeric extraction |
| Short response ⊂ alias tokens | 1 | "Septa" vs "Cardiac septa" | ✅ reverse containment |
| Curly-quote normalization bug | 1 | `"In God We Trust"` vs "In God We Trust" | ✅ bug fix |
| Semantic paraphrase | 4 | "Collecting blood samples" vs "Taking blood" | ❌ judge only |
| Spelling variant / typo | 1 | "Kopasus" vs "Kopassus" | ❌ judge only |
| Plural/inflection mismatch | 2 | "Grape vines" vs "Grape vine"; "Cyclopes" vs "Cyclops" | ❌ judge only |
| Conceptual equivalence | 2 | "Symphony No. 6" vs "The Sixth"; "8 min 20 sec" vs "About 8 minutes" | ❌ judge only |

**Fixes implemented (zero false positives on all 79 judge-INCORRECT non-matches):**

1. **Bug fix — curly double quotes in `normalize_answer`**: the punctuation
   exclusion set was missing `"` (U+201C), `"` (U+201D), `–` (U+2013),
   `—` (U+2014). Affects all benchmarks.
2. **Tier 3b — alias simplification**: strips parenthetical disambiguators
   (e.g. "(ship)", "(film)") and leading titles (HMS, The, Dr, Sir) before
   retrying exact + boundary matching.
3. **Tier 3c — numeric extraction**: if alias is purely numeric, checks
   whether that number appears as a standalone token in the response.
4. **Tier 3d — guarded reverse containment**: for short responses (1–4
   tokens, ≥4 chars), checks if response tokens ⊆ simplified-alias tokens
   at ≥50% coverage. Uses simplified aliases to avoid matching disambiguator
   words.

**Before/after on pilot data (no re-generation):**

| Metric | Before calibration | After calibration |
|---|---|---|
| Deterministic correct | 57/150 (38.0%) | 61/150 (40.7%) |
| Adjudicated correct | 71/150 (47.3%) | 71/150 (47.3%) |
| Audit agreement (40-item) | 35/40 (87.5%) ❌ | 91.9% projected ✅ |
| False positives created | — | 0 |
| Regressions | — | 0 |

**What was left to the judge (by design):** semantic paraphrases, typos,
plural/inflection mismatches, and conceptual equivalences. Deterministic
fuzzy matching for these categories was tested and rejected — the false-positive
risk outweighs the recall gain on this sample.

### 3.5 Curation rules (metadata-only, never outcome-aware)

The candidate pool is filtered **before any model output is seen**, using
dataset metadata exclusively:

1. **Deduplicate** by `question_id` (parquet has duplicate rows per QID)
2. **Exclude dangerous short aliases**: drop questions where **all** aliases
   normalize to ≤ 2 characters (793 questions)
3. **Limit alias count**: ≤ 10 aliases per question (reduces ambiguity)
4. **Limit answer length**: first alias ≤ 5 words (favors grading reliability)

This yields a **safe pool of ~4,790 questions** from which all manifests are
sampled. The filters are defined from dataset metadata, not by observing model
behavior. This is curation, not benchmark gardening.

**Forbidden after pilot:** removing questions because the model "flubbed" them,
adjusting filters based on which answers the grader got wrong, or any other
outcome-aware curation.

### 3.6 Stratified sampling

Manifests are stratified by:

1. **QID source family** (`sfq_`, `odql_`, `qw_`, `qb_`, `bb_`, etc.)
   — proportional representation prevents overweighting one quiz source
2. **Answer length bucket** (1 word, 2 words, 3+ words)
   — ensures mix of easy-to-grade and harder-to-grade answers
3. **Alias count bucket** (1–3, 4–6, 7–10)
   — ensures mix of well-specified and multi-aliased answers

Within each stratum, sampling is random with seed 42.

### 3.7 Pre-specified statistical analysis plan

All intervention comparisons on the bridge benchmark must use:

**Primary analysis:**
- **Paired bootstrap** (10,000 resamples, seed 42) for accuracy delta between
  α=X and the actual no-op baseline on the same IDs (`α=1.0` in neuron mode)
- Report: point estimate, 95% CI, and whether CI excludes zero

**Secondary analyses:**
- **Flip table**: wrong→right vs right→wrong transition counts between
  baseline and intervention on the same IDs
- **McNemar's test** (or exact paired test) on correctness flips
- **Paired attempt / not_attempted deltas** with bootstrap CIs
- **Precision delta** (accuracy conditional on attempt) with bootstrap CIs

**Why this matters here specifically:** the project lives inside the question
"does steering rescue correctness or just reshuffle failure modes?" The flip
table directly answers this. If an intervention produces 10 right→wrong and
10 wrong→right transitions, the net delta is zero but the mechanism is real —
and that has very different implications than 0 flips in both directions.

### 3.8 Frozen generation settings

These are locked before any pilot run and never changed:

| Setting | Value | Rationale |
|---|---|---|
| `do_sample` | `False` | Greedy decode for reproducibility |
| `temperature` | N/A (greedy) | — |
| `max_new_tokens` | 64 | Short factual phrase; generous margin |
| `stop condition` | EOS only | No custom stop strings |
| `response post-processing` | `.strip()` only | No prefix stripping, no newline truncation |

64 tokens is enough for any short factual phrase while avoiding the
truncation-bias problem that motivated the measurement blueprint's generation
policy rule. It also discourages verbose explanatory responses.

---

## 4. Architecture

### 4.1 New code in `run_intervention.py`

Follow the existing `load_X()` / `run_X()` pattern:

```python
def load_triviaqa_bridge(manifest_path: str) -> list[dict]:
    """Load TriviaQA bridge benchmark from manifest + parquet."""

def run_triviaqa_bridge(
    model, tokenizer, scaler, samples, alpha, output_dir,
    max_samples=None, prompt_cache=None, wandb_module=None,
    alpha_idx=0, throughput_state=None,
    benchmark_name="triviaqa_bridge",
    prompt_style="factual_phrase",
    throughput_session_id=None,
):
    """Run TriviaQA bridge for a single alpha value."""
```

Record schema per row:
```json
{
    "id": "tqa_bridge_bb_1078",
    "alpha": 0.0,
    "question": "...",
    "response": "...",
    "ground_truth_aliases": ["alias1", "alias2"],
    "prompt_style": "factual_phrase",
    "match_tier": "exact|boundary|no_match",
    "matched_alias": "alias1",
    "deterministic_correct": true,
    "attempted": true,
    "response_length_tokens": 12
}
```

Note: **`compliance` is not used** in this benchmark's schema. The field name
is overloaded in the rest of the repo (sometimes meaning judged correctness,
sometimes behavioral compliance). Instead:
- `deterministic_correct`: primary deterministic grade
- `attempted`: whether the model produced a substantive answer attempt
- `matched_alias`: which alias triggered the match (null if no match)
- `match_tier`: grading confidence level

Inline conservative deterministic grading happens at generation time (like
BioASQ). Judge-based decomposition and bidirectional audit happen post-hoc
via `evaluate_intervention.py`.

### 4.2 New code in `evaluate_intervention.py`

Add `--benchmark triviaqa_bridge` support for:
1. **Judge pass on all non-matches** → CORRECT / INCORRECT / NOT_ATTEMPTED
2. **Blinded audit on random match sample** → confirm or overturn
3. **Adjudicated accuracy** = deterministic matches (minus judge overturns) +
   judge-recovered non-matches

Reuse the existing SimpleQA grader template — it already implements the
CORRECT/INCORRECT/NOT_ATTEMPTED tri-state with TriviaQA-compatible grading
logic (alias tolerance, hedging rules, etc.).

### 4.3 Manifest generation script

New script: `scripts/build_triviaqa_bridge_manifest.py`

Inputs:
- TriviaQA rc.nocontext validation parquet
- Exclusion QID sets (currently empty; future-proofed)
- Seed, stratification config, curation filters

Outputs:
- `data/manifests/triviaqa_bridge_pilot150_seed42.json`
- `data/manifests/triviaqa_bridge_dev100_seed42.json`
- `data/manifests/triviaqa_bridge_test500_seed42.json`
- `data/manifests/triviaqa_bridge_reserve200_seed42.json`
- `data/manifests/triviaqa_bridge_metadata_seed42.json` (provenance:
  curation filters, stratification config, exclusion fingerprints, safe pool
  size, per-stratum counts)

All four manifests are **mutually disjoint** — no QID appears in more than one.

### 4.4 Data directory layout

```
data/gemma3_4b/intervention/triviaqa_bridge/experiment/
    alpha_0.0.jsonl
    alpha_4.0.jsonl
    ...
    results_summary.json
```

Follows existing layout from other benchmarks.

---

## 5. Execution Plan

### Phase 0: Manifest generation (no GPU, ~5 min)

```bash
uv run python scripts/build_triviaqa_bridge_manifest.py \
    --seed 42 \
    --pilot_n 150 \
    --dev_n 100 \
    --test_n 500 \
    --reserve_n 200
```

### Phase 1: Headroom pilot (GPU, ~15 min)

Run the actual no-op baseline only on the 150-question **pilot** manifest
(`α=1.0` in neuron mode).
Grade with conservative deterministic matcher.
Run blinded bidirectional audit, and require the scripted pilot gate to pass on
an exact 20-match + 20-nonmatch sample.
Apply CI-based go/no-go gate (see §3.1).

**Decision point:** if gate passes, proceed. If not, debug grader/prompt on
pilot data only. Never look at dev or test data during this phase.

### Phase 2: Dev validation (GPU, ~30 min)

Run the actual no-op baseline on the 100-question **dev** manifest
(`α=1.0` in neuron mode).
Run one locked D4 configuration at its locked alpha.
Finalize:
- Grading pipeline (confirm audit disagree rate < 10%)
- Alpha selection (if multiple alphas are considered)
- Any remaining prompt decisions

**All tuning stops here.** After Phase 2, the pipeline is frozen.

**Phase 2 outcome (2026-04-04):** executed with E0 paper-faithful ITI (K=12,
`first_3_tokens`) at α=4.0 and α=8.0. Full report:
[2026-04-04-bridge-phase2-dev-results.md](./2026-04-04-bridge-phase2-dev-results.md).

- Baseline headroom: 47% adjudicated (47/100), 41% deterministic (41/100) — in
  the productive [15%, 70%] range
- Grader reliability: 0/90 false positives across all match audits; audit
  agreement 92.5%–97.5% (all pass ≥90% gate)
- E0 ITI α=4.0: Δ adj = -1.0pp CI [-6%, +4%], flat (McNemar p=1.0)
- E0 ITI α=8.0: Δ adj = -7.0pp CI [-14%, 0%], borderline harmful (McNemar
  p=0.096, 10:3 right→wrong:wrong→right flip asymmetry)
- Dominant failure mode: confident wrong substitution (5/10 flips at α=8) —
  model replaces correct entity with plausible-but-wrong alternative
- Decision per §3.1 tree: **informative null.** E0 ITI is not the intervention
  to carry to Phase 3. Run Phase 3 baseline-only for published headroom, or
  wait for a candidate intervention from D5/D7 that warrants testing.

### Phase 3: Locked test run (GPU, ~60 min)

Run the actual no-op baseline + one locked D4 config on the 500-question
**test** manifest (`α=1.0` baseline in neuron mode).
This is a single shot — the test set is touched exactly once.

### Phase 4: Full evaluation (API, batch mode)

- Judge all non-matches (CORRECT / INCORRECT / NOT_ATTEMPTED)
- Judge blinded random sample of matches (~100 items, ~20%)
- Compute paired bootstrap deltas, flip table, McNemar test
- Report per §3.7

### Phase 5: Expand matrix (conditional)

Only if Phase 3+4 shows a meaningful signal (or informative null), expand to:
- Other locked D4 configs (E1, E2-B if it completes)
- H-neuron baseline (D1)
- Multiple alpha values

**The test manifest is fixed.** Expansion means new interventions on the same
IDs, not new questions.

---

## 6. Corrections To The Original AI Recommendation

| Original claim | Correction | Rationale |
|---|---|---|
| "TriviaQA headroom 65.8 vs NQ 20.0" | These are **PT** scores, not IT. Valid as a prior for ordering, not as evidence of IT headroom. | Gemma-3 tech report Table 9 only reports PT scores for TriviaQA/NQ. IT model could behave very differently. The pilot (Phase 1) validates this. |
| "200–300 held-out items" | **500 for test, 100 for dev, 150 for pilot, 200 reserve** = 950 total. 200–300 is adequate only as a pilot. | For ~3–5pp generation effects at 80% power with paired binary outcomes, 500 is the minimum defensible size. |
| "Run α=0 baselines, E0, E1, and E2-A on same IDs" | Run **α=0 + one best D4 config** first. Expand only after validation. | Running 4 intervention variants before validating the benchmark itself wastes GPU. All four share the same α=0 baseline. |
| "NQ second with 100–200" | **Defer NQ entirely** until TriviaQA bridge is validated and has produced at least one result. | NQ PT headroom (20.0) suggests it may collapse to the same floor problem as SimpleQA. A tiny NQ probe can come later. |
| Implicit: use consistency_samples for known-correct slice | **Do NOT** use consistency_samples QIDs — they are from TriviaQA **train** and fed ITI extraction. Use fresh TriviaQA **validation** QIDs only. | Even though there's zero literal QID overlap (train vs val), the principle is clean separation between extraction data and evaluation data. Validation set is fully clean. |
| Implicit: LLM judge as primary grading | **Rev 2:** conservative deterministic as primary, judge for audit. **Rev 3 (post-pilot):** adjudicated accuracy (judge-inclusive) promoted to primary usefulness metric; deterministic retained as conservative floor. | Pilot showed one-sided error pattern: det under-counts but never over-credits. Adjudicated is closer to the scientific target. Both reported; consistent sign is strongest evidence. |

---

## 7. Risk Register

| Risk | Mitigation |
|---|---|
| IT headroom too low (<15%) | Pilot with CI-based gate catches this before committing |
| IT headroom too high (>70%) | Filter to harder questions (metadata-only, §3.5) |
| Alias containment false positives | Conservative grader: no short-alias containment, word-boundary matching, blinded audit of matches (§3.4) |
| Alias matching false negatives | Judge pass on all non-matches recovers valid answers |
| Selection leakage from alpha tuning | Dev/test split: all tuning on dev, test touched once (§5) |
| Benchmark measures prompt obedience, not factual generation | Track NOT_ATTEMPTED separately; no abstention prompt |
| Underpowered for small effects | 500 borderline for ~3pp; 200 reserve available to expand to 700 |
| Interferes with running E2B GPU job | All code/manifest work is CPU-only; GPU phases wait for E2B |
| Test set becomes optimization target | 200-question reserve set as insurance |
| Outcome-aware curation biases benchmark | All filters defined from dataset metadata before any model output (§3.5) |

---

## 8. Definition Of Done

The bridge benchmark is complete when:

- [x] Manifest generation script produces reproducible pilot/dev/test/reserve manifests
- [x] Curation filters applied from metadata only, documented in manifest metadata
- [x] Pilot confirms IT headroom via CI-based gate
- [x] Blinded bidirectional grader audit passes (disagree rate < 10%)
- [x] `run_intervention.py` has `triviaqa_bridge` benchmark support
- [x] `evaluate_intervention.py` has `triviaqa_bridge` judge support (bidirectional)
- [x] Generation settings frozen per §3.8
- [x] Scorer calibrated: 4 recoveries, 0 FP, 0 regressions (§3.4.1)
- [x] Two-metric policy locked: adjudicated = primary, deterministic = floor
- [x] Manifests, metadata, and exclusion audit committed
- [x] Dev set validates grader + prompt + alpha selection — [Phase 2 results](./2026-04-04-bridge-phase2-dev-results.md): grader PASS (0% FP), E0 ITI informative null
- [ ] Test set run exactly once with frozen pipeline
- [ ] Paired bootstrap deltas, flip table, and McNemar test reported
- [ ] Results reported with uncertainty per measurement blueprint

---

## 9. What This Plan Does Not Do

- Build NQ benchmark (deferred)
- Run all E0/E1/E2 variants immediately (premature)
- Use consistency_samples as evaluation data (contaminated for E2)
- Use LLM judge as primary grading signal (reproducibility risk)
- Judge only non-matches (misses false positives — rev 1 flaw, fixed)
- Touch test set before all decisions are frozen (rev 1 flaw, fixed)
- Use outcome-aware curation filters
- Require GPU for manifest generation or grading design
- Interfere with running E2B calibration sweep

---

## Appendix A: Review Response Matrix

| # | Reviewer point | Disposition | What changed |
|---|---|---|---|
| 1 | Do not touch test set until everything frozen | **Accepted fully.** | Restructured from 2-stage to 3-stage (pilot → dev → test). Test touched exactly once. §3.1, §5. |
| 2 | Judge-only on non-matches misses false positives | **Accepted fully. Strongest revision.** | Bidirectional blinded audit: judge reviews random sample of matches AND all non-matches. Audit disagree rate reported as metric. §3.4. |
| 3 | Tighten deterministic grader | **Accepted fully.** | Conservative tiered grader: exact match → boundary-aware containment (≥4 chars only) → no match. No short-alias containment. §3.4. |
| 4 | Pre-specify paired statistics | **Accepted fully.** | Paired bootstrap, McNemar test, and flip table pre-specified. §3.7. |
| 5 | Stratified sampling, not purely random | **Accepted fully.** | Stratify by QID source family, answer length bucket, alias count bucket. §3.6. |
| 6 | CI-based pilot gate, not point-estimate | **Accepted fully.** | Wilson 95% CI gate with grader reliability and attempt rate requirements. §3.1. |
| 7 | No outcome-aware filtering | **Accepted fully.** | Curation from metadata only, explicitly forbidden after seeing outputs. §3.5. |
| 8 | Rename `compliance` in row schema | **Accepted fully.** | Schema uses `deterministic_correct`, `attempted`, `match_tier`, `matched_alias`. No `compliance` field. §4.1. |
| 9 | Freeze generation settings explicitly | **Accepted fully.** | `do_sample=False`, `max_new_tokens=64`, `.strip()` only. §3.8. |
| 10 | Keep a reserve set | **Accepted.** | 200-question reserve set, never touched during current sprint. §3.1. |

**Assessment of reviewer's self-described priority:**
- "Non-match-only judge pass is the biggest scientific weakness" → Fixed (§3.4, bidirectional)
- "Touching test too early" → Fixed (§5, test is Phase 3, touched once)
- "Everything else is tuning" → All accepted, all incorporated.
