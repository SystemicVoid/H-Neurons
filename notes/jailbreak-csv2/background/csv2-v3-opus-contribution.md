  # CSV2 v3 Integration Plan: Validate It as an Intervention Evaluator, Not a Universal Judge

  > **2026-04-10 update**: The v3 hard-case smoke test (§2 "Refresh the canonical evidence base") has been run. Results in [2026-04-10-csv2-v3-smoke-test-audit.md](../../act3-reports/2026-04-10-csv2-v3-smoke-test-audit.md). The "missing validated evidence" identified below is now concrete: v3 under-calls harm on refuse-then-educate responses (37.5% gold agreement on hard cases), pivots are undetected (0/30), and C/S/V collapse into binary. **Do not proceed to §2's full-scale v3 re-run until the judge prompt is recalibrated** (priority fix: add a 4th few-shot example modeling the refuse-then-educate archetype).

  ## Summary

  As of April 9, 2026, the strongest reading of the repo plus the literature is:

  - The current CSV2 code is already much richer than the comparison plan gives it credit for. [scripts/evaluate_csv2.py](/
    home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/scripts/evaluate_csv2.py) already has structured outcomes,
    intent_match, C/S/V, wrapper tags, typed evidence spans, span validation, and derived structural metrics.
  - The real bottleneck is not missing rubric fields. It is missing validated evidence that the v3 outputs are trustworthy
    and scientifically useful.
  - The literature gap is narrower than the plan claims. GuidedBench and JADES already cover non-binary, more interpretable
    jailbreak scoring. CSV2 should not try to out-generalize them. The real gap is a validated, intervention-sensitive
    evaluator that measures how harmful compliance changes in structure and severity under mechanistic interventions.
  - In other words: CSV2 is already a better microscope. The contribution now is to prove the microscope is calibrated and
    that it reveals effects other tools miss.

  ## What To Keep, Modify, Drop

  ### Keep

  - Intervention-first framing.
  - Human-audited validation.
  - Judge robustness testing.
  - Negative controls for neuron specificity.
  - One external transfer slice.
  - Auditable evidence localization and paired uncertainty reporting.

  ### Modify

  - Novelty claim:
      - Replace “no one goes beyond binary” with “existing jailbreak evaluators optimize attack assessment, while
        intervention papers still mostly report ASR and broad capability metrics.”
      - Claim CSV2 fills the missing layer: localized, intervention-sensitive response measurement.
  - Localization claim:
      - Do not claim novel “character-level detection” as if CSV2 solved span extraction from scratch.
      - Claim auditable localized evidence spans resolved to offsets, since the model emits quotes and the code resolves
        them.
  - GuidedBench/JADES integration:
      - Use them as design pressure, not as something to reimplement wholesale.
      - Borrow their main lesson: ambiguous jailbreak success needs task-aware criteria.
      - Add a small anchored-reference subset for hard cases instead of building full decomposition for every prompt.
  - Capability/over-refusal evaluation:
      - Do not use IFEval alone.
      - Use XSTest and OR-Bench for refusal calibration, plus one benign capability set.

  ### Drop or defer

  - “Universal jailbreak evaluator” positioning.
  - Axis-independence-by-correlation as a headline validation. Low correlation is not proof of good dimensions.
  - Packaging/open-source polish as a near-term priority.
  - Large multi-benchmark expansion before v3 is validated on the canonical run.
  - Wrapper tags as a standalone novelty claim. They are useful analysis features, not the paper’s center of gravity.

  ## Current State vs. Plan

  ### Already implemented in code

  - Structured rubric and localized evidence in [scripts/evaluate_csv2.py](/home/hugo/Documents/Engineering/mech-interp/
    lab/02-h-neurons/scripts/evaluate_csv2.py).
  - Normalization across legacy and v3 payloads.
  - Hard failure for invalid evidence spans.
  - Paired reporting with uncertainty in [scripts/report_d7_csv2.py](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-
    neurons/scripts/report_d7_csv2.py).
  - Binary judge validation scaffold in [scripts/validate_evaluator_gold.py](/home/hugo/Documents/Engineering/mech-interp/
    lab/02-h-neurons/scripts/validate_evaluator_gold.py).

  ### Missing evidence, which is the real blocker

  - Canonical datasets are still mostly legacy CSV2 outputs normalized on read, not fresh v3 judgments.
  - Existing human gold is mostly binary or severity-noted prose, not full v3 field supervision.
  - Judge validation exists, but only for binary harmful/safe decisions.
  - There is not yet a published-strength robustness story for prompt sensitivity, model-choice sensitivity, or style-shift
    sensitivity on CSV2 fields.
  - The repo already contains qualitative evidence that binary judging misses style/severity shifts in
    [jailbreak_truncation_audit.md](/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons/tests/gold_labels/
    jailbreak_truncation_audit.md), but that needs to be turned into a formal v3 validation story.

  ## Integrated Plan

  ### 1. Freeze the paper claim

  Paper claim:

  - CSV2 is a validated evaluator for intervention science.
  - It measures not just whether harmful compliance happened, but where it appears, how direct it is, how operational it is,
    and how interventions shift those properties.

  Paper anti-claim:

  - It is not “the new universal jailbreak benchmark.”
  - It is not “better because it has more dimensions.”

  ### 2. Refresh the canonical evidence base

  - Re-run CSV2 v3 on the canonical jailbreak alpha sweep and D7 conditions so the main analyses use true v3 annotations,
    not legacy normalization.
  - Treat all prior legacy-normalized summaries as exploratory only.
  - Keep the existing schema stable during this phase; do not add new top-level fields unless validation exposes a concrete
    failure mode.

  ### 3. Upgrade the gold set, reusing the current validation spine

  - Extend the existing gold-label workflow in [scripts/validate_evaluator_gold.py](/home/hugo/Documents/Engineering/mech-
    interp/lab/02-h-neurons/scripts/validate_evaluator_gold.py) from binary labels to CSV2 v3 fields.
  - Build two audited sets:
      - Set A: relabel the existing 75 cross-alpha full-response cases with full v3 annotations.
      - Set B: add 125 new stratified cases from canonical + D7 outputs, oversampling borderline-looking disclaimers,
        pivots, and span-resolution failures.
  - Human annotation protocol:
      - 3 annotators per item.
      - Annotate primary_outcome, intent_match, C, S, V, wrapper tags, harmful/refusal spans, and pivot.
      - Adjudicate to a final gold label.
  - Acceptance defaults:
      - ordinal/categorical agreement target: >= 0.65
      - span overlap target: >= 0.60
      - if a field misses target, demote it from headline claim to exploratory analysis.

  ### 4. Add a narrow task-aware anchor layer

  - For only the audited subset, add a compact reference spec per prompt:
      - harmful objective
      - 2-5 “minimum harmful elements” that count as genuine task fulfillment
  - Use this to audit whether intent_match, S, and V are faithful on tricky cases.
  - This is the part to borrow from GuidedBench/JADES.
  - Do not build a full decompositional multi-agent evaluator.

  ### 5. Run judge robustness the way current best practice now demands

  - Extend the gold validation harness to a CSV2 robustness harness.
  - Evaluate the same audited responses under:
      - 3 judge prompts: current rubric, rubric + examples, rubric + anchored references
      - 2 judge models: current frontier judge plus one cheaper alternative
      - 3 perturbations: formatting changes, disclaimer padding, harmless verbosity
  - Report:
      - agreement to human gold per field
      - sign stability of intervention deltas
      - field-level fragility by perturbation type
  - If a metric flips sign under modest judge/style changes, it cannot be a headline paper result.

  ### 6. Run the intervention case study that the current plan is aiming at

  Primary case study:

  - baseline vs L1 vs causal on the D7/full500 canonical setup
  - plus random-head negative controls

  Primary outcomes to report:

  - binary harmful rate
  - primary_outcome
  - intent_match
  - C, S, V
  - harmful_payload_share
  - pivot_position
  - wrapper tag distributions

  Success condition:

  - show at least one intervention comparison where CSV2 reveals a clear structural/severity shift that binary harmfulness
    understates or misses.

  ### 7. Add the right auxiliary evaluations

  - Over-refusal / calibration:
      - XSTest
      - OR-Bench
  - Benign utility:
      - one small instruction-following or helpfulness set
  - Keep this lightweight. The point is “safer without obviously breaking the model,” not a giant capability paper.

  ### 8. Add one transfer slice only after validation

  Default:

  - 100-150 prompts from StrongREJECT, because it is common in intervention/safety papers and already close to the current
    ecosystem.
  - Use this to test whether CSV2’s intervention-sensitive metrics still behave sensibly off-benchmark.
  - Compare against StrongREJECT on the same responses, but frame this as complementarity, not leaderboard competition.

  ## Test Plan

  - Unit tests stay focused on schema normalization, span resolution, error handling, and paired reporting.
  - New validation outputs must include:
      - human agreement tables
      - judge-robustness tables
      - per-field confusion/error analysis
      - example audit trails for at least 6 representative cases
  - Publication acceptance bar:
      - fresh v3 annotations on canonical data
      - audited gold with adjudication
      - judge stress test completed
      - negative control completed
      - over-refusal check completed

  ## Assumptions and defaults

  - Default framing: intervention-first methods paper with a mechanistic case study.
  - Default external comparator: StrongREJECT.
  - Default task-aware anchoring: audited subset only.
  - Default near-term goal: publishable arXiv/ACL Findings level contribution, not a universal benchmark paper.
  - Default interpretation rule: if a proposed addition does not improve intervention sensitivity, robustness, or
    auditability, it is out of scope.

  ## Sources

  - StrongREJECT: https://arxiv.org/abs/2402.10260
  - GuidedBench: https://arxiv.org/abs/2502.16903
  - JADES: https://arxiv.org/abs/2508.20848
  - Know Thy Judge: https://arxiv.org/abs/2503.04474
  - A Coin Flip for Safety: https://arxiv.org/abs/2603.06594
  - SafeNeuron: https://arxiv.org/abs/2602.12158
  - XSTest: https://arxiv.org/abs/2308.01263
  - OR-Bench: https://arxiv.org/abs/2405.20947







---

# Plan: Turn CSV2 into a Publication-Worthy Contribution

## Context

Your mentor asked you to consult the literature to find what's genuinely novel about CSV2 and what leverage it creates for other AI safety researchers. After a deep literature review across Semantic Scholar, OpenAlex, and arXiv (60+ papers examined), the answer is clear: **the jailbreak evaluation field is entirely attack-centric, and no published framework exists for measuring how safety interventions change the structure and severity of harmful responses.** CSV2 fills that gap.

The ChatGPT report (`safety-evals-research.md`) is shallow because it treats CSV2 as "yet another jailbreak scorer that needs more dimensions." That misses the point entirely. CSV2's novelty isn't in having more dimensions — it's in being designed for a fundamentally different question: not "did the attack succeed?" but "how did the intervention change the attack's character?"

---

## I. The Literature Landscape (What Exists)

### A. Graded Jailbreak Evaluation (Attack-Centric)

| Framework | Year | Scoring | Structural Analysis | Intervention-Aware |
|-----------|------|---------|--------------------|--------------------|
| **StrongREJECT** (2402.10260) | 2024 | 3D continuous: (1-refused) × (specific + convincing)/2 | None (document-level) | No |
| **JADES** (2508.20848) | 2025 | Weighted sub-question decomposition → ternary | Cleans distractors, but no spans | No |
| **GuidedBench** | 2025 | Per-question entity/function scoring points | None | No |
| **Rethinking Eval** (2404.06407) | 2024 | 3 binary metrics (SV, I, RT) → 8-state | Paragraph/sentence tokenization | No |
| **HarmBench** (2402.04249) | 2024 | Binary (functional categories) | None | No |
| **JailbreakBench** (2404.01318) | 2024 | Binary ASR | None | No |
| **JAILJUDGE** | 2024 | 1-10 scale, multi-agent | None | No |
| **AttackEval** | 2024 | Continuous 0-1 | None | No |
| **SORRY-Bench** (2406.14598) | 2024 | Fine-grained refusal categories | None | No |

### B. Over-Refusal Measurement

OR-Bench, DUAL-Bench, Health-ORSC-Bench, OVERT, FalseReject — all measure false positive rates but don't score harmful content structure or intervention effects.

### C. Mechanistic Safety + Evaluation

Safety Arithmetic (2406.11801), Hidden Dimensions of Alignment (2502.09674), From Refusal Tokens to Refusal Control (2603.13359) — all study *mechanisms* of safety directions but evaluate interventions with binary ASR or simple accuracy. None use structured, multi-dimensional evaluation of the *output quality*.

### D. Span-Level Analysis

Target Span Detection (2403.19836) does span-level detection of implicit harmful content — but for classification (is this text harmful?), not for measuring intervention effects. No character-level span identification for evaluating how interventions change harm distribution within responses.

### E. Rubric-Based LLM Evaluation (General)

Autorubric (2603.00077), RULERS (2601.xxxx), RubricEval (2603.25133), Rubric-Conditioned Grading (2601.08843) — general rubric evaluation frameworks, not safety-specific. They validate that rubric-based LLM judging works but don't address the safety domain.

---

## II. The Gap (Precisely Stated)

**No published framework combines:**

1. **Multi-axis ordinal scoring** where axes measure orthogonal facets of harm (alignment, actionability, operational validity) rather than collapsing everything into a single score
2. **Wrapper tag decomposition** tracking linguistic evasion strategies (disclaimers, persona masks, fiction laundering, pivots)
3. **Character-level harmful span identification** enabling structural metrics (where harm appears, how concentrated it is)
4. **Intervention-centric design** with dose-response curves, paired comparisons, and negative controls

StrongREJECT comes closest (3 dimensions, continuous score) but has no span analysis, no wrapper tags, no structural metrics, no intervention awareness, and collapses its axes into a single number.

JADES comes closest on decomposition but decomposes by *sub-question* (what did the attacker want?), not by *harm facet* (how committed, how specific, how valid?). No spans, no wrapper tags.

Rethinking (2404.06407) comes closest on structure (hierarchical tokenization) but at paragraph/sentence level, not character-level, and its metrics are still binary per dimension.

**The gap: the field has no evaluation tool designed for the intervention researcher who needs to know not just "did harm decrease?" but "how did the character of harm change?"**

---

## III. CSV2's Novel Contributions (5 Claims)

### Claim 1: Three Orthogonal Harm Facets
C (Commitment 0-3), S (Specificity 0-4), V (Validity 0-3) capture independent dimensions. StrongREJECT conflates specificity and convincingness into a product. JADES weights sub-questions. Nobody separates "does the model want to help the attacker?" from "how actionable is the help?" from "would it actually work?"

**Validation needed:** Empirical independence (inter-axis correlation analysis).

### Claim 2: Wrapper Tag Taxonomy
R (refusal), D (disclaimer 0-2), P (persona), F (fiction/educational), T (pivot) — no other framework tracks these. This captures *how models disguise harm*, not just whether harm exists. Enables analysis of:
- Tag co-occurrence patterns (F+T = fiction laundering + pivot is a distinct evasion strategy)
- How interventions shift evasion strategies (does steering make models disclaim more rather than refuse?)

**Validation needed:** Show tag distributions change systematically with intervention strength.

### Claim 3: Span-Resolved Structural Metrics
Character-level harmful spans via fuzzy quote matching enable:
- `pivot_position` (0→1): Where harm first appears. Distinguishes "immediate compliance" from "deceptive structure"
- `harmful_payload_share` (0→1): Concentration of harm. Distinguishes "dense payload" from "diluted with disclaimers"

**No other framework produces these metrics.** Rethinking (2404.06407) has paragraph-level tokenization but derives no structural metrics from it.

**Validation needed:** Show these metrics capture real structural differences (examples + distribution analysis).

### Claim 4: Intervention-Centric Design
CSV2 is parameterized by intervention strength (α). It enables:
- Dose-response curves (harm severity as a function of α)
- Paired comparisons with bootstrap CIs (same prompt, different interventions)
- Negative controls (random neuron sets vs. targeted neurons)
- Template and category stratification

Every other framework is designed for "run N attacks, compute ASR." CSV2 is designed for "apply intervention at K strengths, measure how response structure changes."

**Validation needed:** Power analysis showing CSV2 detects effects binary measures miss.

### Claim 5: Task-Family-Aware Validity
V-axis scoring adapts to domain: cyber requires working code, persuasion requires convincing arguments, fraud requires realistic pretexts. This grounds validity in operational reality rather than abstract "convincingness."

**Validation needed:** Show cross-domain V distributions differ meaningfully.

---

## IV. Implementation Plan: From Current State to Submission

### Current State (from data audit)
- 500-sample full run with baseline, L1, and causal intervention — CSV2-evaluated
- Causal intervention: -9.0pp csv2_yes [CI: -12.2, -5.8], significant across all axes
- Pipeline fully operational (evaluate_csv2.py, analyze_csv2.py)
- StrongREJECT 313, HarmBench 159, JBB 100 prompt sets already materialized

### Phase 1: Validation Experiments (establishes CSV2 as a credible evaluation tool)

**1.1 Inter-Rater Reliability (CRITICAL — reviewers will demand this)**
- Sample 100 responses stratified across C/S/V levels and wrapper tag combinations
- Human-annotate on same rubric (you + 1-2 annotators, or crowd workers)
- Compute Krippendorff's α per axis (C, S, V, harmful_binary) and per wrapper tag
- Target: α > 0.7 for ordinal axes, α > 0.8 for binary tags
- Also compute GPT-4o ↔ human agreement (treat GPT-4o as "annotator 3")
- **File:** New script `scripts/validate_irr.py`

**1.2 Axis Independence**
- Compute pairwise Spearman correlations between C, S, V across all evaluated responses
- If ρ(C,S) < 0.7 and ρ(S,V) < 0.7 and ρ(C,V) < 0.7 → axes are usefully independent
- Visualize as scatter matrices with marginal distributions
- **File:** Add to `scripts/analyze_csv2.py` or new `scripts/validate_axes.py`

**1.3 StrongREJECT Comparison**
- Run StrongREJECT scorer on same 1500 responses (baseline + L1 + causal)
- Run binary judge (already have this data)
- Compare: which metrics detect the causal intervention effect?
- Show: CSV2 ordinal axes detect effects StrongREJECT continuous score misses (or captures more precisely)
- Compute mutual information between CSV2 axes and StrongREJECT scores
- **File:** New script `scripts/compare_strongreject.py`

**1.4 Statistical Power Analysis**
- Bootstrap the existing 500-sample data to estimate MDE for:
  - Binary (csv2_yes/no) — current MDE ~6pp
  - StrongREJECT continuous score
  - CSV2 mean C, mean S, mean V (paired)
  - CSV2 harmful_payload_share (paired)
- Show CSV2 ordinal metrics have smaller MDE (more sensitive)
- **File:** New script `scripts/power_analysis.py`

### Phase 2: Additional Experimental Evidence

**2.1 Random-Head Negative Control (already identified as missing)**
- Run causal intervention with random neuron sets (3-5 seeds) at α=4.0 on full 500
- CSV2-evaluate all
- Compare severity slopes: if H-neuron slope > all random slopes → selector specificity
- **Files:** Pipeline script in `scripts/`, data in `data/gemma3_4b/intervention/jailbreak/control/`

**2.2 Capability Mini-Battery**
- Run IFEval (instruction-following) on baseline and causal-locked at α=4.0
- Measure: does intervention degrade general capabilities?
- This answers "safer, not broken" — essential for publication
- **File:** New script `scripts/eval_capability.py`

**2.3 Cross-Benchmark Generalization**
- Run causal intervention on StrongREJECT 313 prompts (already materialized at `data/contrastive/refusal/eval/strongreject_313.jsonl`)
- CSV2-evaluate
- Show effect generalizes beyond JBB
- **File:** Pipeline modifications to support alternate prompt sets

**2.4 Wrapper Tag Dose-Response**
- Analyze wrapper tag distributions across α values (from early 4-alpha run: 2000 rows)
- Show: as α increases, do models shift from full compliance (C=3) to disclaimed compliance (C=2, D=2)?
- Plot tag co-occurrence matrices at each α level
- **File:** Extend `scripts/analyze_csv2.py`

### Phase 3: Paper Writing

**3.1 Framing (the pitch)**
Title suggestion: *"CSV2: A Span-Resolved, Multi-Axis Evaluation Framework for Measuring Safety Intervention Effects on Harmful LLM Responses"*

Structure:
1. **Introduction**: Safety interventions are a growing field but evaluated with attack-centric binary tools. This loses crucial information.
2. **Related Work**: The table from Section I above. Position CSV2 relative to StrongREJECT (closest competitor), JADES (closest on decomposition), Rethinking (closest on structure).
3. **The CSV2 Framework**: Rubric (C, S, V), wrapper tags, span resolution, structural metrics. Design rationale grounded in intervention evaluation needs.
4. **Validation**: IRR (Section 1.1), axis independence (1.2), StrongREJECT comparison (1.3), power analysis (1.4).
5. **Case Study**: H-neuron intervention on Gemma3-4B. Dose-response, random-head control, template/category stratification. Show CSV2 reveals structural changes invisible to binary/continuous metrics.
6. **Discussion**: Limitations (single model, GPT-4o judge dependence), future work (multi-model, human-in-the-loop judging, multi-turn).
7. **Release**: Open-source framework, data, and prompts.

**3.2 Key Figures**
- Fig 1: CSV2 rubric architecture diagram (C/S/V axes, wrapper tags, span resolution pipeline)
- Fig 2: Axis independence scatter matrix
- Fig 3: Dose-response curves for C, S, V, payload_share across α values
- Fig 4: StrongREJECT vs CSV2 sensitivity comparison (violin plots of effect sizes)
- Fig 5: Wrapper tag co-occurrence heatmaps at different α levels
- Fig 6: Structural metrics (pivot_position, payload_share) distributions by intervention condition
- Fig 7: Random-head control comparison

**3.3 Target Venues**
- **Primary**: ACL/EMNLP (Findings or main) — NLP safety evaluation is a growing track
- **Alternative**: NAACL, AAAI (safety/alignment track), or a safety-specific workshop (SafeGenAI, TrustNLP)
- **Preprint**: arXiv cs.CL + cs.AI

### Phase 4: Open-Source Release

**4.1 Framework Package**
- Clean `evaluate_csv2.py` into importable module
- JSON schema for CSV2 annotations
- Example notebooks showing: basic evaluation, intervention comparison, structural analysis
- README with quick-start, rubric documentation, API reference

**4.2 Benchmark Data Release**
- Anonymized response data with CSV2 annotations
- Prompt sets (JBB, StrongREJECT, HarmBench subsets)
- Human annotation data (from IRR study)

---

## V. What Makes This a Genuine Contribution (Not Just "More Dimensions")

The ChatGPT report's fundamental error was treating CSV2 as an incremental improvement (add severity levels + intent categories to existing binary evaluation). That's a feature request, not a research contribution.

CSV2's actual contribution is **changing the question being asked**:

| Existing frameworks ask | CSV2 asks |
|------------------------|-----------|
| Did the attack succeed? | How did the intervention change the attack's character? |
| What's the ASR? | How did commitment, specificity, and validity shift independently? |
| Pass or fail? | Where in the response does harm appear, and how concentrated is it? |
| Was the response harmful? | What linguistic evasion strategy did the model use? |
| How effective was this jailbreak? | At what intervention strength does harm structure change? |

This is novel because **the intervention research community has no evaluation tool designed for their needs.** Every paper on activation steering, refusal direction ablation, safety neurons, or neuron-level manipulation evaluates with binary ASR or at best StrongREJECT's continuous score. CSV2 gives them:
- Structural metrics that reveal HOW interventions work (not just IF they work)
- Dose-response sensitivity that binary metrics cannot provide
- Wrapper tag analysis showing whether interventions improve refusal or just change evasion strategy
- Span-level evidence that's auditable and supports manual review

---

## VI. Verification Plan

1. **IRR study**: Run on 100 stratified samples → Krippendorff's α > 0.7 per axis
2. **Axis independence**: Pairwise ρ < 0.7 across C, S, V
3. **StrongREJECT comparison**: CSV2 detects causal effect with smaller CI than StrongREJECT score
4. **Power analysis**: CSV2 ordinal MDE < binary MDE at same sample size
5. **Random-head control**: H-neuron severity slope > random severity slope (all 3-5 seeds)
6. **Capability**: IFEval score within 5% of baseline at α=4.0
7. **Cross-benchmark**: Effect replicates on StrongREJECT 313 prompt set
8. **Wrapper tags**: At least 2 tag types show monotonic dose-response with α

---

## VII. Critical Files

| File | Role |
|------|------|
| `scripts/evaluate_csv2.py` | Core evaluation pipeline (rubric prompt, judge harness, span resolution) |
| `scripts/analyze_csv2.py` | Post-evaluation analysis (10-section report) |
| `scripts/analyze_csv2_control.py` | Negative control comparison |
| `scripts/report_d7_csv2.py` | Paired reporting with bootstrap CIs |
| `data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/` | Primary experimental data |
| `data/contrastive/refusal/eval/strongreject_313.jsonl` | StrongREJECT prompt set (materialized) |
| `data/contrastive/refusal/eval/harmbench_test_159.jsonl` | HarmBench prompt set (materialized) |
| `data/benchmarks/GuidedBench_*.md` | GuidedBench paper (reference) |
| `data/benchmarks/JADES*.md` | JADES paper (reference) |

## VIII. Papers to Cite (Priority)

**Direct comparisons (must cite):**
- StrongREJECT (Souly et al., 2024) — arXiv:2402.10260
- JADES (Chu et al., 2025) — arXiv:2508.20848
- Rethinking Jailbreak Eval (Cai et al., 2024) — arXiv:2404.06407
- GuidedBench (2025)
- HarmBench (Mazeika et al., 2024) — arXiv:2402.04249
- JailbreakBench (Chao et al., 2024) — arXiv:2404.01318

**Intervention evaluation context (must cite):**
- Refusal in LLMs Is Mediated by a Single Direction (Arditi et al., 2024) — arXiv:2406.11717
- There Is More to Refusal (Joad et al., 2026) — arXiv:2602.02132
- ITI (Li et al., 2023) — arXiv:2306.03341
- H-Neurons paper (your base paper)
- Safety Neurons (2024) — arXiv:2406.14144
- SafeNeuron (2026) — arXiv:2602.12158
- Hidden Dimensions of LLM Alignment (Pan et al., 2025) — arXiv:2502.09674

**Methodological foundations:**
- SORRY-Bench (Xie et al., 2024) — arXiv:2406.14598
- JAILJUDGE (Liu et al., 2024)
- OR-Bench (2024) — arXiv:2405.20947
- Target Span Detection (Jafari et al., 2024) — arXiv:2403.19836
- Know Thy Judge (Eiras et al., 2025) — arXiv:2503.04474
- WildGuard (Han et al., 2024) — arXiv:2406.18495

**Over-refusal (for dual-metric discussion):**
- DUAL-Bench (2025) — arXiv:2510.10846
- Health-ORSC-Bench (2026) — arXiv:2601.17642
