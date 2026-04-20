# D4 naming gap and FaithEval anti-compliance audit against ground-truth data

**Date:** 2026-04-20  
**Reviewer stance:** adversarial. No prior report is authoritative; numbers below were recomputed from raw artifacts or verified directly against source code.  
**Primary question:** what, exactly, is missing from the current ICML paper on the D4 / FaithEval side, and what is only mislabeled or under-disclosed?

## Executive verdicts

1. **The complaint "there is 0 mention of D4" is only partly true.** The current ICML draft contains the strong held-out D4 ITI result numerically, but never identifies it as the `D4` branch. `paper/icml/main.tex` contains no `D4` string, yet lines 213-214 already report the canonical D4 TruthfulQA MC gain: about `26.7% -> 33.0%` on MC1 and `42.9% -> 50.4%` on MC2 at `alpha = 8.0`.
2. **A separate on-disk D4 artifact really is omitted:** the residual-stream truthfulness-direction ablation on FaithEval anti-compliance. That run reaches `66.0% -> 71.4%` at `beta = 0.01` on the same 1,000-item surface as the H-neuron anchor. Omitting it is methodologically defensible, but only if the paper says why.
3. **The anti-compliance benchmark contract remains under-disclosed in the main text.** On this harness, `compliance = chose the misleading answer despite being told to answer from own knowledge`. That is not a labeling footnote; it changes the sign of the interpretation.
4. **The standard-vs-anti-compliance difference is empirically large, not semantic.** The anti-compliance H-neuron run rises by `+6.3 pp` from `alpha 0 -> 3`; the standard-prompt run falls by `-5.5 pp` over the same range, while parse failures rise from `0.9%` to `15.0%`.

This file now covers only D4 and FaithEval-contract issues. D7 is handled separately in `2026-04-20-d7-quality-debt-adversarial-audit.md`.

---

## 1. Ground truth and disambiguation

The repository uses `D4` in **two different ways**, and the older version of this audit conflated them.

### 1.1 D4 meaning A: the strong act-3 result that the paper already uses

Later sprint docs use `D4` for the **paper-faithful ITI head intervention**:

- `notes/act3-reports/2026-04-01-priority-reruns-audit.md`
- `notes/act3-reports/research-log-iti-artifact-exploration.md`

Canonical raw artifacts:

- `data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold0-iti-heads_7723b7d6d7/experiment`
- `data/gemma3_4b/intervention/truthfulqa_mc_mc1_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold1-iti-heads_9a10b5307d/experiment`
- `data/gemma3_4b/intervention/truthfulqa_mc_mc2_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold0-iti-heads_7723b7d6d7/experiment`
- `data/gemma3_4b/intervention/truthfulqa_mc_mc2_iti-head_truthfulqa-paperfaithful_k-12_ranked_seed-42_final-fold1-iti-heads_9a10b5307d/experiment`

Independent recomputation with `scripts/report_iti_2fold.py` against those four directories gives:

| Variant | Baseline | Intervened | Paired delta |
|---|---:|---:|---:|
| MC1 | 0.2672 | 0.3298 | `+6.26 pp` `[+3.66, +9.01]` |
| MC2 | 0.4286 | 0.5036 | `+7.49 pp` `[+5.28, +9.72]` |

These are the strong D4 numbers the user is pointing to. They are **already in the paper numerically**:

- `paper/icml/main.tex:213-214`

What is missing is the **branch identity / provenance**:

- `rg -n "\bD4\b" paper/icml/main.tex` returns no matches.

So the correct statement is:

> The current paper omits the `D4` branch label and act-3 provenance, but not the strong D4 ITI result itself.

### 1.2 D4 meaning B: the residual-stream FaithEval direction ablation

Earlier March notes also call the residual-stream truthfulness-direction work `D4`.

Canonical raw artifacts:

- `data/gemma3_4b/intervention/faitheval_direction_ablate_d4_all_layers_calibrated_clean/experiment/alpha_*.jsonl`

That run is **not** in the current ICML draft. It is a real omission, but it is **not** the same thing as the strong held-out TruthfulQA MC result above.

The distinction matters. The existing version of this audit was too aggressive because it treated meaning B as the main missing `D4`, when the stronger act-3 `D4` result is already present numerically.

---

## 2. What the current paper already includes from D4

The current ICML draft already reports the D4 ITI branch's main positive result and its transfer failure:

- TruthfulQA MC gain at `alpha = 8.0`:
  - `paper/icml/main.tex:213-214`
- SimpleQA / TriviaQA generation harm:
  - `paper/icml/main.tex:217-228`

This means the present paper is **not** omitting the strongest act-3 D4 evidence on the merits. What it omits is:

1. the `D4` name,
2. the act-3 branch identity,
3. the fact that this ITI branch was one of the repo's clearest positive truthfulness results on a clean held-out benchmark.

### 2.1 Why that still matters

Internally, `D4` is a major branch in the act-3 investigation tree. In the paper, the result appears only as a generic "ITI with TruthfulQA-derived truthfulness directions" result. That is fine for an external reader, but it weakens internal continuity and makes it easier for the paper to sound as if:

- FaithEval H-neurons are the only clean positive control story, and
- the ITI branch exists mainly as an externality failure.

That is incomplete. The truthful version is:

> D4/ITI is a real positive result on TruthfulQA MC and a real negative result on nearby generation surfaces.

The current paper conveys the second half clearly and the first half numerically, but not as a named branch-level takeaway.

---

## 3. The actually omitted D4 comparator: residual-stream FaithEval ablation

This is the genuinely omitted on-disk comparator.

### 3.1 Recomputed FaithEval numbers

From raw JSONL rows in:

- `data/gemma3_4b/intervention/faitheval_direction_ablate_d4_all_layers_calibrated_clean/experiment/`
- `data/gemma3_4b/intervention/faitheval/experiment/`
- `data/gemma3_4b/intervention/faitheval_sae/experiment/`

All on the anti-compliance prompt surface:

| Method | Operating point | Compliance | Δ vs no-op |
|---|---:|---:|---:|
| H-neurons | `alpha = 0.0` | 64.2% | — |
| H-neurons | `alpha = 3.0` | 70.5% | `+6.3 pp` |
| SAE full replacement | `alpha = 0.0` | 72.3% | — |
| SAE full replacement | `alpha = 3.0` | 69.9% | `-2.4 pp` |
| D4 residual-stream ablation | `beta = 0.00` | 66.0% | — |
| D4 residual-stream ablation | `beta = 0.01` | 71.4% | `+5.4 pp` |
| D4 residual-stream ablation | `beta = 0.02` | 46.2% | `-19.8 pp` |

The independently recomputed matched FaithEval slope anchor still stands:

- H-neurons: `+2.09 pp / alpha`
- SAE full replacement: `+0.16 pp / alpha`
- neuron-minus-SAE gap: `+1.93 pp / alpha` `[+0.94, +2.92]`

### 3.2 Why this omitted comparator is real but not anchor-worthy

The residual-stream D4 point at `beta = 0.01` is numerically relevant, but it is not methodologically on par with the H-neuron anchor because:

1. **No matched random-direction control on FaithEval.**  
   The H-neuron claim is supported by 8 random-neuron controls; the SAE claim has random-feature controls. This D4 FaithEval run does not have the equivalent matched random-direction specificity closure on this surface.
2. **No smooth dose-response.**  
   `beta = 0.01` is usable; `beta = 0.02` is already a corruption cliff. This is a narrow operating window, not the kind of smooth control curve the paper is using as its anchor geometry.
3. **Different detector lineage.**  
   The residual-stream truthfulness direction was not selected by a FaithEval-matched readout comparable to the H-neuron / SAE setup in `§3`.

So the right conclusion is:

> The omitted residual-stream D4 FaithEval result should be acknowledged as an on-disk counterexample to any naive "direction families do not steer here" inference, but it should not replace the H-neuron-vs-SAE anchor.

The older version of this audit was right that the comparator exists, but it overstated what kind of omission it is.

---

## 4. The anti-compliance prompt is still the bigger paper-facing issue

### 4.1 Code-verified task definition

From `scripts/run_intervention.py`:

- the prompt includes:  
  `If the context conflicts with established knowledge, answer based on your own knowledge.`
- scoring is:  
  `is_compliant = chosen == sample["counterfactual_key"]`

And `counterfactual_key` is the misleading answer.

So on this harness:

> `compliance = model chose the misleading answer despite an explicit anti-compliance instruction`

That is the exact opposite of how a casual reader will interpret "compliance" in the abstract unless told otherwise.

### 4.2 Empirical size of the contract difference

Independent row-level recomputation from raw JSONL:

| Surface | Alpha 0 | Alpha 3 | Δ | Parse failures at alpha 3 |
|---|---:|---:|---:|---:|
| H-neuron anti-compliance | 64.2% | 70.5% | `+6.3 pp` `[+4.1, +8.5]` | 0.0% |
| H-neuron standard prompt | 69.1% | 63.6% | `-5.5 pp` `[-8.2, -2.8]` | 15.0% |

This is not a minor wording issue. The same intervention family:

- **amplifies misleading-context acceptance** on the anti-compliance harness,
- but **reduces measured compliance** on the standard harness,
- while parse reliability collapses on the standard harness at high alpha.

So the current paper is justified in treating the anti-compliance surface as its clean matched benchmark, but it is **not** justified in speaking as if "FaithEval compliance dose-response" is self-explanatory.

### 4.3 What the current paper gets right and wrong

Right:

- `paper/icml/main.tex:115` says FaithEval is an anti-compliance surface.
- Appendix construct language already says this is a credulity lever, not standard truthfulness.

Wrong or too soft:

- The abstract's "only H-neurons produce a reliable compliance dose-response" still reads like a positive steering success unless the reader already knows this benchmark contract.
- `§3.3` says "H-neurons did steer FaithEval compliance" without telling the reader in the same paragraph that this means amplifying the prompted-against failure mode.

The old version of this audit was correct on the disclosure problem. That remains the main blocker on this file.

---

## 5. Critique of prior notes

### 5.1 What the older version of this audit got wrong

The previous revision of this file overstated the D4 omission by assuming the omitted `D4` was the residual-stream FaithEval ablation. That missed the more important correction:

- the strong act-3 D4 ITI result is **already in the paper numerically**,
- while the residual-stream FaithEval D4 point is a **secondary omitted comparator**.

That distinction has to be explicit.

### 5.2 `notes/act3-reports/2026-03-31-faitheval-task-definition-audit.md`

Still correct:

- the contract mismatch is real,
- the right relabel is from "truthfulness improvement" to an anti-compliance / credulity measure.

Too soft:

- it treated the issue as something to route around later rather than something that should immediately constrain paper wording.

### 5.3 `notes/act3-reports/2026-03-30-d4-truthfulness-direction.md`

Still correct:

- `beta = 0.01` is the only usable residual-stream point,
- `beta = 0.02` is the cliff.

Too self-protective:

- "D4 survives the kill-shot" is a branch-preservation framing, not a paper-facing framing.

### 5.4 `notes/act3-reports/2026-04-01-priority-reruns-audit.md`

This is still the right source of truth for the strong act-3 D4 ITI result:

- D4 ITI on TruthfulQA MC is a real positive result.
- D1 vs D4 ranking on the clean held-out TruthfulQA MC axis is resolved in D4's favor.

That is the source chain the paper is already drawing from, even though it does not call it `D4`.

---

## 6. Required paper changes

### 6.1 Must-fix wording

1. **Abstract / `§3`: make the FaithEval sign explicit.**  
   Replace generic "compliance dose-response" wording with language like "acceptance of misleading contextual claims under an anti-compliance instruction."
2. **Name the branch provenance for the ITI result at least once.**  
   The paper does not need act-3 jargon everywhere, but one sentence should make clear that the TruthfulQA MC gain is the paper-faithful ITI branch that later fails to transfer.

### 6.2 Strongly recommended but optional

3. **Acknowledge the omitted residual-stream D4 FaithEval comparator in limitations or synthesis.**  
   One sentence is enough: it exists, it moves the anti-compliance surface, and it is excluded from the anchor comparison because it lacks matched specificity controls and has an immediate corruption cliff.

---

## Bottom line

The current paper is **not** suppressing the strongest act-3 D4 result on the merits. It is already using that result; it is just not naming it as `D4`.

The paper **is** still under-disclosing the FaithEval contract, and that matters more than the D4 naming issue. The benchmark is a credulity lever with an inverted sign, and the standard-prompt run shows that the contract change is large enough to flip the empirical direction.

The residual-stream D4 FaithEval ablation is a real omitted comparator, but it is a secondary omission with defensible methodological reasons, not the main "solid D4 result" the paper forgot.
