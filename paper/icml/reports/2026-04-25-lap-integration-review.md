# LAP Integration Review — Predicting Where Steering Vectors Succeed (Billa, 2026)

> Date: 2026-04-25
> Author: agent (oracle-assisted synthesis)
> Status: Advisory. No paper edits applied; this report is the audit trail for any subsequent integration.
> Source paper: `papers/2604.15557v1.pdf` / `papers/2604.15557v1.md` (arXiv:2604.15557v1, 2026-04-16)
> Governing framing rule: `notes/2026-04-21-claim-framing-governance.md` — route by question, do not promote any single result to project-wide supremacy, preserve the four-stage audit scaffold.

## 1. What LAP is

The Linear Accessibility Profile (LAP) is a training-free, per-layer diagnostic that **repurposes the logit lens as a predictor of steering vector effectiveness**. Its key measure is

$$A_{\mathrm{lin}}(\ell) = \text{argmax accuracy of } W_U \cdot \mathrm{LayerNorm}(h_\ell) \text{ on a single-token concept family},$$

complemented by a probe gap $\Delta(\ell) = A_{\mathrm{mlp}}(\ell) - A_{\mathrm{lin}}(\ell)$ and a perturbation sensitivity $\lambda(\ell)$.

Headline empirical claims:

- Across 24 controlled binary concept families on five models (Pythia-2.8B → Llama-8B), peak $A_{\mathrm{lin}}$ predicts steering effectiveness at $\rho = +0.86$ to $+0.91$ and layer selection at $\rho = +0.63$ to $+0.92$.
- A **three-regime framework**:
  1. $A_{\mathrm{mlp}}$ low → concept absent → no method works;
  2. $A_{\mathrm{mlp}}$ high but $A_{\mathrm{lin}}$ low → concept present but nonlinearly encoded → difference-of-means fails, **SAE-style methods may help** (regime 2 explicitly flagged as "where SAE features should be most useful");
  3. $A_{\mathrm{lin}}$ high → output-aligned → DoM works.
- An end-to-end demo (Gemma-2-2B, OLMo-2-1B-Instruct) shows that the LAP-recommended layer redirects entity completions while the standard middle-layer heuristic has no effect.

Validation setting is narrow: **single-token next-token completion**, mostly **single-layer DoM steering**, factual concept families (geography, arithmetic, sequence, word transform, analogy, plus 25 controlled binaries).

## 2. Most defensible relevance to our ICML paper

Our paper is a single-model Gemma-3-4B-IT case study whose central messages are:

- **Localization → control gap**: matched held-out AUROC (H-neurons 0.843, SAE 0.848) does NOT yield matched compliance dose-response on FaithEval (neuron-minus-SAE slope $+1.93$ pp/$\alpha$, CI $[+0.94, +2.92]$).
- **Control → externality**: ITI gains on TruthfulQA MC harm TriviaQA bridge accuracy via wrong-entity substitution; SimpleQA non-attempt rate rises.
- **Measurement → conclusion**: truncation and scoring granularity move the jailbreak verdict; rubric audit exposes residual evaluator dependence.
- **Synthesis**: a four-stage audit scaffold (measurement, localization, control, externality).

Routing LAP against this:

| Where LAP touches our claims | Reading |
|---|---|
| **Corroborates** | LAP independently rejects "good readout ⇒ good steering": even at high probe accuracy, steering can fail when the target is not aligned with the model's own output projection. This sharpens our §3 negative finding about matched AUROC. |
| **Complicates** | LAP's regime-2 prediction is that **SAE-style methods should be the rescue** in "concept present but nonlinearly encoded" settings. Naively cited, this gives reviewers the loose retort: *"your SAE null may just be a regime-2/wrong-layer artifact; better SAE steering should have worked."* |
| **Orthogonal** | LAP is silent on cross-surface externality (§4) and on measurement instability (§5). Treating it as paper-wide framing would inflate its role beyond its evidence base. |

The defensible reading is therefore **gate-local corroboration of localization → control**, not a paper-wide explanation, and not the explanation of the FaithEval dissociation.

## 3. Recommended integration — surgical, not structural

### 3.1 Related Work (single sentence)

Insertion point: after the existing concurrent-work paragraph (around `paper/icml/main.tex` line 136–141).

Draft:

> Concurrent work by \citet{billa2026lap} introduces the Linear Accessibility Profile (LAP), a training-free logit-lens diagnostic predicting when single-layer difference-of-means steering should succeed and at which layer. LAP emphasises output alignment rather than probe separability alone, complementing our localization$\to$control question, but is validated mainly on single-token next-token completion and does not address cross-representational comparisons, measurement audits, or cross-surface externalities.

### 3.2 §3 (Similar Readout Quality Does Not Guarantee Control) — append to Synthesis

Draft (preferred, conservative version):

> LAP sharpens a compatible point: high readout quality can coexist with poor steering when the selected target is not output-aligned at the intervention layer \citep{billa2026lap}. Our claim here remains empirical and narrower: in this Gemma-3-4B-IT FaithEval setting, matched held-out readout quality did not suffice to select equally effective steering targets across two representational families.

Rationale: keeps LAP as **post hoc refinement**, explicitly states it was not used for target selection, and prevents the result from collapsing into a "layer choice" story.

### 3.3 §4 (Control → Externality) — optional boundary sentence

Skip unless space permits. If included, use as a delimiter:

> This section asks a different question from diagnostics that predict whether a steering intervention will act at all at a chosen layer \citep{billa2026lap}: even when an intervention is behaviorally active on its source surface, it may remain surface-local or externalize harm on a nearby one.

### 3.4 §5 (Measurement Choices Changed the Conclusion) — do NOT cite

LAP is orthogonal here. Citing it would muddy the measurement claim and falsely imply LAP is part of the scaffold. Only consider one delimiting sentence if a reviewer explicitly invites it.

### 3.5 §6 (Four-Stage Audit Framework) — single sentence under R1

Draft (Option B, preferred):

> A useful refinement is to supplement held-out readout quality with layer-specific accessibility checks, such as whether the target is output-aligned at the intervention layer \citep{billa2026lap}. Even then, direct behavioral intervention tests remain necessary.

Rationale: subordinates LAP to the scaffold, frames it as a within-gate diagnostic, prevents framework displacement.

### 3.6 BibTeX

```
@misc{billa2026lap,
  title  = {Predicting Where Steering Vectors Succeed: The Linear Accessibility Profile},
  author = {Billa, Jayadev},
  year   = {2026},
  eprint = {2604.15557},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url    = {https://arxiv.org/abs/2604.15557}
}
```

## 4. The framing question: does LAP explain our SAE null?

**No, and adopting that reading would be a mistake.**

Safe reading:

> LAP offers a plausible within-gate hypothesis for some localization → control failures: a target may be detectable yet poorly aligned with the model's own output projection at the chosen layer.

Unsafe reading (do **not** put in the paper):

> "Our SAE null is a LAP regime-2 case; H-neurons are regime 3."

Reasons not to adopt the regime mapping:

1. **We did not measure $A_{\mathrm{lin}}$ on FaithEval at our intervention layers.** Asserting a regime is unsupported.
2. **Setting mismatch.** Our interventions are not LAP's validated class: not single-token next-token concepts, not single-layer additive DoM vectors, not pure neuron-vs-SAE basis comparisons under matched protocol.
3. **LAP's regime-2 narrative is sympathetic to SAEs.** Threading it through our paper hands reviewers the line "better SAE steering should have worked," which directly attacks our cross-representational headline.

Safest sentence (already in §3 draft above):

> LAP is consistent with the broader idea that output alignment matters for steerability, but our FaithEval result does not depend on that explanation and should not be collapsed to a layer-choice story.

## 5. Cheap follow-up experiments worth doing before ICML

Two tiers, in priority order.

### Tier 1 (recommended, ~0.5–1.5 days): FaithEval option-restricted accessibility curves

For each FaithEval item, at the final extraction position, take the residual stream at each layer of Gemma-3-4B-IT and apply the unembedding. Compute, on the **answer-option tokens only** (not full vocabulary):

- restricted-choice accuracy on the gold option;
- gold-minus-best-foil logit margin;
- rank of the gold option among valid choices;
- (optional) AUROC of compliant vs anti-compliant option margin across items.

**Why option-restricted, not full-vocab argmax**: FaithEval uses single-letter (A/B/C/D) extraction. Raw full-vocab argmax over letter tokens conflates concept accessibility with token-frequency artifacts. An option-restricted metric is the honest analogue of $A_{\mathrm{lin}}$ in this setting and pre-empts the obvious reviewer complaint.

**Reporting**: call it a "LAP-style accessibility analysis" or "output-alignment diagnostic", not a LAP reproduction.

**Outcomes are useful either way**:

- If H-neuron source layers cluster in higher-accessibility regions and SAE extraction layers do not → one paragraph + one figure in an appendix supports the localization → control gap with an output-alignment angle.
- If no such pattern → equally publishable, because it lets us state that output alignment does NOT obviously explain away the cross-representational dissociation, neutralising the regime-2 critique.

### Tier 2 (more work, only if Tier 1 motivates it): layerwise utility vs accessibility

- For H-neurons: scale only one source layer at a time (or leave-one-layer-out) and recompute the FaithEval slope.
- For SAEs: aggregate selected features by extraction layer, run delta-only validation interventions per layer.
- Plot layer utility against the Tier 1 accessibility curve.

This is closer to LAP's actual claim ("layer matters") but is appendix-scope at best.

### Do **not** before ICML

- Reproducing the full LAP stack ($A_{\mathrm{lin}} + A_{\mathrm{mlp}} + \lambda$).
- Tuned-lens variants.
- Assigning regimes.

These open more questions than they answer for this paper.

## 6. Pitfalls and reviewer-pushback guardrails

| Pitfall | Risk | Guardrail |
|---|---|---|
| "Why didn't you use LAP to pick the layer/target?" | Paper reframed as failed optimization. | Explicitly state: LAP is concurrent and post hoc; our paper audits what matched readouts buy under independently motivated interventions, not optimal-layer steering. |
| "Your SAE null is a regime-2 / wrong-layer artifact." | Undermines the cross-representational comparison. | Do not assign a LAP regime to our setting. Lean on the within-SAE selector ablation (readout vs utility selectors, opposite-sign margins). |
| Overextending LAP into §4 / §5. | Muddies the scaffold; makes LAP feel central. | Keep LAP in Related Work, §3, §6 only. At most one delimiting sentence in §4. None in §5. |
| Using raw A/B/C/D argmax as a stand-in for $A_{\mathrm{lin}}$. | Token-format artifact masquerading as accessibility. | Use option-restricted metrics; describe as "LAP-style", not "LAP". |
| Letting a single-author April-2026 preprint become load-bearing. | Citation fragility. | Use language: "concurrent", "complements", "post hoc refinement". Avoid: "explains our main result", "provides the correct framework". |

## 7. Reasoning trace (for transparent review)

Why this minimalist integration is the right level:

1. **Governance compliance.** `notes/2026-04-21-claim-framing-governance.md` forbids promoting a single result to repo-wide supremacy and instructs question-specific evidence routing. LAP touches exactly one of the four scaffold gates (localization → control). Routing it there and nowhere else is a direct application of that rule.
2. **Evidence-base hygiene.** LAP's empirical base is single-token next-token completion with single-layer DoM steering. Our setting is multi-token contextual-faithfulness compliance with neuron scaling and SAE encode-modify-decode interventions. The genres do not align tightly enough to support a load-bearing citation; they align tightly enough to support a corroborating refinement sentence.
3. **Defensive coherence.** LAP's regime-2 reading is structurally favourable to SAEs as the rescue method. If we adopt the regime framework, we hand a reviewer the most efficient possible attack on our cross-representational headline. Holding LAP at "compatible refinement" distance avoids that without hiding the work.
4. **Marginal cost of follow-up.** A one-day option-restricted accessibility analysis on FaithEval is cheap, has positive expected value under both possible outcomes (supportive or null), and pre-empts the regime-2 critique with measurement rather than rhetoric.
5. **Scaffold preservation.** The four-stage audit framework is the paper's central contribution. LAP is a within-gate diagnostic for one gate; that is exactly the level at which §6 already invites supplementation.

## 8. Action checklist

- [ ] Add `billa2026lap` to `paper/icml/references.bib`.
- [ ] Insert one Related Work sentence (text in §3.1 of this report).
- [ ] Append the Synthesis sentence at the end of §3 (text in §3.2).
- [ ] Add the §6 R1 supplementation sentence (text in §3.5).
- [ ] Decide whether the §4 boundary sentence is worth its space (default: skip).
- [ ] Schedule the Tier 1 FaithEval accessibility analysis; write up as an appendix subsection only if results are clean enough to either support the gap or rule out the regime-2 critique.
- [ ] Do **not** map our result to LAP regimes anywhere in the manuscript.
- [ ] Update `notes/research-log.md` with a dated entry once integration is applied.

## 9. Pointers

- Paper markdown: `papers/2604.15557v1.md`
- Paper PDF: `papers/2604.15557v1.pdf`
- Quick-map entry: `papers/INDEX.md` under "Steering reliability and evaluation"
- Governing framing: `notes/2026-04-21-claim-framing-governance.md`
- Manuscript surface for §3 / §4 / §5 / §6 anchors: `paper/icml/main.tex` lines 146 / 206 / 291 / 351
