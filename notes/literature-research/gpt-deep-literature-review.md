# Claim-Boundary Audit and Novelty Map for “Detection Is Not Enough: Strong Readouts Often Fail as Steering Targets in Gemma-3-4B-IT”

Live literature search executed **2026-04-12** (Atlantic/Canary). Unless otherwise noted, all web sources were retrieved **2026-04-12**. I could not access the project’s internal documents (including the named strategic assessment) from this environment, so this audit treats the prompt’s described anchors and scaffold as the canonical working framing and evaluates novelty strictly against public prior work.

## Executive novelty verdict

The **weak/background claim** (“decodability / probe accuracy does not by itself establish causal relevance or functional use”) is **well-established** and should be treated as background, not novelty. Classic probing methodology work explicitly argues probe accuracy can reflect probe capacity and dataset structure rather than model-internal reliance, motivating controls and selectivity. citeturn9search15turn9search31 The “amnesic probing” line makes the behavioral/causal point even more directly: probing can show a property is present/accessible, but that does not license behavioral conclusions about *use* without intervention-based evidence. citeturn9search0turn9search8

The **core steering-era methodological claim** (“predictive readout quality is an unreliable heuristic for choosing intervention targets”) is **partially established** but **not fully closed**. Several steering and SAE papers now say, with data, that *how you select targets* can dominate whether steering works—even when the detector/readout looks good. The most threatening near-neighbors are:

- **SAE feature selection work** that explicitly separates “features that light up on relevant inputs” from “features that causally change outputs,” showing input-activation-based selection can be systematically misleading and that filtering by an output-effect score improves steering by ~2–3×. citeturn17view0  
- **ITI-style truthfulness work** that (a) emphasizes a large gap between probe accuracy and generation accuracy and (b) shows alternative “more ‘granular’ probe-based selection” (e.g., point-wise coefficient selection) can underperform a coarser head-wise selection, despite similar-or-better probe accuracy—an early, concrete “detector ≠ good steering target” result in the same general intervention family. citeturn7view0turn8view2  
- **Causal-vs-probe location selection** work (e.g., Generative Causal Mediation) that explicitly reports beating linear-probe baselines for selecting where to steer in open-ended generative settings. citeturn2view5

So: the “detector-quality-is-not-a-steering-target-quality” message is **not novel as a vague slogan**, but it is **still plausibly novel** in the paper’s intended *empirical form* if you can show (i) matched readout strength across two localization objects (e.g., SAE features vs specific neuron sets) and (ii) sharply diverging steering outcomes on the **same** evaluation surface, with careful operator matching and robust measurement.

The **integrated four-stage claim** (“measurement, localization, control, externality are separable stages and must not be conflated”) looks **more novel as a paper-level synthesis** than the weak claim, but there is meaningful partial precedent: multiple literatures separately insist on (a) evaluation reliability for LLM judges and safety metrics, citeturn6view0turn21view2turn21view3 (b) localization vs causality in probing, citeturn9search15turn9search0turn14search3 and (c) steering reliability/generalization limits across distribution shifts. citeturn4view0turn20view0 What appears less common is a single, prescriptive scaffold that treats these as *distinct empirical gates* and uses that scaffold to interpret failures across multiple intervention/localization families. That synthesis is a defensible “flagship contribution” **if** you clearly label it as a methodological organizing framework rather than claiming you invented the underlying concerns.

Anchor-level novelty outlook (based only on public prior art):

- **Anchor 1 (localization → control break, SAE vs H-neurons on FaithEval)**: *Moderately threatened* by SAE steering selection papers and by ITI’s “selection matters” ablation, but **still plausibly novel** if you demonstrate **matched detector performance** with **divergent steerability** for two target classes on **FaithEval’s counterfactual context** surface (or a clearly equivalent surface) and you control for operator strength and injection location. citeturn17view0turn8view2turn27view0  
- **Anchor 2 (control → externality break, ITI helps MC/selection but fails on open-ended factual generation with wrong-entity substitution)**: *Conceptually threatened* by strong evidence that steering can be brittle and format-dependent (including explicit open-ended vs multiple-choice divergence), citeturn4view0turn20view0 but may retain novelty if you show a **truthfulness-specific** cross-surface failure mode (e.g., confident wrong-entity substitution) that is not already documented for truthfulness steering.  
- **Anchor 3 (measurement → conclusion break in jailbreak evaluation)**: the general point (“judges/metrics can mislead; binary vs graded can matter; evaluation artifacts exist”) is **already strongly established** in multiple safety-evaluation papers. citeturn6view0turn6view1turn21view3turn21view2turn6view2 Novelty must come from *demonstrating a reversal/flip for your concrete interventions* under realistic evaluator choices, not from asserting evaluator dependence exists.

Bottom line: the paper should not sell “detectors don’t work.” It can plausibly sell: **detector strength is not a reliable target-selection heuristic**, and even when steering “works,” it can be **surface-local** and **measurement-fragile**—*if the empirical cases are tightly controlled and staged.*

## Claim decomposition and novelty-safe wording ladder

Claim variants (from conservative to ambitious), with what the literature already supports:

**C1 (background, not novel):** “High probe/readout accuracy does not imply a representation is causally used by the model.” This is textbook probing hygiene (controls/selectivity, intervention-based causal probing). citeturn9search15turn9search0turn14search3

**C2 (background-to-supporting):** “There can be large quantitative gaps between probe accuracy and behavioral/generation accuracy on the ‘same’ concept.” ITI explicitly reports a large probe–generation gap on TruthfulQA. citeturn7view0turn2view0

**C3 (core, moderately supported):** “Readout/detector quality is an unreliable heuristic for choosing steering targets; target-selection choices can dominate intervention outcomes.” ITI’s selection ablation and SAE steering selection results already support this directionally. citeturn8view2turn17view0

**C4 (core, closer to novel if shown in your setup):** “Two localization methods can yield comparably strong readouts on the same benchmark surface, yet intervening on one yields strong steering while intervening on the other yields weak or negative steering.” Public work makes this plausible, but the *matched-strength, same-surface, cross-method* version is not obviously saturated by one canonical prior experiment. Closest precedents come from SAE selection work (activation-based selection looks right but steers poorly) and causal-vs-probe steering localization papers. citeturn17view0turn2view5

**C5 (core, partly supported):** “Success on an answer-selection / multiple-choice surface does not guarantee success on open-ended generation; steering may be format-local.” Steering reliability work and the “causality ≠ invariance” line provide direct evidence that vectors/components that steer well in one format can degrade across formats. citeturn4view0turn20view0

**C6 (flagship synthesis, plausibly novel as packaging):** “Measurement, localization, control, and externality are separable empirical stages; conflating them leads to overclaims and fragile safety conclusions.” Many papers cover pieces (judge reliability, probe-causality, steering brittleness), but fewer present a single staged scaffold as the lens for interpreting *multiple* intervention families’ failures. citeturn6view0turn14search3turn4view0turn21view2

A novelty-safe wording ladder you can reuse (with suggested scope boundaries):

**Safe formulations (very defensible):**
- “Strong readouts are not sufficient evidence that the probed representation is a good intervention target.” citeturn9search0turn17view0turn8view2  
- “Steering effects can be unreliable, input-dependent, and sensitive to distribution shift; aggregate metrics can hide large variance.” citeturn4view0turn2view3  
- “Evaluator choice and metric design can materially change conclusions in jailbreak/safety evaluation.” citeturn6view0turn6view1turn21view3turn6view2  

**Stronger but plausible (requires your anchor evidence):**
- “In Gemma-3-4B-IT, matched-strength truthfulness/faithfulness readouts can fail as steering targets, even on the benchmark family the readout was trained to predict.” (Needs matched-strength + same-surface steering divergence.)  
- “Truthfulness control discovered on multiple-choice surfaces can fail to transfer to open-ended factual generation, producing qualitatively different failure modes.” citeturn20view0turn4view0 (as precedent for format transfer brittleness; your work would supply the truthfulness-specific instantiation)

**Risky / likely-to-draw-fire (avoid unless you prove very broadly):**
- “Detectors/readouts are generally useless for steering.” (Contradicted by multiple positive counterexamples, including H-neuron and ITI-style successes.) citeturn26view0turn2view0turn20view0  
- “Our results show detector-based steering doesn’t work.” (Too absolute; the literature already contains successful detector-linked interventions.) citeturn26view0turn17view0turn20view0  
- “Measurement artifacts explain most prior steering results.” (Overreach; the right claim is *they can change conclusions* and therefore must be audited.) citeturn6view0turn21view2turn21view3  

## Prior-work map by the four-stage scaffold

### Measurement

A large 2024–2026 wave treats evaluator reliability as a first-order object, especially for safety/jailbreak settings. One ACL paper shows LLM safety judges can be highly sensitive to superficial artifacts (e.g., apologetic/verbose phrasing) and that such artifacts can dominate comparative “which is safer?” verdicts. citeturn6view0 A 2025 findings paper argues that jailbreak success rates and LLM-judge pipelines can over-index on superficial toxic tone, producing mismatches between “high jailbreak success” and “actual harmful knowledge possession,” and demonstrates inconsistent judgments across judge frameworks even when substantive content is held fixed. citeturn6view1

Metric choice is also highlighted as structural: a 2026 survey-style analysis of safety benchmarks reports heavy reliance on binary pass/fail proportions and warns this can obscure severity and uncertainty, making results look like calibrated “risk” when they are not. citeturn21view2 Complementing this, HarmMetric Eval proposes explicit harmfulness criteria (unsafe + relevant + useful) and uses a graded scoring scheme to evaluate judges/metrics across response types, explicitly targeting the problem that “unsafe-looking but irrelevant/useless” responses can confound binary metrics. citeturn21view3

For jailbreak benchmarks specifically, a recent entity["organization","MLCommons","ai benchmark consortium"] methodology paper argues that evaluation design choices (taxonomy, judge selection, disaggregation by attack type) can undermine interpretability and stability if not treated as core commitments, explicitly warning against assuming uniform judge competence across heterogeneous attack classes. citeturn21view1 Finally, judge systems themselves can be attacked: an OpenReview paper reports tokenization/segmentation biases that can bypass judge-based harmfulness detection. citeturn6view2

**Implication for your flagship:** Anchor 3 should not claim “judge dependence exists” (old news). It should claim “in *our* steering/intervention setting, reasonable evaluator/metric/truncation/parser choices can flip the sign or ranking of conclusions,” and then document that as an audited stage boundary.

### Localization

The probing literature already draws a bright line between “information is decodable” and “information is used,” emphasizing controls/selectivity and caution in behavioral interpretation. citeturn9search15turn9search31 Amnesic probing operationalizes the idea that one should test use/necessity via representational interventions rather than only probe accuracy. citeturn9search0turn9search8 Recent causal-probing reliability work proposes explicit desiderata like completeness and selectivity for interventions and shows tradeoffs and failure modes in practice. citeturn14search3

In the mechanistic/steering-adjacent space, “Finding Neurons in a Haystack” highlights that sparse probes can miss redundant-but-important neurons and that sparsity constraints can distort conclusions about what is truly important, pushing toward iterative/robust identification rather than single-shot sparse selection. citeturn23search2

On hallucination/faithfulness, the H-Neurons line localizes hallucination-associated neurons via sparse logistic regression over neuron contribution features and evaluates detection generalization across multiple hallucination scenarios, including on Gemma-3-4B. citeturn26view0 FaithEval itself provides a structured contextual-faithfulness benchmark (including counterfactual contexts) that is now being used as a diagnostic surface for mechanistic interventions. citeturn28search0turn27view0

### Control

ITI is a key precedent for the claim family: it is explicitly a “find truth-correlated probe directions at certain attention heads, then shift activations along those directions” method, evaluated on TruthfulQA multiple-choice and generation tracks, and it foregrounds the probe–generation gap motivating the intervention. citeturn2view0turn7view0turn7view2 Critically, ITI also contains an internal “selector ablation” that shows a seemingly natural probe-based alternative (point-wise coefficient selection) can be worse than head-wise selection, despite similar-or-better probe accuracy—directly relevant to “strong readouts can fail as steering targets.” citeturn8view2

Broader steering work has moved from “it works!” to measuring reliability and variance. The NeurIPS 2024 “steering vectors reliability” paper shows large per-sample variance, anti-steerable cases (steering flips direction), and that steerability is often dataset-level and brittle to prompt injections—i.e., control success is not uniform even when aggregate metrics look good. citeturn4view0 A related analysis of CAA-style steering attributes success to properties like agreement/separability of activation differences across prompts, again reinforcing that readout-like statistics and selection matter for causal control. citeturn2view3

In the SAE ecosystem, “SAEs Are Good for Steering—If You Select the Right Features” argues that common selection by activation patterns can pick “input features” that look semantically aligned but don’t causally steer; selecting “output features” improves steering substantially. citeturn17view0

### Externality

Externality/transfer failures are now explicitly documented as a core limitation of many steering methods. The NeurIPS reliability paper studies OOD prompt shifts and shows steering can degrade and become unpredictable across distributions. citeturn4view0 Even more directly aligned with your “surface mismatch” framing, “Causality ≠ Invariance” (ICLR 2026) shows (i) vectors/components that causally drive performance (Function Vectors) can be nearly orthogonal across input formats (open-ended vs multiple-choice) and (ii) steering with these vectors is strongest when extraction and application formats match, while more invariant vectors generalize better but weaker—explicitly separating **in-distribution control strength** from **cross-surface externality/transfer**. citeturn20view0

ITI also tests cross-benchmark transfer (Natural Questions, TriviaQA, MMLU) using TruthfulQA-learned directions and reports mostly modest but non-negative transfer (on their multi-choice-style evaluation protocol), which is a useful precedent but does not settle the open-ended generation externality question in your framing. citeturn8view0

## Anchor audits against nearest prior work

### Anchor audit for localization → control

**Your intended anchor:** “SAE features vs H-neurons on FaithEval, with matched detection/readout quality but diverging steering.”

**Closest established precedent signals (what already exists):**

1. **SAE activation-based selection can be misleading for steering.** The EMNLP 2025 SAE steering paper explicitly argues that selecting SAE features by “inputs that activate them” can pick features that appear relevant but have low causal impact on outputs; it introduces an output-score selection method and reports large steering improvements from this selection alone. citeturn17view0 This is a direct precedent for “detection/interpretability signal ≠ good steering target,” though it is framed around SAE feature taxonomy rather than “matched readout strength vs alternate localization.”  

2. **Probe-based selection granularity can backfire even when probe accuracy is high.** ITI’s “Why not intervene on all heads?” section shows that a probe over concatenated heads can have slightly higher probe accuracy than the best single head, but intervention schemes that don’t sparsify correctly (or that select point-wise via probe coefficients) produce worse truthfulness/helpfulness tradeoffs than head-wise selection. citeturn8view2 This is unusually close to your thesis because it explicitly tests a “stronger readout / more detailed selection” idea as a target selector and finds it can be worse.  

3. **Causal localization can outperform probe-based localization for where-to-steer.** Generative Causal Mediation reports beating baselines that select attention heads with linear probes, in settings where behavior is evaluated from longer-form outputs scored by a judge prompt. citeturn2view5 This is a strong “probe isn’t enough for steering target selection” precedent, but it differs in object (heads), method (mediation), and often concept class.  

4. **A strong detector can be a strong steering target (counterexample).** H-Neurons identifies a tiny neuron subset predictive of hallucination/over-compliance and shows that scaling those neurons causally modulates compliance across multiple benchmarks, including FaithEval’s counterfactual context subset. citeturn26view0turn27view0 This is powerful evidence that detector-selected targets *can* steer—so your claim must be about unreliability and boundary conditions, not impossibility.

**What still looks plausibly novel (if your experiments are clean):**

- A **matched-strength readout** comparison between **two different localization objects** (SAE feature set vs “H-neuron” set), on the **same FaithEval evaluation surface**, where (a) both readouts predict equally well, but (b) one intervention family fails to steer and the other succeeds. None of the above prior work obviously provides that *exact* comparison class. The SAE paper shows selection matters within SAEs; H-Neurons shows neuron sets can steer; ITI shows selection granularity matters; but the specific “two matched detectors, divergent steerability” design is not clearly standardized yet. citeturn17view0turn26view0turn8view2

**Claim-hygiene warning for this anchor:** many nearby papers already say “activations/interpretations don’t equal causal effect on outputs” and demonstrate practical steering gaps. If you claim “this is the first evidence detectors fail as steering targets,” reviewers can cite SAE selection and ITI’s selector ablation as counterexamples. citeturn17view0turn8view2

### Anchor audit for control → externality

**Your intended anchor:** “ITI improves TruthfulQA-style answer selection but fails on open-ended factual generation (‘bridge’), including confident wrong-entity substitution.”

**Closest established precedent signals:**

- Steering methods often show **OOD brittleness** and even sign flips (anti-steerable examples), implying that “it works on average” is not an externality guarantee. citeturn4view0  
- Format mismatch is explicitly demonstrated to matter: in ICLR 2026, vectors/components that causally drive task performance can be non-invariant across open-ended vs MC formats, and steering with them degrades out-of-distribution; more invariant vectors generalize better but weaker. citeturn20view0  
- ITI itself treats generalization beyond TruthfulQA as a concern and tests transfer to other datasets (using a multiple-choice-like probability comparison protocol), reporting modest improvements. This is evidence *against* a blanket “ITI never transfers,” but it does not settle open-ended generation externalities of the specific kind you describe. citeturn8view0turn7view2

**Where novelty may still live:**

- If your “bridge” surface is genuinely **open-ended factual generation** with evaluation sensitive to entity correctness (not just multiple-choice probability ranking), and you can show qualitatively systematic failures (e.g., wrong-entity substitution with high confidence) specifically induced or exacerbated by an intervention that helps MC/selection, that could be a new, truthfulness-specific externality demonstration. The closest public parallels establish that cross-format transfer can fail and that generalization is not guaranteed, but they do not canonically document *this exact failure mode* for truthfulness steering. citeturn20view0turn4view0

**Claim-hygiene warning:** reviewers can argue “this is just distribution shift / format mismatch,” and they will have strong ammo (Causality ≠ Invariance; steering reliability). Your defense should be: “Yes—our point is that these are stage-separated empirical questions that strong readouts and benchmark-local success do not answer.” citeturn20view0turn4view0

### Anchor audit for measurement → conclusion

**Your intended anchor:** “jailbreak evaluation artifacts (judge dependence, truncation, parser assumptions, graded-vs-binary reversals) can flip conclusions.”

**Closest established precedent signals:**

- Judge robustness is empirically fragile; superficial style artifacts can dominate safety verdicts. citeturn6view0  
- Jailbreak success rates can be ambiguous and can drift toward optimizing superficial toxic patterns; LLM judges can be inconsistent and insensitive to factual harmfulness content. citeturn6view1  
- Benchmark/metric design discourse explicitly criticizes binary-only pass/fail reporting and unprincipled ordinal scales; this is a “measurement changes conclusions” argument at the benchmark ecosystem level. citeturn21view2  
- HarmMetric Eval operationalizes graded scoring and criteria designed to distinguish safe refusal from irrelevant junk and from truly harmful/helpful content—explicitly because binary setups are insufficient. citeturn21view3  
- Judge pipelines can be attacked (token segmentation bias), further undermining naive reliance on a single judge. citeturn6view2

**Where novelty may still live:**

- The novelty cannot be “judges are imperfect.” It can be: “for *representation-engineering interventions*, measurement artifacts can reverse scientific conclusions about safety/control, therefore measurement must be treated as its own audited stage in mechanistic work.” That is an application-specific methodological claim; to make it stick, you need crisp demonstrations where: changing truncation length, judge model, or graded vs binary scoring changes whether an intervention counts as improving or worsening safety. citeturn21view3turn6view0turn21view1

## Positive counterexamples and boundary conditions

If the paper is titled “Detection Is Not Enough,” it must also show (or cite) when detection *is* enough—otherwise reviewers will read it as adversarially overbroad.

Strong counterexamples / “it can work” precedents:

- **H-Neurons:** a sparse detector over neuron contributions identifies a tiny neuron set predictive of hallucination/over-compliance, and simple inference-time scaling of those neurons causally modulates compliance across multiple tasks (including FaithEval counterfactual contexts). citeturn26view0turn27view0  
- **ITI:** a probe-guided intervention over selected attention heads improves TruthfulQA metrics (both MC and generation track metrics are discussed; generation uses GPT-judge + true*informative), and transfer to other datasets is at least non-negative in their protocol. citeturn7view2turn8view0turn2view0  
- **Causality ≠ Invariance:** function vectors provide strong in-distribution steering effects when extraction/application formats match (showing that “detector/selector → steering” can succeed), while transfer requires different components. citeturn20view0  
- **SAE steering can be competitive if you select output-causal features.** The EMNLP 2025 SAE paper claims SAE steering performance improves substantially when selection accounts for output effects, making unsupervised SAE steering more competitive with supervised approaches. citeturn17view0

Boundary conditions suggested by the literature (useful for narrowing your claim):

- **Format-matching is a hidden confounder**: vectors extracted from one format can encode format artifacts and fail when applied elsewhere; separating “causal performance vectors” from “invariant concept vectors” reveals a tradeoff between in-distribution strength and out-of-distribution stability. citeturn20view0  
- **Prompt-template / dataset artifacts can masquerade as steerability**: per-sample reliability can be low and anti-steerable cases common, so error bars and per-sample analysis are not optional. citeturn4view0  
- **Target selection must respect intervention sparsity/structure**: ITI shows naïvely intervening “everywhere” or selecting by pointwise coefficients worsens tradeoffs, suggesting selection should be aligned with architectural units or causal structure rather than raw readout weights. citeturn8view2  
- **Measurement must be robust to style and to adversarial artifacts**: if your “success” is evaluated by an LLM judge, the judge can be biased by apologetic/verbose language or even attacked via tokenization tricks; robustness requires explicit judge validation or jurying. citeturn6view0turn6view2

## False friends and reviewer-objection memo

False friends are papers that share vocabulary (probes / steering / SAEs) but do **not** test the key relationship: *readout strength as a heuristic for steering-target selection*, or *stage breaks across measurement/localization/control/externality*.

A compact false-friend log (non-exhaustive):

| Looks relevant by keywords | Why it’s not a true precedent for your core claim |
|---|---|
| Auto-interpretability pipelines that explain SAE features at scale | Often evaluate explanation quality, not whether “better readout/explanation quality” predicts steering efficacy or transfer; they can support framing but don’t settle detector→control reliability. citeturn18search9turn17view0 |
| Generic “activation steering works” demos without target-selection tests | If they only show a direction exists and can steer, they don’t test whether “stronger detector” is a good selector for targets or whether success transfers across surfaces. citeturn4view0turn20view0 |
| Model editing papers that rewrite facts | Adjacent in spirit but often focus on parameter edits; your thesis is about inference-time representation interventions and the reliability of detector-readouts as steering targets. (Not directly addressed by the sources above.) |
| Broad “LLM-as-a-judge surveys” | Useful orientation, but you need concrete, mechanistic measurement flips in your own intervention setting for novelty. citeturn6view0turn21view3 |

Reviewer-objection memo (what they’ll say, and how hard it bites):

**Objection: “This is just old decodability-vs-causality.”**  
They’re right for any claim that sounds like “probe accuracy doesn’t imply causal use.” You must explicitly cite probing controls/amnesic probing and then say your novelty is *not* the general critique—it’s the steering-era empirical pattern: high-quality readouts failing as intervention targets and stage breaks across measurement/control/externality. citeturn9search15turn9search0turn17view0turn8view2

**Objection: “You’re mixing apples and oranges across operators/tasks.”**  
This is the most serious threat. You need to be painfully specific: what is read out (neurons? SAE latents? head activations?), what intervention operator is used (scaling? addition? suppression?), and what surface is evaluated (MC probability ranking vs open-ended generation vs judge-scored safety). The literature shows format differences alone can make vectors nearly orthogonal and change steering generalization. citeturn20view0

**Objection: “Your positive H-neuron results undercut your thesis.”**  
Only if you appear to claim “detectors don’t steer.” H-Neurons is actually your friend if your thesis is “detector quality is an unreliable heuristic.” It establishes a *positive* regime, helping you carve boundary conditions (“sometimes detectors do yield causal levers”), which makes your claim more credible and precise. citeturn26view0turn27view0

**Objection: “Selector comparison is caveated / missing controls.”**  
This is where ITI and GCM become relevant: prior work already shows selection choices matter and that probe-based selection can lose to causal selection. You should pre-emptively align your selector comparisons with best practices: matched sparsity budgets, matched intervention strength ranges, and common evaluation surfaces. citeturn8view2turn2view5

**Objection: “This is only one model.”**  
This is a real limitation, especially because steerability and generalization can be dataset- and model-dependent; NeurIPS 2024 steering reliability explicitly measures across multiple models and finds steerability often behaves like a dataset-level property. citeturn4view0 You can partially blunt this by arguing that your strongest contribution is a *stage-separation audit* demonstrated concretely in one model, and by anchoring the model choice in the entity["company","Google DeepMind","ai research lab"] Gemma release ecosystem (Gemma Scope interpretability suite exists, so model choice is pragmatically motivated). citeturn24view2

## Related work architecture and evidence table

### Citation packs

I recommend separating citations into (A) flagship paper (stage breaks + cross-method pattern) and (B) companion measurement note (judge/metric/pathology details). The goal is to avoid bloating the flagship with evaluator taxonomy minutiae while still being unassailably grounded.

**Flagship paper: must cite (high-load-bearing)**
- Probing-selectivity + “accuracy ≠ interpretation” baseline. citeturn9search15turn9search31  
- Amnesic probing / behavioral-use framing. citeturn9search0turn9search8  
- ITI as the canonical “probe-guided truthfulness intervention,” including its own selector ablation. citeturn2view0turn8view2  
- Steering reliability/generalization limits (variance, anti-steerability). citeturn4view0turn2view3  
- SAE steering target-selection mismatch (input vs output features). citeturn17view0  
- H-Neurons as both near-neighbor and positive counterexample on Gemma-3-4B and FaithEval. citeturn26view0turn27view0  
- Causality≠Invariance as the strongest format/surface transfer precedent. citeturn20view0  
- If you cite FaithEval, cite the benchmark itself as the diagnostic surface. citeturn28search0  
- Context on Gemma 3 / Gemma-3-4B-IT as the experimental object. citeturn24view0turn24view1  

**Flagship paper: should cite (useful but not core)**
- Causal probing reliability desiderata (completeness/selectivity) as stage-two hygiene for interventions. citeturn14search3  
- Causal-vs-probe selection for “where to steer” (Generative Causal Mediation), if your paper discusses selector methodology beyond ITI/SAEs. citeturn2view5  
- Sparse probing limitations and redundancy/selection pitfalls, as support for “detector interpretation artifacts.” citeturn23search2  

**Companion measurement note: must cite**
- Judge bias to superficial artifacts. citeturn6view0  
- Jailbreak ASR ambiguity and judge insensitivity to substantive harmfulness. citeturn6view1  
- Metric/judge benchmarking with graded-vs-binary considerations. citeturn21view3turn21view2  
- Jailbreak evaluation methodology emphasizing taxonomy/evaluator heterogeneity. citeturn21view1  
- Judge vulnerability to tokenization/segmentation artifacts. citeturn6view2  

### Evidence table

Overlap strength legend: **near-direct** (tests detector/readout → target selection → steering reliability on comparable objects), **strong** (tests a close stage break with comparable operator), **moderate** (supports framing but differs in object/operator/surface), **weak** (background only).

| Paper (venue/status) | Stage(s) informed | Measured/readout object | Intervention operator | Eval surface(s) | Overlap strength vs your flagship | Novelty impact (what it “uses up”) |
|---|---|---|---|---|---|---|
| Designing and Interpreting Probes with Control Tasks (2019) | Localization | probe accuracy + selectivity framing | none (methodology) | probing tasks | weak | Removes novelty for “probe accuracy ⇒ insight” claims. citeturn9search15 |
| Probing Classifiers: Promises, Shortcomings, and Advances (2022 survey) | Localization | probe methodology limits | none | survey | weak | Background only; don’t claim novelty here. citeturn9search31 |
| Amnesic Probing (TACL 2021) | Localization → Control | decodability vs use; amnesic counterfactuals | representation randomization/projection (amnesic ops) | NLP tasks | moderate | Supports “behavioral claims need interventions.” citeturn9search0turn9search8 |
| How Reliable are Causal Probing Interventions? (2024/2025) | Localization → Control | completeness/selectivity via validation probes | intervention evaluation framework | prompt tasks | moderate | Supports stage-separation and intervention auditing. citeturn14search3 |
| Inference-Time Intervention (arXiv 2023; NeurIPS 2023 poster) | Localization → Control → Externality | head-wise linear probes for truthfulness; probe–generation gap | activation shifting on selected heads | TruthfulQA MC + generation; transfer to other datasets | near-direct | Strong precedent: probe-guided steering + selector ablation already exists. citeturn2view0turn8view2turn8view0 |
| Analysing the Generalisation and Reliability of Steering Vectors (NeurIPS 2024) | Control → Externality | steering vectors (mean-diff) | vector addition in residual stream | many MC-style datasets; OOD prompt injections | strong | Uses up “steering is brittle; aggregate hides variance.” citeturn4view0 |
| Understanding (Un)reliability of Steering Vectors (ICLR 2025 workshop) | Control | prompt-type effects on steering vectors; predictors of success | CAA vector addition | binary-choice datasets | moderate | Supports “readout properties predict steering success” but not your exact cross-method claim. citeturn2view3 |
| SAEs Are Good for Steering—If You Select the Right Features (EMNLP 2025) | Localization → Control | SAE features: input-score vs output-score taxonomy | single-feature amplification; output-score-based selection | AxBench steering evaluation; coherence/quality | near-direct | Directly threatens “activation-based selection works”; supports your thesis strongly but narrows novelty to cross-method staging. citeturn17view0 |
| H-Neurons (arXiv 2025) | Localization → Control → Externality (within over-compliance) | sparse neuron set detector via L1 logistic regression over CETT | inference-time activation scaling | TriviaQA/NQ/BioASQ/Non-exist; FalseQA; FaithEval; sycophancy; jailbreak | near-direct | Establishes detector→steering can work and uses FaithEval; forces your claim to be “unreliable heuristic,” not “impossible.” citeturn26view0turn27view0 |
| FaithEval benchmark (arXiv 2024; ICLR 2025) | Measurement surface | contextual-faithfulness tasks (unanswerable/inconsistent/counterfactual contexts) | none | contextual QA faithfulness | supporting | Ensures you treat FaithEval as a diagnostic surface with specific task structure. citeturn28search0 |
| Causality ≠ Invariance (ICLR 2026) | Control → Externality | Function vs Concept vectors; RSA/AIE heads | steering with vectors | open-ended vs MC; cross-language | strong | Uses up “MC vs open-ended can break transfer” framing; supports your Anchor 2 logic. citeturn20view0 |
| Activation Steering via Generative Causal Mediation (arXiv 2026) | Localization → Control | head selection: causal mediation vs probe baselines | steering on top-k heads | long-form concept scoring via judge prompts | strong | Threatens novelty if you claim “no one tested probe selection vs causal selection”; but supports your selector-stage framing. citeturn2view5 |
| Safer or Luckier? LLMs as Safety Evaluators… (ACL 2025) | Measurement | judge artifacts; self-consistency; human alignment | none (evaluation study) | safety judging tasks | strong | Uses up “judge dependence / style artifacts can dominate.” citeturn6view0 |
| Rethinking Jailbreak Evaluation / VENOM (EMNLP Findings 2025) | Measurement → Conclusion | jailbreak ASR ambiguity; judge insensitivity to factual harmfulness | none (benchmarking framework) | harmful knowledge possession; judge robustness tests | strong | Uses up “ASR & judge pipelines can mis-measure what we care about.” citeturn6view1 |
| HarmMetric Eval (arXiv 2026) | Measurement | compares judges/metrics; graded scoring; criteria (unsafe/relevant/useful) | none (metric evaluation) | harmfulness assessment | strong | Supports graded-vs-binary reversals and metric dependence. citeturn21view3turn21view2 |
| MLCommons Jailbreak 0.7 methodology (2026) | Measurement | evaluator heterogeneity; taxonomy-driven reporting | none (benchmark methodology) | jailbreak benchmarking | moderate | Supports “evaluation design is a first-order commitment.” citeturn21view1 |
| Judge LLM segmentation/tokenization vulnerability (OpenReview) | Measurement | judge vulnerability | attack on judge pipeline | harmfulness judging | moderate | Supports “judge dependence is not just noise; it can be adversarially exploited.” citeturn6view2 |
| Probing Ethical Framework Representations… (arXiv 2026) | Localization → Control | probe directions detect internal states; steering changes preference not accuracy | activation steering along probe directions | ethical-choice tasks | strong | Direct precedent for “probe-identified direction is not a good causal lever for improving the intended property.” citeturn14search9 |
| Gemma 3 Technical Report; Gemma-3-4B-IT model card (2025) | Experimental object | model family + IT variant | n/a | n/a | supporting | Grounds model identity; useful for “one-model” limitation context. citeturn24view0turn24view1 |
| Gemma releases (2026 updates) | Experimental context | interpretability suite availability; release timeline | n/a | n/a | supporting | Motivates Gemma as a MI-friendly choice (but does not claim your result generalizes). citeturn24view2 |

### Optional paper-ready paragraphs

Flagship Related Work paragraph (concise, claim-hygienic):

Mechanistic studies often use probes/readouts to localize where a property is decodable, but classic probing work emphasizes that probe accuracy alone is insufficient for behavioral conclusions without controls and intervention-based validation. citeturn9search15turn9search0 In truthfulness steering, Inference-Time Intervention operationalizes a probe-guided activation-shift approach and already finds that selection choices matter: more granular probe-based feature selection can worsen truthfulness–helpfulness tradeoffs relative to head-wise selection. citeturn8view2turn2view0 More broadly, steering vectors show high variance and brittle generalization, with anti-steerable inputs and prompt-shift sensitivity that aggregate metrics can hide. citeturn4view0 Recent SAE work further separates “features that activate on relevant inputs” from “features that causally change outputs,” showing that activation-based selection can pick misleading targets and that output-effect-based selection improves steering substantially. citeturn17view0 Finally, format transfer is not guaranteed: components that causally drive behavior can be non-invariant across open-ended and multiple-choice formats, producing strong in-distribution control but degraded cross-format effects. citeturn20view0

Companion measurement-note Related Work paragraph:

Safety and jailbreak evaluation increasingly relies on LLM judges, yet empirical work shows judge verdicts can be distorted by superficial artifacts (e.g., apologetic/verbose style) and can disagree across judge frameworks even when substantive content is controlled. citeturn6view0turn6view1 Benchmark studies further criticize heavy dependence on binary pass/fail proportions, arguing such metrics obscure severity and uncertainty, while newer benchmarks propose graded criteria-based scoring to distinguish truly harmful/helpful content from irrelevant or reflexively toxic outputs. citeturn21view2turn21view3 Methodology efforts emphasize that evaluation design—including taxonomy, disaggregated reporting, and evaluator robustness—must be treated as a first-order commitment rather than an afterthought. citeturn21view1

Intro-framing paragraph (accurately positioning novelty without overclaiming):

Across probing, steering, and safety evaluation, it is now clear that (i) decodability does not entail causal use, (ii) steering success can be unreliable and distribution-sensitive, and (iii) measurement choices can bias conclusions. citeturn9search0turn4view0turn6view0turn21view3 This paper treats these as separable empirical stages—measurement, localization, control, and externality—and studies how failures at each stage can invalidate common heuristics for choosing intervention targets. Our focus is not to argue that detector-selected interventions never work (they sometimes do), but to characterize when strong readouts fail as steering targets and when apparent control is benchmark- or surface-local. citeturn26view0turn20view0turn17view0

Direct URLs for key primary sources (retrieved 2026-04-12):

```text
ITI (arXiv:2306.03341): https://arxiv.org/pdf/2306.03341
NeurIPS 2024 Steering Reliability: https://proceedings.neurips.cc/paper_files/paper/2024/file/fb3ad59a84799bfb8d700e56d19c231b-Paper-Conference.pdf
SAEs Are Good for Steering (EMNLP 2025): https://aclanthology.org/2025.emnlp-main.519.pdf
H-Neurons (arXiv:2512.01797): https://arxiv.org/pdf/2512.01797
Causality ≠ Invariance (arXiv:2602.22424): https://arxiv.org/pdf/2602.22424
Safer or Luckier? (ACL 2025): https://aclanthology.org/2025.acl-long.970.pdf
VENOM / Rethinking Jailbreak Evaluation (EMNLP Findings 2025): https://aclanthology.org/2025.findings-emnlp.92.pdf
HarmMetric Eval (arXiv HTML): https://arxiv.org/html/2509.24384v2
Gemma 3 Technical Report (arXiv:2503.19786): https://arxiv.org/abs/2503.19786
Gemma releases page: https://ai.google.dev/gemma/docs/releases
```