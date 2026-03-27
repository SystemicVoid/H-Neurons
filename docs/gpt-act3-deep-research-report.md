# Act Three Pivot: From H-Neurons to Pragmatic Safety Steering

> Reference document. Live execution is tracked in [act3-sprint.md](./act3-sprint.md). Live evaluation rules are tracked in [measurement-blueprint.md](./measurement-blueprint.md).

## What your replication already demonstrated

Your project is no longer “about 38 neurons.” It is now about a larger, field-relevant problem: **how easy it is to get the *wrong safety conclusion* from the *wrong measurement*, and how neuron-level interventions can create large, hard-to-see safety externalities**.

Two of your strongest, most generalizable findings are methodological rather than model-specific:

First, **response truncation can dominate the error budget** for jailbreak evaluation. In your full-population rerun, every α=0.0 sample exceeded 256 tokens, meaning the legacy 256-token setup was not “a little noisy,” it was *information-destroying by design* (100% truncation at α=0.0). fileciteturn2file0L432-L444 This matters because many refusal-then-comply behaviors place the harmful payload *after* a long disclaimer/preamble, so truncation preferentially hides the very content you are trying to measure. fileciteturn2file0L382-L401

Second, you showed that **binary “harmful vs safe” judging (even with a strong LLM judge)** can be too low-resolution to detect the real intervention effect. Your CSV-v2 graded evaluation uncovered a statistically significant α slope for “strict harmful” (yes) outputs (+7.6pp from α=0.0→3.0), while the binary judge’s “harmful” rate appeared much flatter—because it over-called borderline, disclaimer-wrapped responses as harmful at baseline. fileciteturn2file0L481-L490 fileciteturn2file0L550-L560

Those measurement fixes didn’t just change a p-value; they changed the story:

- **Severity escalates with α** in ways a binary label can’t see: (i) high-utility harmful outputs (V=3) nearly quadruple (3.8%→14.0%), (ii) “turnkey artifacts” (S=4) nearly triple (3.0%→8.4%), and (iii) the harmful payload becomes a larger share of the output while the “pivot” into harmful content moves earlier—i.e., “disclaimer erosion” becomes quantifiable. fileciteturn2file0L520-L548  
- The intervention effect is also **prompt-sensitive and non-monotonic** at the individual prompt level, with substantial churn even when aggregate rates look stable. fileciteturn2file0L420-L431

This set of results is already a pragmatic interpretability contribution: it’s a **measurement blueprint** for evaluating safety steering that the wider community can adopt (full generations, graded severity, explicit accounting for disclaimer-wrapped borderline behavior). fileciteturn2file0L481-L560

## What the best current literature implies about “what to do next”

The literature since 2024 has converged on a blunt lesson: **if you want predictable behavioral control, individual-neuron interventions are rarely the cleanest unit of causality; directions, heads, and causally-validated components win**.

### Refusal control is (often) direction-dominated, but not always one-dimensional

The “single refusal direction” line of work shows that across many aligned chat models, refusal behavior is strongly mediated by a **one-dimensional subspace** in residual-stream activations, where:
- **erasing** that direction prevents refusal of harmful instructions, and  
- **adding** that direction can induce refusal even for harmless requests. citeturn5view0  

Crucially for your Act Three: the paper also specifies a low-cost, high-signal method (difference-in-means over harmful vs harmless prompts) and demonstrates concrete intervention operators (activation addition, directional ablation). citeturn7view0  

It also contains two mechanistic hooks that connect directly to your “disclaimer erosion” observations:

- The refusal direction appears to be **present in base models and repurposed (“hooked into”) during safety fine-tuning**, rather than created from scratch. citeturn11view0  
- Adversarial suffixes can suppress refusal by **hijacking attention** of heads that write to the refusal direction, shifting attention from instruction to suffix tokens. citeturn11view2  

However, 2025–2026 work also stresses that refusal is not a monolith: multiple papers argue for **multiple refusal-related directions or even higher-dimensional cones** (different refusal categories, different geometries, non-linearities). citeturn2search1turn2search2  
The practical implication is: **a single “refusal direction baseline” is necessary, but not sufficient, as the final story**—especially if your own data already suggests category-dependent effects and prompt-level non-monotonicity. fileciteturn2file0L420-L450

### Steering is an attack surface unless you explicitly audit safety externalities

Two March–February 2026 papers are especially relevant to your pivot moment, because they show the *same failure mode* you are seeing—at scale, across models, and across “benign” steering objectives:

- A systematic audit of contrastive steering vectors finds that steering can **drastically increase or decrease jailbreak attack success rates** (reported up to +57% or −50%), and links this to **geometric overlap between the steering vector and refusal-related directions**. citeturn5view1  
- “Steering externalities” shows that even steering derived from benign datasets (e.g., enforcing compliance or structured output formats) can **erode safety guardrails** and act as a force multiplier for jailbreak success. citeturn10search0turn10search4  

This basically forces a new norm for pragmatic interpretability: **every steering method should ship with a safety externality audit** (what does it do to jailbreak/refusal robustness?), not just a “does it achieve the target behavior?” metric. citeturn5view1turn10search0  
Your CSV-v2 work is well-positioned to be the measurement layer for exactly that audit. fileciteturn2file0L481-L560

### For hallucinations, the frontier is shifting from “neurons” to “truthfulness / uncertainty representations”

The hallucination side has matured into two complementary threads:

**Inference-time truthfulness steering**: Inference-Time Intervention (ITI) improves TruthfulQA substantially by shifting activations along learned “truth directions” at a small set of attention heads, and explicitly observes a **truthfulness–helpfulness tradeoff** that can be tuned by intervention strength. citeturn5view3  

**Cross-task truthfulness representations**: The “universal truthfulness hyperplane” work suggests that a truthfulness-separating hyperplane can generalize better when trained across many datasets; **diversity beats volume** for generalization. citeturn1search0turn5view2  

A separate, very pragmatic detection angle is: **do not only steer—detect and gate**. Semantic-entropy approaches aim to detect certain hallucination modes using uncertainty at the level of meaning rather than tokens. citeturn13search0 And “LLMs must be taught to know what they don’t know” argues that prompting alone is insufficient for reliable calibration, but that small fine-tuning can yield uncertainty estimates with good generalization. citeturn13search1  

Finally, theory work argues there are settings where some hallucination-like errors have unavoidable lower bounds under calibration assumptions—so “zero hallucinations” is not a sane target; “detect, abstain, and reduce the dangerous subset” is. citeturn13search2  

### Causal localization is the “adult supervision” that correlational probes lack

Two causal-mechanistic approaches matter because they directly answer your concern: “we’re just making their probe better, but the foundation may be wrong.”

- “Safety neurons” work uses activation contrasting plus **dynamic activation patching** to identify neurons causally implicated in safety behavior, and reports that patching a sparse subset can recover safety performance while preserving general ability. citeturn9view0  
- Generative Causal Mediation (GCM) focuses on long-form behaviors and selects components (often attention heads) by **causal mediation**, outperforming correlational probe-based baselines for sparse steering. citeturn8view1  

This hits the core critique of the original H-neurons-style approach: **an L1 probe can be an excellent detector and still be a sloppy intervention selector**. If your goal is “make things safer,” causal localization is the more principled endpoint. citeturn8view1turn9view0  

## Decision criteria for pragmatic interpretability in safety

For Act Three, the right framing is not “what is intellectually consistent with the original paper,” but “what creates a field-useful artifact under real constraints.”

A pragmatic interpretability technique, in 2026, should satisfy four criteria:

It should provide **causal leverage** (intervene on the thing you measured, get the behavior you claim), not just correlational salience. Directional ablation/addition for refusal and causal mediation / activation patching are explicit examples of this standard. citeturn7view0turn8view1turn9view0  

It should be **robust under distribution shift** (new prompts, new templates, new domains). Your own results show that prompt-level churn can be huge even when aggregate metrics look stable, and the truthfulness literature shows how single-dataset probes can overfit. fileciteturn2file0L420-L431 citeturn1search0  

It should explicitly measure the **controllability–safety tradeoff**: a method that improves “utility” but quietly increases jailbreak success is not alignment—it’s a new interface for failure. The 2026 steering audits make this non-negotiable. citeturn5view1turn10search0  

It must be evaluated with **measurement that matches the failure mode**. Your CSV-v2 result is a case study: “binary harmful” missed the actual α effect because disclaimer-wrapped borderline behavior formed a large noise floor at baseline. fileciteturn2file0L481-L490 fileciteturn2file0L550-L560  

If you adopt these criteria, the project’s north star becomes clear:

> Move from “38 neurons as the object” → “a safety steering protocol with causal targets, safety externality audits, and graded evaluation.”

## Tradeoffs across plausible Act Three paths

### Continuing to iterate on the neuron set

This includes C-sweeps, alternative probes, and exploring “negative” neurons.

The upside is narrative continuity and low switching cost: you already have tooling, data, and intuitions. Your own work also suggests there is meaningful structure in *how* the model fails (disclaimer erosion, pivot position moving earlier, severity axis shifts), and neuron-level hypotheses can be a microscope for that. fileciteturn2file0L520-L548  

The downside is that this path is increasingly dominated by a known limitation: **correlational selection of neurons is not the best available mechanism for steering**, and the field’s baseline for “clean control” is now direction- and head-based with causal checks. citeturn7view0turn8view1  
If you spend your final two weeks optimizing neuron selection without benchmarking against these baselines, you risk producing the academic equivalent of tuning a carburetor in the age of fuel injection: technically interesting, strategically obsolete.

Strong opinion: treat neuron iteration as a **short “closure experiment,” not the main act**.

### Pivoting to direction-based baselines and using your work as an evaluation + safety audit layer

This path says: keep your replication as Act One/Two, then do Act Three as **comparative benchmarking against best-in-class steering**.

The upside is that it directly answers “what technique should other people use?” and it gives you a publishable, field-facing result even if neuron-level interventions remain messy. Directional refusal control is well specified (difference-in-means, directional ablation/addition) and comes with capability measurements (small drops on MMLU/ARC/GSM8K, bigger differences on TruthfulQA). citeturn7view0turn7view3turn7view4  

The downside is perceived novelty: “we implemented the refusal direction paper and got expected results” is not enough. The novelty needs to be in the *bridge*:
- using CSV-v2-style graded metrics to show what binary metrics miss,  
- quantifying steering externalities, and  
- connecting hallucination steering to safety risks (refusal overlap). citeturn5view1turn10search0 fileciteturn2file0L481-L560  

### Going causal: GCM-style mediator selection or safety-neuron patching

This is the “most correct” mechanistic direction.

The upside is conceptual strength: it aligns exactly with your intuition that the foundation is wrong. GCM argues probe-based localization fails for long-form diffused concepts and shows a causal mediator approach that outperforms probe baselines. citeturn8view1  
Safety-neuron work similarly elevates from “identify” to “causally validate and apply,” including the sobering point that safety and helpfulness can overlap at the component level (alignment tax). citeturn9view0  

The downside is execution risk in a two-week window: implementing, validating, and comparing causal mediation pipelines can balloon quickly unless you run a sharply scoped pilot.

Strong opinion: do **one causal pilot** (enough to demonstrate the principle), but don’t bet the whole sprint on a brand-new causal pipeline unless you already have it half-built.

## Recommended Act Three experiments and deliverables

The optimal path is a hybrid that preserves narrative continuity while producing field-useful guidance:

### Establish a three-way baseline suite

Baseline A: your current H-neuron intervention, evaluated with full-length generations and CSV-v2-style graded scoring (you’ve already done the hard part here). fileciteturn2file0L432-L444 fileciteturn2file0L481-L548  

Baseline B: a refusal-direction intervention extracted for your model family (difference-in-means harmful vs harmless prompts; then directional ablation/addition). This is cheap, crisp, and comes with a mechanistic story: a single residual direction whose removal/addition directly modulates refusal. citeturn7view0turn5view0  

Baseline C: a truthfulness/hallucination steering baseline:
- either ITI-style attention-head interventions (truth directions; tune strength; watch helpfulness tradeoff), citeturn5view3  
- or a “truthfulness hyperplane” style direction trained with deliberate dataset diversity to avoid TriviaQA-style overfitting. citeturn1search0turn5view2  

Why this baseline suite matters: it turns Act Three into a **Pareto frontier** exercise: jailbreak risk vs hallucination reduction vs capability retention—rather than a single-axis “did the neuron trick work?” narrative. citeturn5view1turn8view0  

### Quantify the geometric mechanism behind safety regressions

Both major 2026 steering audits attribute safety erosion to **overlap with refusal-related directions**. citeturn5view1turn10search0  

Your most valuable “bridge” experiment is therefore:

- compute a vector representation of your intervention (e.g., the induced residual-stream shift at the intervention layer),  
- compute/refit the refusal direction for your model, and  
- measure overlap (cosine similarity / projection) and how it predicts CSV-v2 harm severity changes.

This would connect your empirical “disclaimer erosion + severity” findings to a mechanistic explanation the current literature is converging on: **you’re not discovering “a special hallucination neuron set,” you’re likely pushing the model along (or against) a shallow set of safety-relevant directions—sometimes unintentionally.** citeturn5view1turn7view0turn11view0 fileciteturn2file0L520-L560  

### Build a “safety-aware steering” mitigation as your positive contribution

Both the steering-vector audit and the steering-externalities paper point to the same mitigation idea: **remove or control the refusal-overlap component** to reduce safety erosion. citeturn5view1turn10search0  

Concretely, Act Three can propose (and test):

- a truthfulness / hallucination steering vector (from ITI or hyperplane methods),  
- a refusal direction, and  
- a “refusal-orthogonalized” truthfulness steering vector (project out refusal component), then compare:

1) truthfulness improvement,  
2) jailbreak risk change (with CSV-v2),  
3) capability drift (at least a small proxy suite), and  
4) response-structure shifts (pivot position, payload share, disclaimer rate). citeturn7view3turn7view4turn5view1 fileciteturn2file0L520-L560  

This is the kind of result that is both **pragmatic** and **interpretability-grounded**: “here is the vector; here is the overlap; here’s how to neutralize the safety side effect.”

### Run one scoped causal pilot to validate your critique

Pick a single target behavior (e.g., refusal robustness to templated jailbreaks) and do a minimal GCM-style mediator identification pilot:
- select a small contrastive dataset of long-form responses,  
- rank a modest number of components (heads or layers) by indirect causal effect,  
- steer using only the top-k components.

You don’t need to outdo the GCM paper; you need to show that **causal selection changes what you pick compared to correlational probes**, and that this changes steering reliability or safety externalities. citeturn8view1  

### Package your work as a field-useful artifact

If the goal is “help the field do better,” the most valuable deliverable is a *protocol*, not merely a negative result:

- A reproducible evaluation recipe for steering safety: full-length generation, graded safety severity, and explicit reporting of judge blind spots (false negatives from disclaimer-framing). fileciteturn2file0L382-L401 fileciteturn2file0L550-L574  
- A standard “steering externality audit” section: report jailbreak success drift for any steering intended to improve another property (truthfulness, format compliance, style). citeturn10search0turn5view1  
- A reality check for hallucination work: some hallucination modes may be structurally inevitable without abstention or post-training; measure what subset you can detect/mitigate, and consider detection+gating (semantic entropy, uncertainty estimators) as a safer deployment approach than always-on steering. citeturn13search0turn13search2turn13search1  

If you do this, Act Three is no longer “we tried neuron scaling and it’s messy.” It becomes:

> “We built a high-resolution evaluator and showed why common safety measurements miss steering-induced harm; we benchmarked neuron vs direction vs causal methods; and we propose a safety-aware steering protocol that audits and mitigates refusal-overlap externalities.”

## A high-impact narrative that stays connected to your work but scales beyond it

The cleanest story arc (and, frankly, the highest-status contribution) is:

Your replication showed that evaluating jailbreak safety under truncation and binary judging can produce spurious or damped conclusions, and that h-neuron interventions can increase both **rate and severity** of harmful outputs via measurable response-structure shifts (payload share up, pivot earlier, disclaimer disappearance). fileciteturn2file0L432-L444 fileciteturn2file0L481-L548  

You then position H-neurons as a **case study in “correlational localization is not causal control,”** and you benchmark against the modern best practice: residual directions, attention-head interventions, and causal mediation, which the 2024–2026 literature increasingly supports as more reliable and interpretable intervention units. citeturn7view0turn8view1turn5view3turn9view0  

Finally, you produce a positive synthesis: **safety-aware steering** that treats steering as a potential attack surface, quantifies overlap with refusal mechanisms, and mitigates externalities—directly aligned with the newest steering risk literature. citeturn5view1turn10search0  

One last (gentle) jab of humor because you’ve earned it: the “38 neurons” are not your villain; they’re your plot device. The real antagonist is **evaluating safety with a blindfold and calling it science**. Your CSV-v2 results are basically ripping off the blindfold. fileciteturn2file0L481-L560
