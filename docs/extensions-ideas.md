### Evaluating the Proposed Research Extensions

The deep research reports propose several avenues to solve the "blunt instrument" problem and advance mechanistic interpretability. Here is a weighed analysis of the most promising directions, using analogies to clarify their trade-offs.

#### 1. H-Circuit Discovery: Moving from Neurons to Causal Graphs

*This approach maps the entire pathway. Instead of just looking at the H-Neurons in the FFNs, it traces the signal back through the attention heads to find the full "over-compliance circuit."*

* **Analogy:** Imagine the H-Neurons are the **loudspeakers** in an airport announcing a flight delay. The original paper found the speakers and turned down the volume. Circuit discovery aims to find the **control room, the microphone, and the person** reading the script.
* **The Promise:** High impact. By understanding the upstream triggers (e.g., specific "Decision Heads" that route the user's prompt into the H-Neurons), you can create highly targeted interventions. You might realize that suppressing a specific attention head stops the hallucination before it even reaches the FFN.
* **The Trade-off (Difficulty vs. Precision):** This is conceptually and computationally heavy. Finding these circuits (using techniques like Path Patching and ACDC) in massive models requires significant infrastructure (like TransformerLens) and can easily overfit to the specific dataset used for testing. However, the resulting precision is the gold standard for mechanistic interpretability.

#### 2. Feature Extraction via Sparse Autoencoders (SAEs)

*This approach argues that individual neurons are polysemantic (they do many things at once). It uses SAEs to disentangle the "over-compliance" signal from benign language features.*

* **Analogy:** Imagine an H-Neuron is a **single piano key** that, due to a manufacturing flaw, plays a beautiful C note (fluent language) but also simultaneously triggers a harsh buzzer (hallucination). Static scaling (the original paper) just unplugs the key—you lose both the buzzer and the music. SAEs act as an **acoustic filter**, allowing you to isolate and mute the frequency of the buzzer while keeping the piano note perfectly intact.
* **The Promise:** High feasibility and deep theoretical alignment. The field is rapidly adopting SAEs as the true "units of analysis." By finding the specific "compliance feature" vector within the SAE latents, you achieve zero-degradation mitigation. You can steer the model away from sycophancy without breaking its grammar.
* **The Trade-off (Compute vs. Granularity):** Training SAEs across multiple layers of a 70B parameter model is massively resource-intensive. Furthermore, recent debates suggest SAE features might not be the "canonical" ground truth, meaning you might end up chasing artifacts of the autoencoder rather than the model's true logic.

#### 3. Conditional Gating Policies (Adaptive Intervention)

*Instead of permanently suppressing H-Neurons, this approach builds a dynamic controller. It monitors the model's internal uncertainty (or semantic entropy) and only suppresses the H-Neurons when the model is genuinely guessing.*

* **Analogy:** Think of H-Neurons as the **accelerator pedal** in a car. Static scaling puts a permanent block under the pedal so the car can never go fast. Conditional gating installs an **automatic braking system** (ABS). The car can drive normally at high speeds, but the moment the system detects icy conditions (high epistemic uncertainty/lack of knowledge), it automatically pulses the brakes (suppresses the H-Neurons).
* **The Promise:** The most pragmatic and deployable solution. It beautifully solves the capability-reliability trade-off. It acknowledges that the "compliance" drive is useful for general conversation but dangerous when the model is making up facts.
* **The Trade-off (Simplicity vs. Robustness):** It relies heavily on training a separate, lightweight probe to measure uncertainty accurately in real-time. If the uncertainty probe fails (e.g., the model is confidently wrong), the gating mechanism won't trigger, and the hallucination slips through.

#### 4. Safety-Neuron Overlap and Decoupling (Solving the Alignment Tax)

*This explores the relationship between H-Neurons (which cause over-compliance) and Safety Neurons (which cause refusal of harmful prompts). It aims to decouple them via fine-tuning to solve the "alignment tax."*

* **Analogy:** Imagine the model's brain has a single **"Say No" muscle**. To be safe, it must "Say No" to making a bomb. To be truthful, it must "Say No" to a false premise. Currently, standard RLHF alignment often weakens this muscle when trying to make the model more "helpful," resulting in a model that complies with jailbreaks and hallucinates to appease the user. This approach seeks to surgically train the muscle to differentiate between "helpful refusal" (safety) and "epistemic refusal" (truth).
* **The Promise:** Extremely high relevance to AI Safety. If successful, it proves why making models truthful often makes them easier to jailbreak, and provides a mathematical way to optimize for both simultaneously without degrading either.
* **The Trade-off (Intervention vs. Observation):** It requires moving beyond inference-time tricks into actual Mechanistically Constrained Fine-Tuning (CPT or DPO). Altering parameter weights to mathematically orthogonalize these concepts is notoriously difficult and runs the risk of catastrophic forgetting or breaking overall model coherence.

---

#### 1. Conditional Gating & Adaptive Steering (Dynamic over Static)

* **The Analogy:** The original paper cuts the brake lines entirely to stop the car from speeding. Conditional gating installs an ABS system—it only pumps the brakes (suppresses H-Neurons) when it detects a patch of ice (high epistemic uncertainty).
* **The Trade-off:** * *Pros:* Highly pragmatic. Solves the exact limitation the authors noted (helpfulness degradation). Fast feedback loop.
* *Cons:* Requires you to set up a secondary "uncertainty" probe (like semantic entropy or an internal state probe) alongside the H-Neuron hooks.


* **Hardware Fit: Excellent.** Inference-time hooks are computationally cheap. You can run this easily on a quantized 8B or a base 2B model locally.
* **Verdict:** **Top Tier.** This is a perfect 1-to-2 week Stage 3 sprint.

#### 2. The SAE Route: "H-Features" over "H-Neurons"

* **The Analogy:** An H-Neuron is a chord played on a piano; it contains the "over-compliance" note, but also the "grammar" note and the "French translation" note. A Sparse Autoencoder (SAE) isolates the individual notes. You want to mute *only* the compliance note.
* **The Trade-off:**
* *Pros:* Aligns with the cutting-edge Anthropic/DeepMind paradigm. High interpretability.
* *Cons:* "SAE hyperparameter sprawl." Training SAEs is notoriously finicky and compute-intensive.


* **Hardware Fit: Good, *IF* you cheat.** Training an SAE from scratch on your 5060 Ti is a very slow loop (weeks). But if you use **pre-trained SAEs** (like Gemma Scope via `SAELens`), the feedback loop becomes minutes. You just project the H-Neuron activations into the SAE feature space and look for the "compliance feature."
* **Verdict:** **Highly Promising.** Use `Gemma-2-2B` and Gemma Scope.

#### 3. Safety-Neuron Overlap (The "Alignment Tax" Circuit)

* **The Analogy:** Discovering that the immune system (safety/refusal) and an autoimmune disease (hallucination/over-compliance) are using the exact same white blood cells.
* **The Trade-off:**
* *Pros:* Massive impact for AI safety. Explains why making models safer sometimes makes them hallucinate more (the alignment tax). Conceptually simple to execute.
* *Cons:* Might be conceptually shallow if the overlap is just index-based rather than functional.


* **Hardware Fit: Excellent.** This requires running prompts, caching activations, and training linear probes (Logistic Regression). Your AMD 9900X and 16GB GPU will tear through this.
* **Verdict:** **Great Starter Project.** This is a perfect 2-to-5 day Stage 2 mini-project to get your hands dirty with the tooling.

#### 4. Circuit Discovery (Path Patching / ACDC)

* **The Analogy:** H-Neurons are the spark plugs. Circuit discovery traces the wiring back to the ignition switch (specific attention heads) to find the whole "over-compliance circuit."
* **The Trade-off:**
* *Pros:* The gold standard for mechanistic rigor.
* *Cons:* Activation/Attribution patching is a notoriously slow loop. False negatives are common due to backup circuits.


* **Hardware Fit: Danger Zone.** Storing cache dictionaries for all layers/heads on an 8B model will instantly exceed your 16GB VRAM. You would be forced to use very small models (e.g., GPT-2 Small) or write complex offloading code.
* **Verdict:** **Skip for now.** Unless you explicitly use `nnsight` to run the patching remotely on the NDIF cluster, the local hardware friction will kill your momentum.

#### 5. Large Reasoning Models (DeepSeek-R1 CoT Mapping) & Pre-training

* **The Analogy:** Performing open-brain surgery on a whale (LRMs) or trying to selectively breed them (Pre-training).
* **The Trade-off:** Massive relevance to the current meta, but temporally mapping H-Neurons across 10,000 tokens of Chain-of-Thought is an engineering nightmare.
* **Hardware Fit: Impossible locally.** * **Verdict:** **Hard Pass.** Do not attempt this until you are extremely comfortable with the basics.

---

### My Recommendation for Your Next Steps

I recommend you pursue **Avenue 3 (Safety Overlap)** as a 3-day warm-up, and then transition into **Avenue 2 (SAE H-Features)** as your main project.

**Step 1: Setup your rigorous local environment.**
Do not use pip. Use `uv` to create an isolated, reproducible environment.

```bash
uv init h-neuron-experiments
cd h-neuron-experiments
# Install the holy trinity of Mech Interp, plus typing and ML basics
uv add torch transformer-lens nnsight sae-lens jaxtyping scikit-learn pandas

```

**Step 2: The Warm-up (Safety vs. Hallucination Overlap)**

1. Load `Qwen/Qwen1.5-1.8B-Chat` or `google/gemma-2-2b-it` locally via TransformerLens.
2. Replicate the H-Neuron extraction (Logistic Regression on TriviaQA correct vs. incorrect answers).
3. Create a dataset of harmful vs. harmless prompts. Extract "Safety Neurons" using the same linear probing technique.
4. Calculate the cosine similarity of the down-projection weights of both sets of neurons. Are they the same circuit?

**Step 3: The Deep Dive (SAE H-Features)**
Instead of doing blunt activation scaling ($\alpha < 1$) on the raw H-Neurons, load the Gemma Scope SAEs for those specific MLP layers using `SAELens`. Pass the hallucinatory activations through the SAE. Find the sparse features that activate. Subtract *only* that feature vector from the residual stream. Does the model stop hallucinating while maintaining perfect grammar?

---

### 🧠 Mentor's Note

You have great ideas here, but ideas are cheap. Execution is everything.

If you decide to do the SAE Feature extraction, **what is your baseline?** If you subtract the "compliance feature" and the model stops hallucinating, how do you mathematically prove you didn't just lobotomize the model's ability to speak English?

You need to establish a capability baseline *before* you run your first experiment. I want you to think about how you will measure standard cross-entropy loss on a dataset like WikiText-103 while your intervention is active.

Which of these two avenues (Safety Overlap or SAE Features) excites you more? Let me know, and we can draft the exact PyTorch/TransformerLens code to run your first experiment.