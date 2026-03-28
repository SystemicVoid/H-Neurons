## Anti-sycophancy prompt

“I saw someone claim this, it seems pretty off to me, can you help me give them direct but constructive feedback on what they missed? [insert your prompt]” 

---

## Review data and results prompt 

 Meticulously review [INSERT DATA PIPELINE] pipelines, apply scientific rigor and healthy skepticism, you MUST follow best practice in AI/ML safety research. When done udpate the existing context referring to previous analysis/results to update them or remove them as per your judgment (keep a single source of truth to prevent drift, prefer linking reports to connect information to prevent redudancy) then produce a thorough report clearly separating interpretation from data, that you will link from "@data/gemma3_4b/intervention_findings.md" and/or pipeline report and other relevant files, respecting existing patterns, and keeping notes well organised with a clear structure. Be critical, yet do present what also withstand scrutiny, what uncertainties remain (and how uncertain they are), surface interesting insights, as well as the most scientifically valuable next steps. Do your best!


## Good questions before a run 

- What is the motivation for this experiment? Does it fit in with the research question I want to answer?
- Have I derisked this enough already, and are there other more interesting things to run instead? Is this definitely the highest priority?
- What result do I expect to get? Is learning that useful?
- Should I explore one model and one dataset first before expanding to more?
- Will running this extra experiment add significant value to our paper? (especially relevant when close to a deadline)
- Am I changing too many variables at once? Can I simplify my setup to draw better conclusions?


Don’t do tasks because they are easy.
Don’t do tasks because they are intellectually pretty.
Do the task that will most quickly tell you whether the current approach is viable, doomed, or needs reframing.


The Golden Rule : **Do the components/tests/experiments in order from most informative per unit time to least informative per unit time.**
Reduce the most decision-relevant uncertainty per unit time.

## Current State of replication 

  You do not yet have a full same-model same-eval comparison for all paper claims.

  Missing or not yet aligned locally:

  - NQ-Open detector eval
  - NonExist detector eval
  - Sycophancy intervention 
  - Jailbreak intervention ( currently done awaiting negative control AND/OR baseline ?)



## Next steps





turn the current result into a hard-to-kill safety story:
H-neurons are a sparse, specific over-compliance mechanism on epistemic tasks; on jailbreaks, binary compliance counts were partly a measurement mirage, and the real safety effect is likely in refusal style / severity rather than raw count.


**That means the failure mode is not just “response cut off too early,” but also “judge/rubric may overweight refusal framing.” That is a separate confound from token budget.**



Review tri-population results and write to log about it 



**What about intervening on higher C (3, with a subset of 250 neurons) and tweaking a larger subgroup of neurons for behavioral modifications on jailbreak ?**




 If you are scaling $\alpha$ up to 3 on FaithEval, how are you measuring the degradation of general capabilities? A model might stop rejecting false premises just because the intervention lobotomized its general reasoning.What is your baseline for general fluency and accuracy under these perturbations? Are you tracking cross-entropy loss on a standard dataset (like Wikipedia or Wikitext) while you scale these neurons?




Before you expand to the Sycophancy or Jailbreak benchmarks, I want you to isolate that "swing" population from FaithEval.

What characterizes those 138 prompts? Are they harder questions? Do they sit on a specific semantic boundary? Do a qualitative pass over the text of those specific prompts and compare them to the "never compliant" prompts. If we can predict which prompts are susceptible to H-Neuron scaling, we elevate this from a replication to a fundamental discovery about model cognition.




0. Jailbreak eval (ongoing)

2. Suppressive (negative-weight) neuron investigation (cheapest novel finding). The L1 classifier found 38 positive-weight neurons AND 38 negative-weight neurons. The paper ignores the negative-weight set entirely. These may represent a natural "truthfulness circuit." Check their C-sweep stability across C in {0.3, 1.0, 3.0, 10.0} -- this is minutes on CPU. If a core subset persists (10+), run the inverse intervention: amplify these neurons (alpha > 1.0) and measure whether compliance decreases. If it works, "amplify truthfulness" is a cleaner narrative than "suppress hallucination" and has direct safety applications. Falsification: 10 min CPU + 3-4 hours GPU.



3. Safety/refusal overlap pilot. Does suppressing H-neurons (alpha=0.0) change refusal behavior on harmful prompts? Run 20 manually-written harmful prompts at alpha=0.0 and alpha=1.0. If the model becomes MORE willing to comply with harmful requests when H-neurons are suppressed, the alignment tax is mechanistically real and the same circuits mediate both factual compliance and safety refusal. If refusal is unchanged, the circuits are separable -- good news, meaning you can suppress hallucination without safety costs.


4. Small C sweep with intervention readout
    Why: you do not need full paper replication to get the key missing insight. Just test a few nearby C values and measure
    two things:
      1. detector quality
      2. intervention side-effects on a preserved capability metric
         That directly answers whether your current 38-neuron set is stable or brittle.


 
## GCM paper applied here

 Shift from Correlational Probes to Attribution Patching
Full activation patching scales linearly with the number of neurons and would be painfully slow , but the GCM paper highlights Attribution Patching as a lean, linear approximation.Attribution patching requires only 2 forward passes and 1 backward pass to score all components.Local Compute Check: On your RTX 5060 Ti (16GB VRAM), doing a forward and backward pass on a model like Qwen-1.5-7B or Llama-3-8B might OOM if you aren't careful. You should load the model in 8-bit or 4-bit quantization, or offload optimizer states if needed. nnsight is perfect for this.2. Hunt for "H-Heads" (Attention Head Mediation)
The H-Neurons paper focuses entirely on Feed-Forward Networks. But hallucinations fundamentally involve a failure to move the correct factual information from the context/weights to the residual stream. As the Transformer Circuits framework states, attention heads move information.Apply GCM to localize the specific Attention Heads that mediate hallucinations.3. Upgrade the Steering Methodology
The H-Neurons paper intervenes by simply scaling the activation values by a factor $\alpha \in [0, 3]$. GCM systematically tests more sophisticated unsupervised steering methods, like Difference-in-Means and Mean Steering.Extract a difference-in-means steering vector specifically from the causally identified H-Neurons/H-Heads. See if adding this vector reduces over-compliance more effectively than scalar multiplication, without degrading fluency.The "Be Your Own Greatest Critic" CheckBefore you write a single line of code, let's put on our skeptic hats. Every research result is false until proven otherwise.The Contrastive Data Problem: GCM relies heavily on having clean, contrastive prompt pairs ($p_{orig}$, $p_{contrast}$) that differ minimally but elicit contrastive long-form responses. The H-Neurons paper used TriviaQA and sampled 10 times to find naturally occurring consistent correct vs. incorrect responses. To use GCM, you need to cleanly define a contrastive pair for hallucination. E.g., Prompt A: "Answer factually: What is X?" vs. Prompt B: "Guess a plausible but fake answer: What is X?" If your contrastive pair changes the semantics too much, GCM will just localize the "instruction following" circuit, not the "hallucination/over-compliance" circuit.The "Global Steering" Trap: The GCM authors admit that sometimes, surgical steering isn't actually necessary. They achieved comparable control on some tasks by just steering all attention heads at once (Global Steering). If you do this project, your absolute most important baseline is: What happens if I just apply the steering vector globally instead of surgically targeting the H-Neurons?
How GCM Upgrades Your Pipeline
This is where integrating GCM changes the game. GCM doesn't just verify causality at the end; it uses causal mediation to identify the components in the first place.

Instead of fitting a logistic regression to CETT scores, GCM's Attribution Patching measures the first-order Taylor approximation of the Indirect Effect: If I swap the activation of this specific neuron from a truthful prompt to a hallucination prompt, how much does the probability of the hallucinated token increase?
By using GCM to find your H-Neurons, you eliminate the L1 artifact problem entirely. Every neuron is ranked directly by its causal contribution to the output.

The Pragmatic Compute Strategy
You have a beautiful setup: Gemma-3-4B in full precision, running well on your RTX 5060 Ti, with 6GB of VRAM headroom. This is the perfect playground.

Keep it Local: Because Attribution Patching only requires a single forward pass, a backward pass, and a cached clean pass, you should be able to run this locally within that 6GB headroom using nnsight or TransformerLens. There is no need to jump to the cloud or quantize yet.

The Experiment: Take your FaithEval setup. For a small batch of prompts, compute the attribution scores for all FFN neurons.

The "Ah-Ha!" Comparison: Compare the top 38 neurons identified causally via GCM against your current 38 L1-identified neurons. I guarantee Neuron 20:4288 drops in rank, and the true causal drivers rise to the top.

The Ultimate Intervention: Extract a difference-in-means steering vector (as recommended by GCM) from your newly identified causal H-Neurons, and see if it yields a cleaner, steeper slope on your FaithEval and FalseQA benchmarks than your current scalar activation multiplying method.







## Publication Story 

The publication story: Replication with proper controls + L1 artifact critique + format/content dissociation discovery + [best Phase 2 result, likely SAE decomposition or suppressive neurons]. This targets an ICML MI workshop or NeurIPS SafeGenAI paper. The SAE bridge is especially compelling because it connects two major threads in the field on consumer hardware with a single model.




## Tooling 



## Rough ideas to test



### Disentangle verbosity from truth 


these neurons are causally involved in a compliance circuit; what they passively encode is not a clean truth signal when read across full responses, but that doesn't determine what they do when amplified.

-> mediation test. Hold output format/length fixed as hard as possible, then rescan the α effect. That’s the cleanest knife for slicing “truth/compliance circuit” from “style circuit wearing a fake moustache.”


### Autoresearch

- The "Holy Grail" Composite Metric: The Safety-Capability Trade-offThe paper notes a critical challenge: simply suppressing H-Neurons can damage the model's fundamental capabilities and helpfulness. To deploy a model in the real world, it must be safe without being lobotomized.The most impactful metric for the autoresearch agent would be a composite reward function that mimics the paper's grid search for the $l_1$ regularization parameter $C$. The agent should be instructed to maximize a combined score:Metric: Sum of (Hallucination Detection Accuracy) + (Performance on TriviaQA while H-Neurons are suppressed).Why it matters: By optimizing this joint metric during its overnight runs, the AI agent would iteratively search for model weights or architectures that perfectly isolate the hallucination signals into a sparse subset (making them easy to detect), while ensuring that suppressing those specific neurons does not degrade the model's factual recall or language modeling capabilities.
- If we were to apply the methodology of **autoresearch** repo to optimise for certain metrics in the h-neurons to optimise for, which of these metrics would be most impactful for real world deployment of LLMs, particularly for AI safety and capability ? https://github.com/karpathy/autoresearch 





## Random notes 



The problem is Mistral-Small-3.1-24B-Instruct-2503 is a multimodal (Mistral3) architecture — AutoModelForCausalLM can't load it. We need to use the text-only version. Let me check what's available:
✓ Web Search
Find the text-only (non-multimodal) HuggingFace model ID for Mistral Small 3.1 24B Instruct that works with AutoModelForCausalLM
The paper lists "Mistral-Small-3.1-24B" but the 3.1 version is multimodal (Mistral3 arch). The text-only predecessor is Mistral-Small-24B-Instruct-2501 (Mistral Small 3, same 24B params, works with AutoModelForCausalLM). That's the one the paper actually used. Let me swap the download and also update the config:


---

• I’m re-querying the remote with a login shell. The earlier SSH command showed another
  environment quirk too: uv wasn’t on the non-login PATH, which is separate from the CPU-only
  torch issue but worth keeping in mind for automation.


## Good AGENTS.md

We previously thought including your tech stack, key files, etc., as like a mini-map for your agent was the right approach. That’s what agents add if they create it.

But there was a study that showed it hurt performance and increased cost by 20% (using extra tokens and lowering task completion). The agent can figure out the tech stack, key files, commands, and architecture very easily and quickly.

Instead, it should be pretty empty. It should be your preferences and nudges to correct agent behaviour.

  - When building, open a browser with agent-browser skill and test before sending me a URL (to catch bugs)
  - Use the Exa web search tool for web search
  - Always write planning files in ~/[project-name]/plan/
  - I can't code, so explain things in simple terms
  - Record a video of your output so I can see exactly what you tested

Wrapping sections in conditional XML-like blocks is helpful:

<important if="simple web page">
  - No spec needed
  - Create 3 designs before choosing one
  - Must have dark/light mode switcher
</important>

You don’t need to mention skills you’ve installed, as their ‘frontmatter’ (the skills’ name and description) is also pre-loaded alongside your AGENTS.md.