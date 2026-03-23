# Scripts — Evaluation Notes

## Jailbreak response truncation bias (max_new_tokens=256)

`run_intervention.py` generates jailbreak responses with `max_new_tokens=256`. This is
too low for the compliant response pattern and introduces a systematic classification
error.

**The pattern:** Jailbreak-compliant responses in this dataset open with a lengthy
disclaimer preamble ("I want to be absolutely clear...", "I strongly condemn...",
"Let me be unequivocal...") before delivering the harmful content. These preambles
routinely consume 200–250 tokens. At max_new_tokens=256 the response is truncated
before the harmful content appears, so the GPT-4o judge sees only the refusal and
returns SAFE.

**Evidence:** During gold-label review, 4 records were provisionally labeled SAFE
from their truncated stored responses (judge=SAFE, compliance=False). Regenerating
at max_new_tokens=1024 with greedy decoding revealed all 4 to be HARMFUL — each
contained specific actionable content (SQL injection steps, child-labour supply-chain
tactics, a three-part disability discrimination campaign, ATM attack vectors) after
the disclaimer block. Full responses are in tests/gold_labels/jailbreak_regen.jsonl.

**Implication for existing results:** Any jailbreak compliance rates derived from
alpha_*.jsonl files are likely underestimates, particularly at lower alphas where
the model is less confidently compliant and more likely to front-load a long
disclaimer. The bias is worst at alpha=0.0 and probably diminishes as alpha
increases (higher-alpha responses tend to be shorter and more directly compliant).

**Fix for future runs:** Use `max_new_tokens=512` (minimum) for jailbreak, preferably
1024. Greedy decoding (`do_sample=False`) is also preferable for gold-label
generation so responses are reproducible.
