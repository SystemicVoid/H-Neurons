# Pilot 100 — Generation Settings Caveat

**Generated 2026-04-05 with incorrect decode settings:**

- `max_new_tokens=1024` (should be 5000 — see week3-log 256→5000 truncation-bias finding)
- `do_sample=False, temperature=0.0, top_k=1, top_p=1.0` (greedy; canonical jailbreak eval uses `do_sample=True, temperature=0.7`)

## Impact assessment

**Alpha locking (relative ranking) is likely still valid:** The same truncation and decode bias
applies symmetrically to all alphas within each selector, so the *ordering* of alphas by csv2_yes
decrease is probably stable. Both selectors locked α=8.0.

**Absolute csv2_yes rates are NOT comparable to the full 500 run** which uses correct settings
(`max_new_tokens=5000`, `do_sample=True, temperature=0.7`). Do not mix pilot absolute numbers
with full-run numbers in any table or claim.

**Do not re-run the pilot** unless the full 500 results raise doubts about the alpha selection.
The pilot's purpose was alpha locking, not claims.
