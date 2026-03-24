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

## CRITICAL: Never touch output directories while a GPU run is writing

<important>
**Do not create, delete, move, rename, or `git checkout`/`git rm` any file inside an
active run's output directory while the run is in progress.** This includes commits
that "clear old experiment data" or archive directories.

**Why this kills data:** On Linux, deleting a file that a process has open via `fd`
unlinks the directory entry but the kernel keeps the inode alive until the last `fd`
closes. The process continues writing successfully (no error, no signal), but the
data lands on an orphaned inode with no path. When the `fd` closes, the kernel frees
the inode and all written data is irrecoverably lost. `f.flush()` and `fsync()` do
not help — the writes reach the kernel but the inode has no directory entry.

**Incident:** On 2026-03-23, commits 082a82f et al. were made at 20:21–20:24 UTC
while `run_intervention.py` was actively writing `alpha_0.0.jsonl` (started 20:17).
The commit cleared the experiment directory, unlinking the file. The run completed
all 500 samples over ~4 hours with no errors, but the data was written to a dead
inode. Result: 5 hours of GPU time lost, entire alpha=0.0 pass destroyed.

**Rules for agents:**

1. Before any `git add`, `git rm`, `git checkout`, `mv`, `rm`, or directory
   restructuring, check whether a GPU job is running that writes to the affected
   path. Use `ps aux | grep run_intervention` or check tmux panes.
2. If a run is active, do **not** touch its `output_dir` or any parent directory
   in the path. Wait for the run to finish.
3. If you need to archive or restructure data, do it **before** launching the run
   or **after** the run completes and the process has exited.
4. The provenance sidecar (`run_intervention.provenance.json`) in the output
   directory has `"status": "running"` while the job is active. Check it.
5. This applies to **all** benchmark output directories, not just jailbreak.
</important>
