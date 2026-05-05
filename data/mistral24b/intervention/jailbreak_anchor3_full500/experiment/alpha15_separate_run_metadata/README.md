# Alpha 1.5 Separate Run Note

alpha_1.5.jsonl was generated after the initial Mistral Anchor 3 JBB full-500 run because the original requested alpha list omitted 1.5.

The alpha 1.5 generation used the same model key/path, classifier, locked sample manifest, canonical jailbreak run profile, seed 42, and serial generation settings as the original 0.0/1.0/3.0 run. It was first written to /workspace/mistral24b_anchor3_full500_alpha15_tmp/experiment and merged into the final bundle only after validation.

Because alpha 1.5 was generated in a separate same-seed invocation, its random number generator stream position is not the same as a hypothetical original four-alpha sweep containing 0.0, 1.0, 1.5, and 3.0 in one process.

Generation log: /workspace/02-h-neurons/logs/mistral24b_anchor3_alpha15_generation_20260504T215114Z.log
