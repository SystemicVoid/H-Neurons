# SIMID Report

Generated: 2026-04-30T10:44:29.484012+00:00

All primary estimates use base-item pairing and bootstrap 95% CIs. The primary MC endpoint is the lettered forced-choice likelihood prompt.
Open correctness is deterministic alias-grader correctness (not adjudicated human/judge correctness). Claimable open-ended results require adjudication or an explicit audit.

Open grade sources: deterministic_alias=1708
Open correctness claimability: blocked (judge_adjudication_not_loaded).
Open correctness claim contract: diagnostic_only_until_judge_calibration_evidence_recorded.

## unhooked
Pooled aggregate scope: all datasets and MC endpoints in the selected condition panel.
Paired items: 427
Baseline rates: lettered MC=0.5187 [0.4731, 0.5621], open_correct=0.2155 [0.1780, 0.2553]

Dataset / MC endpoint strata (TruthfulQA leakage split when available):
- triviaqa_bridge / synthetic_mc1: n=200; lettered MC=0.6125 [0.5499, 0.6700]; open_correct=0.4000 [0.3350, 0.4700]
- truthfulqa / truthfulqa_mc1 (artifact_split=test, seen_in_iti_fit=false, leakage_policy=heldout_only): n=227; lettered MC=0.4361 [0.3767, 0.4956]; open_correct=0.0529 [0.0264, 0.0837]

## selected
Pooled aggregate scope: all datasets and MC endpoints in the selected condition panel.
Paired items: 427
Baseline rates: lettered MC=0.5187 [0.4731, 0.5621], open_correct=0.2155 [0.1780, 0.2553]

Dataset / MC endpoint strata (TruthfulQA leakage split when available):
- triviaqa_bridge / synthetic_mc1: n=200; lettered MC=0.6125 [0.5499, 0.6700]; open_correct=0.4000 [0.3350, 0.4700]
- truthfulqa / truthfulqa_mc1 (artifact_split=test, seen_in_iti_fit=false, leakage_policy=heldout_only): n=227; lettered MC=0.4361 [0.3767, 0.4956]; open_correct=0.0529 [0.0264, 0.0837]

## Phase 0 Gates
- noop_equivalence: PASS
- bridge_synthetic_mc_sanity: PASS
- bridge_option_order_stability: PASS (global replicate spread=0.0050; item flips=33/200, rate=0.1650)
  - replicate rates: ord:0=0.6150, ord:1=0.6100
  - chosen letters: A=109, B=109, C=94, D=88
  - gold-position correctness: pos 0: 63/104=0.6058; pos 1: 62/92=0.6739; pos 2: 59/101=0.5842; pos 3: 61/103=0.5922
  - flipped base items: simid_bridge_odql_1087, simid_bridge_odql_12415, simid_bridge_odql_13356, simid_bridge_odql_14594, simid_bridge_odql_1925, simid_bridge_odql_257, simid_bridge_odql_5788, simid_bridge_odql_6602, simid_bridge_odql_6763, simid_bridge_odql_8669 ...
- bridge_gold_position_balance: PASS
- bridge_option_length_balance: PASS
- bridge_open_margin_alignment: PASS
