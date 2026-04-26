# SIMID Report

Generated: 2026-04-26T11:41:24.880349+00:00

All primary estimates use base-item pairing and bootstrap 95% CIs. The primary MC endpoint is the lettered forced-choice likelihood prompt.
Open correctness is deterministic alias-grader correctness (not adjudicated human/judge correctness). Claimable open-ended results require adjudication or an explicit audit.

## unhooked
Pooled aggregate scope: all datasets and MC endpoints in the selected condition panel.
Paired items: 16
Baseline rates: lettered MC=0.6875 [0.4688, 0.8750], open=0.3750 [0.1250, 0.6250]

Dataset / MC endpoint strata (TruthfulQA leakage split when available):
- triviaqa_bridge / synthetic_mc1: n=8; lettered MC=0.9375 [0.8125, 1.0000]; open=0.7500 [0.5000, 1.0000]
- truthfulqa / truthfulqa_mc1 (artifact_split=test, seen_in_iti_fit=false, leakage_policy=allow_fitted): n=4; lettered MC=0.6250 [0.2500, 1.0000]; open=0.0000 [0.0000, 0.0000]
- truthfulqa / truthfulqa_mc1 (artifact_split=train, seen_in_iti_fit=true, leakage_policy=allow_fitted): n=4; lettered MC=0.2500 [0.0000, 0.7500]; open=0.0000 [0.0000, 0.0000]

## selected
Pooled aggregate scope: all datasets and MC endpoints in the selected condition panel.
Paired items: 16
Baseline rates: lettered MC=0.6875 [0.4688, 0.8750], open=0.3750 [0.1250, 0.6250]

Dataset / MC endpoint strata (TruthfulQA leakage split when available):
- triviaqa_bridge / synthetic_mc1: n=8; lettered MC=0.9375 [0.8125, 1.0000]; open=0.7500 [0.5000, 1.0000]
- truthfulqa / truthfulqa_mc1 (artifact_split=test, seen_in_iti_fit=false, leakage_policy=allow_fitted): n=4; lettered MC=0.6250 [0.2500, 1.0000]; open=0.0000 [0.0000, 0.0000]
- truthfulqa / truthfulqa_mc1 (artifact_split=train, seen_in_iti_fit=true, leakage_policy=allow_fitted): n=4; lettered MC=0.2500 [0.0000, 0.7500]; open=0.0000 [0.0000, 0.0000]

## Phase 0 Gates
- noop_equivalence: PASS
- bridge_synthetic_mc_sanity: PASS
- bridge_option_order_stability: PASS (global replicate spread=0.1250; item flips=1/8, rate=0.1250)
  - replicate rates: ord:0=0.8750, ord:1=1.0000
  - chosen letters: A=1, B=4, C=3, D=8
  - gold-position correctness: pos 0: 1/2=0.5000; pos 1: 4/4=1.0000; pos 2: 3/3=1.0000; pos 3: 7/7=1.0000
  - flipped base items: simid_bridge_bb_1342
- bridge_gold_position_balance: PASS
- bridge_option_length_balance: PASS
- bridge_open_margin_alignment: PASS

