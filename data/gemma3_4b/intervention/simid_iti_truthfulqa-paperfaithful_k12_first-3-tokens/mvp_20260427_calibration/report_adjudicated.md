# SIMID Report

Generated: 2026-04-27T20:23:53.632833+00:00

All primary estimates use base-item pairing and bootstrap 95% CIs. The primary MC endpoint is the lettered forced-choice likelihood prompt.
Open correctness is reported as adjudicated_open_correct for diagnostics: judge adjudication is used when present, deterministic alias grading is only a fallback for unadjudicated rows, and deterministic alias diagnostics are kept separate. These open metrics remain diagnostic-only until judge calibration evidence is recorded.

Open grade sources: adjudication=4800
Open correctness claimability: blocked (calibration_evidence_not_recorded).
Open correctness claim contract: diagnostic_only_until_judge_calibration_evidence_recorded.

## random_direction_seed1
Pooled aggregate scope: all datasets and MC endpoints in the selected condition panel.
Paired items: 200
Baseline rates: lettered MC=0.5825 [0.5200, 0.6450], adjudicated_open_correct=0.3850 [0.3200, 0.4550]
- alpha -8.0: MC delta -1.50 pp [-3.50, 0.25]; adjudicated_open_correct delta -0.50 pp [-5.00, 4.00]; attempt delta 0.00 pp [-3.00, 3.00]
- alpha 4.0: MC delta 0.25 pp [-0.75, 1.50]; adjudicated_open_correct delta 2.50 pp [-0.50, 6.00]; attempt delta 1.50 pp [0.00, 3.50]
- alpha 8.0: MC delta -0.25 pp [-2.00, 1.50]; adjudicated_open_correct delta 3.00 pp [-1.00, 7.00]; attempt delta 3.50 pp [0.50, 7.00]

Dataset / MC endpoint strata (TruthfulQA leakage split when available):
- triviaqa_bridge / synthetic_mc1: n=100; lettered MC=0.6700 [0.5850, 0.7550]; adjudicated_open_correct=0.4800 [0.3800, 0.5800]
  - alpha -8.0: MC delta 0.50 pp [-1.00, 2.50]; adjudicated_open_correct delta -3.00 pp [-8.00, 2.00]
  - alpha 4.0: MC delta 1.00 pp [0.00, 2.50]; adjudicated_open_correct delta 4.00 pp [0.00, 9.00]
  - alpha 8.0: MC delta -0.50 pp [-3.50, 2.00]; adjudicated_open_correct delta 2.00 pp [-3.00, 7.00]
- truthfulqa / truthfulqa_mc1 (artifact_split=test, seen_in_iti_fit=false, leakage_policy=heldout_only): n=100; lettered MC=0.4950 [0.4050, 0.5850]; adjudicated_open_correct=0.2900 [0.2000, 0.3800]
  - alpha -8.0: MC delta -3.50 pp [-7.00, -0.50]; adjudicated_open_correct delta 2.00 pp [-5.00, 9.00]
  - alpha 4.0: MC delta -0.50 pp [-2.50, 1.00]; adjudicated_open_correct delta 1.00 pp [-3.00, 5.00]
  - alpha 8.0: MC delta 0.00 pp [-2.50, 2.50]; adjudicated_open_correct delta 4.00 pp [-2.00, 10.00]

## random_head_seed1
Pooled aggregate scope: all datasets and MC endpoints in the selected condition panel.
Paired items: 200
Baseline rates: lettered MC=0.5825 [0.5200, 0.6450], adjudicated_open_correct=0.3850 [0.3200, 0.4550]
- alpha -8.0: MC delta -3.25 pp [-5.25, -1.25]; adjudicated_open_correct delta -2.00 pp [-5.50, 1.50]; attempt delta 1.50 pp [-1.50, 5.00]
- alpha 4.0: MC delta 0.75 pp [-0.75, 2.25]; adjudicated_open_correct delta -0.50 pp [-3.50, 3.00]; attempt delta 0.50 pp [-1.50, 2.50]
- alpha 8.0: MC delta -1.50 pp [-4.00, 0.76]; adjudicated_open_correct delta -0.50 pp [-4.50, 3.00]; attempt delta 1.00 pp [-1.50, 4.00]

Dataset / MC endpoint strata (TruthfulQA leakage split when available):
- triviaqa_bridge / synthetic_mc1: n=100; lettered MC=0.6700 [0.5850, 0.7550]; adjudicated_open_correct=0.4800 [0.3800, 0.5800]
  - alpha -8.0: MC delta 0.50 pp [-1.50, 2.50]; adjudicated_open_correct delta -3.00 pp [-8.00, 2.00]
  - alpha 4.0: MC delta 0.50 pp [-1.50, 3.00]; adjudicated_open_correct delta -3.00 pp [-7.00, 0.00]
  - alpha 8.0: MC delta -1.50 pp [-4.50, 1.50]; adjudicated_open_correct delta -5.00 pp [-10.00, -1.00]
- truthfulqa / truthfulqa_mc1 (artifact_split=test, seen_in_iti_fit=false, leakage_policy=heldout_only): n=100; lettered MC=0.4950 [0.4050, 0.5850]; adjudicated_open_correct=0.2900 [0.2000, 0.3800]
  - alpha -8.0: MC delta -7.00 pp [-10.50, -4.00]; adjudicated_open_correct delta -1.00 pp [-6.00, 4.00]
  - alpha 4.0: MC delta 1.00 pp [-1.00, 3.00]; adjudicated_open_correct delta 2.00 pp [-3.00, 8.00]
  - alpha 8.0: MC delta -1.50 pp [-5.50, 2.00]; adjudicated_open_correct delta 4.00 pp [-2.00, 10.00]

## selected
Pooled aggregate scope: all datasets and MC endpoints in the selected condition panel.
Paired items: 200
Baseline rates: lettered MC=0.5825 [0.5200, 0.6450], adjudicated_open_correct=0.3850 [0.3200, 0.4550]
- alpha -8.0: MC delta -2.50 pp [-4.50, -0.75]; adjudicated_open_correct delta -6.00 pp [-10.50, -1.50]; attempt delta 5.00 pp [2.00, 8.50]
- alpha 4.0: MC delta -0.25 pp [-1.50, 1.00]; adjudicated_open_correct delta 2.00 pp [-2.00, 6.00]; attempt delta -2.50 pp [-5.00, 0.00]
- alpha 8.0: MC delta -2.25 pp [-4.50, 0.00]; adjudicated_open_correct delta 3.00 pp [-2.00, 8.50]; attempt delta -1.00 pp [-4.50, 2.50]

Dataset / MC endpoint strata (TruthfulQA leakage split when available):
- triviaqa_bridge / synthetic_mc1: n=100; lettered MC=0.6700 [0.5850, 0.7550]; adjudicated_open_correct=0.4800 [0.3800, 0.5800]
  - alpha -8.0: MC delta 1.00 pp [-1.00, 3.00]; adjudicated_open_correct delta -5.00 pp [-11.00, 1.00]
  - alpha 4.0: MC delta 0.50 pp [-1.00, 2.50]; adjudicated_open_correct delta -3.00 pp [-8.00, 1.00]
  - alpha 8.0: MC delta -1.00 pp [-4.00, 1.50]; adjudicated_open_correct delta -6.00 pp [-12.00, 0.00]
- truthfulqa / truthfulqa_mc1 (artifact_split=test, seen_in_iti_fit=false, leakage_policy=heldout_only): n=100; lettered MC=0.4950 [0.4050, 0.5850]; adjudicated_open_correct=0.2900 [0.2000, 0.3800]
  - alpha -8.0: MC delta -6.00 pp [-9.50, -3.00]; adjudicated_open_correct delta -7.00 pp [-14.00, 0.00]
  - alpha 4.0: MC delta -1.00 pp [-3.00, 1.00]; adjudicated_open_correct delta 7.00 pp [0.00, 14.00]
  - alpha 8.0: MC delta -3.50 pp [-7.00, -0.50]; adjudicated_open_correct delta 12.00 pp [4.00, 20.00]

## Selected Minus Control Slopes
- random_direction_seed1:
  - mc_full_margin: -0.0022 [-0.0522, 0.0489]
  - open_first3_margin: 0.2285 [0.0707, 0.3903]
  - open_full_margin: 0.1765 [0.0119, 0.3465]
- random_head_seed1:
  - mc_full_margin: -0.0494 [-0.0976, -0.0015]
  - open_first3_margin: 0.1218 [-0.0257, 0.2708]
  - open_full_margin: 0.1119 [-0.0601, 0.2800]

## Phase 0 Gates
- bridge_synthetic_mc_sanity: PASS
- bridge_option_order_stability: PASS (global replicate spread=0.0000; item flips=10/100, rate=0.1000)
  - replicate rates: ord:0=0.6700, ord:1=0.6700
  - chosen letters: A=44, B=58, C=50, D=48
  - gold-position correctness: pos 0: 26/41=0.6341; pos 1: 40/57=0.7018; pos 2: 32/43=0.7442; pos 3: 36/59=0.6102
  - flipped base items: simid_bridge_bb_1342, simid_bridge_bb_3171, simid_bridge_bb_5777, simid_bridge_bt_1604, simid_bridge_bt_747, simid_bridge_bt_958, simid_bridge_dpql_1550, simid_bridge_dpql_4551, simid_bridge_dpql_4988, simid_bridge_dpql_6155
- bridge_gold_position_balance: PASS
- bridge_option_length_balance: PASS
- bridge_open_margin_alignment: PASS
