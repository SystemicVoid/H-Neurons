# FaithEval SAE Utility Selector Audit

- Held-out diagnostic bundle on a single locked test manifest (n=840).
- FaithEval families: noop=0.6643 (558/840); readout_selected=0.6595 (554/840); utility_selected=0.6619 (556/840); matched_random_seed_0=0.6548 (550/840); matched_random_seed_1=0.6643 (558/840); matched_random_seed_2=0.6643 (558/840)
- FaithEval paired deltas: utility_minus_readout=+0.238 pp [-1.310, +1.786]; utility_minus_noop=-0.238 pp [-1.667, +1.190]; utility_minus_matched_random_seed_0=+0.714 pp [-0.833, +2.262]; utility_minus_matched_random_seed_1=-0.238 pp [-1.667, +1.190]; utility_minus_matched_random_seed_2=-0.238 pp [-1.786, +1.310]
- FaithEval anti-compliance margin families: noop=+8.3090 [+7.1014, +9.4986]; readout_selected=+9.2274 [+7.8619, +10.5731]; utility_selected=+7.5461 [+6.4747, +8.5985]; matched_random_seed_0=+8.1976 [+6.9860, +9.3901]; matched_random_seed_1=+8.7536 [+7.4852, +10.0019]; matched_random_seed_2=+7.4036 [+6.3303, +8.4736]
- FaithEval anti-compliance margin paired deltas: utility_minus_readout=-1.6813 [-2.1707, -1.1897]; utility_minus_noop=-0.7628 [-1.0839, -0.4224]; utility_minus_matched_random_seed_0=-0.6515 [-1.0262, -0.2728]; utility_minus_matched_random_seed_1=-1.2075 [-1.5748, -0.8221]; utility_minus_matched_random_seed_2=+0.1425 [-0.1843, +0.4776]
- Candidate pool scope: 509 non-zero probe-support SAE features from the existing extraction scope; layer histogram={"0": 39, "13": 49, "14": 42, "15": 38, "16": 49, "17": 75, "20": 189, "5": 6, "6": 9, "7": 13}
- Candidate-pool sign counts: {"negative": 243, "positive": 266}
- Utility-selected sign counts: {"negative": 134, "positive": 132}
- Readout-selected sign counts: {"unknown": 266}
- Outside old |w|>1e-3 shortlist: 75 / 266 (0.2820)
- Utility vs readout overlap: {"intersection_count": 132, "jaccard": 0.33, "union_count": 400}
- Utility layer histogram: {"0": 19, "13": 25, "14": 18, "15": 17, "16": 29, "17": 40, "20": 109, "5": 2, "6": 3, "7": 4}
- Readout layer histogram: {"0": 26, "13": 26, "14": 23, "15": 19, "16": 28, "17": 39, "20": 93, "5": 3, "6": 4, "7": 5}
- Target-family definitions: {"matched_random": "Zero-weight SAE features sampled without replacement from the token-active pool, exact-matched to the utility-selected layer histogram, with within-layer weights proportional to full-sequence token activation rate on the frozen validation split.", "readout_selected": "Top positive probe-weight SAE features from the original FaithEval readout.", "utility_selected": "Top-k SAE features ranked by validation reduction in misleading-minus-preferred logprob margin."}
- Layer-coverage status: Partial L3 closure only: selection searches all non-zero probe-support features within the existing SAE extraction layers, not a wider SAE sweep.
