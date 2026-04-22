# FaithEval SAE Utility-Positive Augment Audit

- Augment slice: k=154 strictly-positive-utility SAE features + 3 layer-matched zero-weight seeds at α=0.0.
- Reference noop (α=1.0) reused from Phase 1 selector bundle.
- Review loophole targeted: Size-matched k=266 utility family crosses selector_score=0 (only 154/509 features have strictly positive validation utility). This slice selects only the positive-utility features so paired comparisons cannot be attacked as 'diluted by neutral or harmful features to match readout cardinality'.
- FaithEval compliance families: noop=0.6643 (558/840); utility_positive_selected=0.6655 (559/840); matched_random_positive_seed_0=0.6655 (559/840); matched_random_positive_seed_1=0.6655 (559/840); matched_random_positive_seed_2=0.6655 (559/840)
- FaithEval compliance paired deltas: utility_positive_minus_noop=+0.119 pp [-1.310, +1.548]; utility_positive_minus_matched_random_positive_seed_0=+0.000 pp [-1.429, +1.429]; utility_positive_minus_matched_random_positive_seed_1=+0.000 pp [-1.429, +1.429]; utility_positive_minus_matched_random_positive_seed_2=+0.000 pp [-1.429, +1.429]
- Anti-compliance margin families: noop=+8.3090 [+7.1014, +9.4986]; utility_positive_selected=+7.5859 [+6.5124, +8.6375]; matched_random_positive_seed_0=+7.9930 [+6.8392, +9.1293]; matched_random_positive_seed_1=+7.9930 [+6.8392, +9.1293]; matched_random_positive_seed_2=+7.9930 [+6.8392, +9.1293]
- Anti-compliance margin paired deltas: utility_positive_minus_noop=-0.7231 [-1.0414, -0.3846]; utility_positive_minus_matched_random_positive_seed_0=-0.4071 [-0.7149, -0.0946]; utility_positive_minus_matched_random_positive_seed_1=-0.4071 [-0.7149, -0.0946]; utility_positive_minus_matched_random_positive_seed_2=-0.4071 [-0.7149, -0.0946]
- Utility-positive layer histogram: {"0": 18, "13": 19, "14": 14, "15": 7, "16": 17, "17": 21, "20": 50, "5": 2, "6": 3, "7": 3}
- Main utility-selected (k=266) layer histogram: {"0": 19, "13": 25, "14": 18, "15": 17, "16": 29, "17": 40, "20": 109, "5": 2, "6": 3, "7": 4}
- Candidate pool layer histogram: {"0": 39, "13": 49, "14": 42, "15": 38, "16": 49, "17": 75, "20": 189, "5": 6, "6": 9, "7": 13}
- Utility-positive weight-sign counts: {"negative": 75, "positive": 79}
- Utility-positive selector-score range: min=7.45058e-10, mean=0.0183055
