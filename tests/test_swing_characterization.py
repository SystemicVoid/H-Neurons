from pathlib import Path

import numpy as np
import pandas as pd

from scripts.characterize_swing import (
    analyze_structural_predictability,
    analyze_transitions,
)
from scripts.export_site_data import build_swing_characterization_payload


def test_analyze_transitions_reports_counts_and_early_share():
    trajectories = {
        "rc_early": [False, True, True, True, True, True, True],
        "rc_late": [False, False, False, False, True, True, True],
        "cr_early": [True, False, False, False, False, False, False],
        "nm": [False, True, False, True, False, True, False],
    }
    swing_ids = list(trajectories)
    subtypes = {
        "rc_early": "R→C",
        "rc_late": "R→C",
        "cr_early": "C→R",
        "nm": "non-monotonic",
    }

    summary = analyze_transitions(trajectories, swing_ids, subtypes)

    rc = summary["transition_alpha"]["R→C"]
    cr = summary["transition_alpha"]["C→R"]

    assert rc["counts_by_alpha"] == {
        "0.5": 1,
        "1.0": 0,
        "1.5": 0,
        "2.0": 1,
        "2.5": 0,
        "3.0": 0,
    }
    assert rc["early_share_le_1_5"]["count"] == 1
    assert rc["early_share_le_1_5"]["n_total"] == 2
    assert cr["counts_by_alpha"]["0.5"] == 1
    assert cr["early_share_le_1_5"]["estimate"] == 1.0


def test_structural_predictability_detects_clear_signal():
    records = []
    for idx in range(30):
        records.append(
            {
                "id": f"always_{idx}",
                "population": "always_compliant",
                "swing_subtype": "",
                "source": "baseline",
                "topic": "physics",
                "question_length": 40 + idx,
                "context_length": 150 + idx,
                "word_overlap": 0.15,
                "num_options": 4,
                "anti_compliance_response_length": 1,
                "standard_response_length": 20,
            }
        )
        records.append(
            {
                "id": f"never_{idx}",
                "population": "never_compliant",
                "swing_subtype": "",
                "source": "baseline",
                "topic": "physics",
                "question_length": 42 + idx,
                "context_length": 155 + idx,
                "word_overlap": 0.18,
                "num_options": 4,
                "anti_compliance_response_length": 1,
                "standard_response_length": 20,
            }
        )
        subtype = "R→C" if idx < 15 else "C→R"
        topic = "biology" if subtype == "R→C" else "chemistry"
        records.append(
            {
                "id": f"swing_{idx}",
                "population": "swing",
                "swing_subtype": subtype,
                "source": "swing_source",
                "topic": topic,
                "question_length": 300 + idx,
                "context_length": 2000 + idx,
                "word_overlap": 0.85 if subtype == "R→C" else 0.65,
                "num_options": 5,
                "anti_compliance_response_length": 1,
                "standard_response_length": 20,
            }
        )

    feature_df = pd.DataFrame(records)
    summary = analyze_structural_predictability(
        feature_df,
        bootstrap_resamples=200,
        permutation_resamples=20,
    )

    swing_task = summary["tasks"]["swing_vs_non_swing"]["feature_sets"]["all_ex_ante"]
    subtype_task = summary["tasks"]["r_to_c_vs_other_swing"]["feature_sets"][
        "all_ex_ante"
    ]

    assert swing_task["metrics"]["auroc"]["estimate"] > 0.95
    assert swing_task["permutation_test"]["metrics"]["auroc"]["p_value"] <= 0.1
    assert subtype_task["metrics"]["auroc"]["estimate"] > 0.8


def test_structural_predictability_stays_near_chance_on_null_data():
    rng = np.random.default_rng(7)
    n_items = 180
    feature_df = pd.DataFrame(
        {
            "id": [f"item_{idx}" for idx in range(n_items)],
            "population": np.where(
                rng.random(n_items) < 0.3, "swing", "always_compliant"
            ),
            "swing_subtype": np.where(rng.random(n_items) < 0.5, "R→C", "C→R"),
            "source": rng.choice(["A", "B", "C"], size=n_items),
            "topic": rng.choice(["biology", "physics", "chemistry"], size=n_items),
            "question_length": rng.normal(100, 15, size=n_items),
            "context_length": rng.normal(500, 40, size=n_items),
            "word_overlap": rng.uniform(0.2, 0.8, size=n_items),
            "num_options": rng.choice([3, 4, 5], size=n_items),
            "anti_compliance_response_length": np.ones(n_items),
            "standard_response_length": np.full(n_items, 20),
        }
    )
    summary = analyze_structural_predictability(
        feature_df,
        bootstrap_resamples=200,
        permutation_resamples=20,
    )
    swing_task = summary["tasks"]["swing_vs_non_swing"]["feature_sets"]["all_ex_ante"]

    assert 0.35 <= swing_task["metrics"]["auroc"]["estimate"] <= 0.65


def test_build_swing_characterization_payload_exports_real_transition_counts():
    repo_root = Path(__file__).resolve().parents[1]
    payload = build_swing_characterization_payload(repo_root)

    rc_series = payload["transition_histogram"]["series"]["R→C"]
    assert payload["schema_version"] == 2
    assert rc_series["counts_by_alpha"] == {
        "0.5": 32,
        "1.0": 15,
        "1.5": 13,
        "2.0": 13,
        "2.5": 15,
        "3.0": 6,
    }
    assert rc_series["early_share_le_1_5"]["count"] == 60
