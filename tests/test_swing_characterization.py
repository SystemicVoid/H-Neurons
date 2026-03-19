import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

from scripts.characterize_swing import (
    analyze_structural_predictability,
    analyze_transitions,
    summarize_llm_enrichment,
)
from scripts.export_site_data import (
    build_classifier_site_payload,
    build_classifier_structure_summary_payload,
    build_swing_characterization_payload,
    coefficient_sha256,
    compact_llm_enrichment,
    validate_classifier_structure_summary,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


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


def test_build_classifier_site_payload_exports_model_structure():
    repo_root = Path(__file__).resolve().parents[1]
    payload = build_classifier_site_payload(repo_root)

    structure = payload["selected_h_neuron_structure"]
    top_positive = structure["top_positive_neurons"]

    assert payload["schema_version"] == 2
    assert len(structure["positive_counts_by_layer"]) == 34
    assert sum(structure["positive_counts_by_layer"]) == payload["selected_h_neurons"]
    assert structure["bands"]["early"]["count"] == 18
    assert structure["bands"]["middle"]["count"] == 10
    assert structure["bands"]["late"]["count"] == 10
    assert top_positive[0]["label"] == "L20:N4288"
    assert top_positive[0]["weight"] == 12.169
    assert top_positive[1]["label"] == "L14:N8547"
    assert top_positive[1]["weight"] == 7.386
    assert [entry["weight"] for entry in top_positive] == sorted(
        (entry["weight"] for entry in top_positive),
        reverse=True,
    )


def test_results_page_intervention_narrative_uses_live_bindings():
    repo_root = Path(__file__).resolve().parents[1]
    results_html = (repo_root / "site/results/gemma-3-4b.html").read_text()
    charts_js = (repo_root / "site/assets/charts.js").read_text()

    bindings = [
        "parse-peak-count",
        "parse-peak-alpha",
        "strict-remap-recovered-count",
        "strict-remap-reviewed-count",
        "strict-remap-recovery-rate",
        "frozen-count",
        "frozen-share-value",
        "always-compliant-count",
        "never-compliant-count",
        "swing-alpha-three-compliant-count",
        "swing-alpha-three-resistant-count",
    ]

    for binding in bindings:
        assert f'data-intervention-bind="{binding}"' in results_html
        assert f"'{binding}'" in charts_js

    assert "150 failures at &alpha;=3.0" not in results_html
    assert "<h2>Most samples are unaffected by scaling</h2>" not in results_html


def test_build_classifier_site_payload_uses_tracked_structure_summary(tmp_path: Path):
    repo_root = tmp_path
    write_json(
        repo_root / "data/gemma3_4b/pipeline/classifier_disjoint_summary.json",
        {
            "model_path": "google/gemma-3-4b-it",
            "selected_h_neurons": 10,
            "selected_ratio_per_mille": 0.1,
            "total_ffn_neurons": 34,
            "evaluation": {
                "n_examples": 10,
                "n_positive": 5,
                "n_negative": 5,
                "metrics": {"accuracy": {"estimate": 0.8}},
                "bootstrap": {},
                "confusion_matrix": {},
            },
        },
    )
    write_json(
        repo_root / "data/gemma3_4b/pipeline/classifier_overlap_summary.json",
        {
            "evaluation": {
                "n_examples": 12,
                "n_positive": 6,
                "n_negative": 6,
                "metrics": {"accuracy": {"estimate": 0.9}},
                "bootstrap": {},
                "confusion_matrix": {},
            }
        },
    )
    write_json(
        repo_root / "data/gemma3_4b/pipeline/test_qids_disjoint.json",
        {"group_a": ["q1", "q2", "q3"]},
    )
    tracked_summary = {
        "schema_version": 1,
        "generated_at": "2026-03-19",
        "generated_by": "scripts/export_site_data.py",
        "model": "google/gemma-3-4b-it",
        "model_path": "models/gemma3_4b_classifier_disjoint.pkl",
        "generation_script": "scripts/export_site_data.py",
        "source_files": [
            "data/gemma3_4b/pipeline/classifier_disjoint_summary.json",
            "models/gemma3_4b_classifier_disjoint.pkl",
        ],
        "selected_h_neurons": 10,
        "total_ffn_neurons": 34,
        "coefficient_sha256": "f" * 64,
        "structure": {
            "n_layers": 34,
            "neurons_per_layer": 1,
            "positive_counts_by_layer": [1] * 10 + [0] * 24,
            "nonzero_layers": [{"layer": layer, "count": 1} for layer in range(10)],
            "bands": {
                "early": {
                    "label": "early",
                    "start_layer": 0,
                    "end_layer": 10,
                    "count": 10,
                    "pct": 100.0,
                },
                "middle": {
                    "label": "middle",
                    "start_layer": 11,
                    "end_layer": 20,
                    "count": 0,
                    "pct": 0.0,
                },
                "late": {
                    "label": "late",
                    "start_layer": 21,
                    "end_layer": 33,
                    "count": 0,
                    "pct": 0.0,
                },
            },
            "top_positive_neurons": [
                {
                    "rank": rank,
                    "layer": rank - 1,
                    "neuron": 0,
                    "label": f"L{rank - 1}:N0",
                    "weight": round(10.0 - rank * 0.1, 3),
                }
                for rank in range(1, 11)
            ],
        },
    }
    write_json(
        repo_root / "data/gemma3_4b/pipeline/classifier_structure_summary.json",
        tracked_summary,
    )

    payload = build_classifier_site_payload(repo_root)

    assert payload["selected_h_neuron_structure"] == tracked_summary["structure"]
    assert payload["source_files"][-1] == (
        "data/gemma3_4b/pipeline/classifier_structure_summary.json"
    )


def test_build_classifier_site_payload_rejects_stale_tracked_structure_when_local_checkpoint_exists(
    tmp_path: Path,
):
    repo_root = tmp_path
    coef = np.array([9.0 - idx * 0.5 for idx in range(10)] + [-1.0] * 24, dtype=float)
    checkpoint_path = repo_root / "models/gemma3_4b_classifier_disjoint.pkl"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(SimpleNamespace(coef_=np.array([coef], dtype=float)), checkpoint_path)
    write_json(
        repo_root / "data/gemma3_4b/pipeline/classifier_disjoint_summary.json",
        {
            "model_path": "google/gemma-3-4b-it",
            "loaded_model_path": "models/gemma3_4b_classifier_disjoint.pkl",
            "selected_h_neurons": 10,
            "selected_ratio_per_mille": 0.1,
            "total_ffn_neurons": 34,
            "evaluation": {
                "n_examples": 10,
                "n_positive": 5,
                "n_negative": 5,
                "metrics": {"accuracy": {"estimate": 0.8}},
                "bootstrap": {},
                "confusion_matrix": {},
            },
        },
    )
    write_json(
        repo_root / "data/gemma3_4b/pipeline/classifier_overlap_summary.json",
        {
            "evaluation": {
                "n_examples": 12,
                "n_positive": 6,
                "n_negative": 6,
                "metrics": {"accuracy": {"estimate": 0.9}},
                "bootstrap": {},
                "confusion_matrix": {},
            }
        },
    )
    write_json(
        repo_root / "data/gemma3_4b/pipeline/test_qids_disjoint.json",
        {"group_a": ["q1", "q2", "q3"]},
    )
    write_json(
        repo_root / "data/gemma3_4b/pipeline/classifier_structure_summary.json",
        {
            "schema_version": 1,
            "generated_at": "2026-03-19",
            "generated_by": "scripts/export_site_data.py",
            "model": "google/gemma-3-4b-it",
            "model_path": "models/gemma3_4b_classifier_disjoint.pkl",
            "generation_script": "scripts/export_site_data.py",
            "source_files": [
                "data/gemma3_4b/pipeline/classifier_disjoint_summary.json",
                "models/gemma3_4b_classifier_disjoint.pkl",
            ],
            "selected_h_neurons": 10,
            "total_ffn_neurons": 34,
            "coefficient_sha256": "0" * 64,
            "structure": {
                "n_layers": 34,
                "neurons_per_layer": 1,
                "positive_counts_by_layer": [1] * 10 + [0] * 24,
                "nonzero_layers": [{"layer": layer, "count": 1} for layer in range(10)],
                "bands": {
                    "early": {
                        "label": "early",
                        "start_layer": 0,
                        "end_layer": 10,
                        "count": 10,
                        "pct": 100.0,
                    },
                    "middle": {
                        "label": "middle",
                        "start_layer": 11,
                        "end_layer": 20,
                        "count": 0,
                        "pct": 0.0,
                    },
                    "late": {
                        "label": "late",
                        "start_layer": 21,
                        "end_layer": 33,
                        "count": 0,
                        "pct": 0.0,
                    },
                },
                "top_positive_neurons": [
                    {
                        "rank": rank,
                        "layer": rank - 1,
                        "neuron": 0,
                        "label": f"L{rank - 1}:N0",
                        "weight": round(10.0 - rank * 0.1, 3),
                    }
                    for rank in range(1, 11)
                ],
            },
        },
    )

    try:
        build_classifier_site_payload(repo_root)
    except ValueError as exc:
        assert "does not match the local disjoint checkpoint" in str(exc)
        assert "coefficient_sha256" in str(exc)
    else:
        raise AssertionError(
            "Expected stale tracked classifier structure to be rejected"
        )


def test_build_classifier_structure_summary_payload_and_validator_use_checkpoint(
    tmp_path: Path,
):
    repo_root = tmp_path
    coef = np.array([9.0 - idx * 0.5 for idx in range(10)] + [-1.0] * 24, dtype=float)
    checkpoint_path = repo_root / "models/gemma3_4b_classifier_disjoint.pkl"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(SimpleNamespace(coef_=np.array([coef], dtype=float)), checkpoint_path)
    write_json(
        repo_root / "data/gemma3_4b/pipeline/classifier_disjoint_summary.json",
        {
            "model_path": "google/gemma-3-4b-it",
            "loaded_model_path": "models/gemma3_4b_classifier_disjoint.pkl",
            "selected_h_neurons": 10,
            "total_ffn_neurons": 34,
        },
    )

    payload = build_classifier_structure_summary_payload(repo_root)

    assert payload["schema_version"] == 1
    assert payload["model_path"] == "models/gemma3_4b_classifier_disjoint.pkl"
    assert payload["generation_script"] == "scripts/export_site_data.py"
    assert payload["coefficient_sha256"] == coefficient_sha256(coef)
    assert payload["structure"]["top_positive_neurons"][0]["label"] == "L0:N0"

    write_json(
        repo_root / "data/gemma3_4b/pipeline/classifier_structure_summary.json",
        {**payload, "generated_at": "2026-03-18"},
    )
    validate_classifier_structure_summary(repo_root)


def test_build_classifier_structure_summary_payload_does_not_use_legacy_overlap_checkpoint(
    tmp_path: Path,
):
    repo_root = tmp_path
    legacy_checkpoint_path = repo_root / "models/gemma3_4b_classifier.pkl"
    legacy_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        SimpleNamespace(coef_=np.array([[1.0] * 34], dtype=float)),
        legacy_checkpoint_path,
    )
    write_json(
        repo_root / "data/gemma3_4b/pipeline/classifier_disjoint_summary.json",
        {
            "model_path": "google/gemma-3-4b-it",
            "selected_h_neurons": 34,
            "total_ffn_neurons": 34,
        },
    )

    try:
        build_classifier_structure_summary_payload(repo_root)
    except FileNotFoundError as exc:
        assert "models/gemma3_4b_classifier_disjoint.pkl" in str(exc)
        assert "models/gemma3_4b_classifier.pkl" not in str(exc)
    else:
        raise AssertionError("Expected missing disjoint checkpoint to fail")


def test_summarize_llm_enrichment_uses_actual_answer_agreement():
    summary = summarize_llm_enrichment(
        [
            {
                "id": "agree_correct",
                "population": "swing",
                "swing_subtype": "R→C",
                "knowledge_class": "COMMON_KNOWLEDGE",
                "llm_answer": "A",
                "model_alpha0_answer": "A",
                "counterfactual_key": "A",
                "answer_agrees_with_model_alpha0": True,
                "llm_answer_correct": True,
                "model_alpha0_answer_correct": True,
                "shared_error": False,
                "persuasiveness": 4,
            },
            {
                "id": "agree_wrong",
                "population": "swing",
                "swing_subtype": "C→R",
                "knowledge_class": "SPECIALIZED",
                "llm_answer": "B",
                "model_alpha0_answer": "B",
                "counterfactual_key": "C",
                "answer_agrees_with_model_alpha0": True,
                "llm_answer_correct": False,
                "model_alpha0_answer_correct": False,
                "shared_error": True,
                "persuasiveness": 3,
            },
            {
                "id": "disagree",
                "population": "always_compliant",
                "swing_subtype": "",
                "knowledge_class": "COMMON_KNOWLEDGE",
                "llm_answer": "D",
                "model_alpha0_answer": "A",
                "counterfactual_key": "D",
                "answer_agrees_with_model_alpha0": False,
                "llm_answer_correct": True,
                "model_alpha0_answer_correct": False,
                "shared_error": False,
                "persuasiveness": 2,
            },
            {
                "id": "unknown",
                "population": "never_compliant",
                "swing_subtype": "",
                "knowledge_class": "AMBIGUOUS",
                "llm_answer": "UNKNOWN",
                "model_alpha0_answer": "A",
                "counterfactual_key": "A",
                "answer_agrees_with_model_alpha0": None,
                "llm_answer_correct": None,
                "model_alpha0_answer_correct": True,
                "shared_error": None,
                "persuasiveness": 1,
            },
        ]
    )

    assert summary["verification_agreement"]["count"] == 2
    assert summary["verification_agreement"]["n_total"] == 3
    assert summary["verification_agreement"]["estimate"] == 2 / 3


def test_compact_llm_enrichment_refuses_shared_error_backfill():
    payload = compact_llm_enrichment(
        {
            "samples": [
                {"both_correct": True},
                {"both_correct": False},
            ]
        }
    )

    assert "verification_agreement" not in payload


def test_compact_llm_enrichment_uses_explicit_agreement_fields():
    payload = compact_llm_enrichment(
        {
            "samples": [
                {"answer_agrees_with_model_alpha0": True},
                {"answer_agrees_with_model_alpha0": False},
                {"answer_agrees_with_model_alpha0": True},
                {"answer_agrees_with_model_alpha0": None},
            ]
        }
    )

    assert payload["verification_agreement"]["count"] == 2
    assert payload["verification_agreement"]["n_total"] == 3
    assert payload["verification_agreement"]["pct"] == 66.7
