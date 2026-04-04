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
    TOP_NEURON_ARTIFACT_TEST_SLUGS,
    build_classifier_site_payload,
    build_classifier_structure_summary_payload,
    build_jailbreak_payload,
    build_swing_characterization_payload,
    coefficient_sha256,
    compact_llm_enrichment,
    validate_classifier_structure_summary,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def make_top_neuron_artifact_summary(
    *,
    label: str,
    weight: float,
    selected_h_neurons: int,
    broader_positive_neurons: int | None = None,
    ci_status: str = "no_ci_fixed_diagnostic",
    test_slugs: tuple[str, ...] = TOP_NEURON_ARTIFACT_TEST_SLUGS,
) -> dict:
    broader_positive_neurons = broader_positive_neurons or (selected_h_neurons + 2)
    tests = [
        {
            "slug": slug,
            "label": slug.replace("_", " ").title(),
            "display_value": f"{index / 10:.1f}",
            "threshold": f">{(index + 1) / 10:.1f}",
            "verdict": "artifact",
        }
        for index, slug in enumerate(test_slugs, start=1)
    ]
    return {
        "schema_version": 1,
        "generated_at": "2026-03-20",
        "generated_by": "tracked_manual_summary",
        "source_files": ["data/gemma3_4b/pipeline/pipeline_report.md"],
        "target_neuron": {
            "layer": 0,
            "neuron": 0,
            "label": label,
            "weight": weight,
            "weight_rank": 1,
        },
        "verdict": {
            "status": "artifact",
            "supporting_tests": 0,
            "total_tests": len(tests),
            "summary": "Synthetic summary",
            "ci_status": ci_status,
        },
        "distributed_detector_context": {
            "sparse_baseline": {
                "c_value": 1.0,
                "positive_neurons": selected_h_neurons,
                "accuracy_pct": 80.0,
                "target_rank": 1,
            },
            "broader_detector": {
                "c_value": 3.0,
                "positive_neurons": broader_positive_neurons,
                "accuracy_pct": 82.0,
                "target_rank": 2,
            },
            "loosest_detector": {
                "c_value": 10.0,
                "positive_neurons": broader_positive_neurons + 3,
                "accuracy_pct": 81.0,
                "target_rank": 4,
            },
        },
        "tests": tests,
    }


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
    artifact = payload["top_neuron_artifact_summary"]
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
    assert artifact["target_neuron"]["label"] == top_positive[0]["label"]
    assert artifact["verdict"]["supporting_tests"] == 0
    assert artifact["verdict"]["total_tests"] == 6
    assert [test["slug"] for test in artifact["tests"]] == list(
        TOP_NEURON_ARTIFACT_TEST_SLUGS
    )
    assert (
        artifact["distributed_detector_context"]["broader_detector"]["positive_neurons"]
        == 219
    )
    assert artifact["tests"][0]["display_value"] == "0.590"
    assert [entry["weight"] for entry in top_positive] == sorted(
        (entry["weight"] for entry in top_positive),
        reverse=True,
    )


def test_build_jailbreak_payload_exports_current_provenance_and_controls():
    import pytest

    repo_root = Path(__file__).resolve().parents[1]
    jailbreak_dir = repo_root / "data/gemma3_4b/intervention/jailbreak/experiment"
    if not any(jailbreak_dir.glob("results*.json")):
        pytest.skip("jailbreak results not yet available (run in progress)")
    payload = build_jailbreak_payload(repo_root)

    assert payload["stochastic_generation"]["sampling"] == {
        "do_sample": True,
        "temperature": 0.7,
    }
    assert "temperature=0.7, do_sample=true" in payload["provenance"]["notes"][0]

    cross_benchmark = {
        entry["name"]: entry for entry in payload["cross_benchmark"]["benchmarks"]
    }
    assert cross_benchmark["FalseQA"]["negative_control"] == "available"
    assert cross_benchmark["JailbreakBench"]["generation"] == "stochastic (T=0.7)"


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


def test_results_page_top_neuron_verdict_uses_live_bindings():
    repo_root = Path(__file__).resolve().parents[1]
    results_html = (repo_root / "site/results/gemma-3-4b.html").read_text()
    shared_js = (repo_root / "site/assets/shared.js").read_text()

    bindings = [
        "support-count-display",
        "diagnostic-count",
        "ci-status",
        "verdict-summary",
        "auc-display",
        "cohen-d-display",
        "c-sweep-display",
        "top-contrib-display",
        "ablation-display",
        "max-r-display",
        "practical-takeaway",
        "takeaway-card-text",
    ]

    for binding in bindings:
        assert f'data-top-neuron-bind="{binding}"' in results_html
        assert f"'{binding}'" in shared_js

    assert "formatTopNeuronArtifactCiStatus(verdict.ci_status)" in shared_js
    assert (
        "setBoundText('data-top-neuron-bind', 'ci-status', 'No CI: fixed held-out diagnostic checks');"
        not in shared_js
    )
    assert (
        "neuron 4288 behaves like a regularization artifact rather than a uniquely causal hub."
        not in results_html
    )
    assert (
        "L1 weight ranking overstates the importance of individual top neurons and understates distributed signal."
        not in results_html
    )


def test_results_page_jailbreak_copy_uses_live_bindings():
    repo_root = Path(__file__).resolve().parents[1]
    results_html = (repo_root / "site/results/gemma-3-4b.html").read_text()
    charts_js = (repo_root / "site/assets/charts.js").read_text()
    shared_js = (repo_root / "site/assets/shared.js").read_text()

    jailbreak_bindings = [
        "negative-control-value",
        "negative-control-detail",
        "negative-control-comparison",
        "stochastic-generation-detail",
    ]

    for binding in jailbreak_bindings:
        assert f'data-jailbreak-bind="{binding}"' in results_html
        assert f"'{binding}'" in charts_js

    cross_benchmark_bindings = [
        "faitheval-negative-control",
        "faitheval-evaluator",
        "faitheval-generation",
        "falseqa-negative-control",
        "falseqa-evaluator",
        "falseqa-generation",
        "jailbreakbench-negative-control",
        "jailbreakbench-evaluator",
        "jailbreakbench-generation",
        "interpretation-caveat",
    ]

    for binding in cross_benchmark_bindings:
        assert f'data-cross-benchmark-bind="{binding}"' in results_html

    assert "hydrateCrossBenchmarkBindings" in shared_js
    assert "`${key}-negative-control`" in shared_js
    assert "`${key}-evaluator`" in shared_js
    assert "`${key}-generation`" in shared_js

    stale_literals = [
        "FaithEval and FalseQA both have negative controls, but those results do not automatically transfer to JailbreakBench.",
        "Responses were generated with temperature=0.7. Per-item outcomes are not exactly reproducible; aggregate rates carry sampling noise above the usual alpha-sweep uncertainty.",
        "However, JailbreakBench is still the only benchmark here without its own negative control, and it uses stochastic generation",
        "Negative control: available</div>",
        'data-jailbreak-bind="cross-falseqa-n">n=687</span> &middot; GPT-4o judge',
        'data-jailbreak-bind="cross-jailbreakbench-n">n=500</span> &middot; Stochastic gen',
        "temperature=0.6",
        "500 JailbreakBench prompts &times; 7 alpha values &times; 5 templates",
    ]

    for literal in stale_literals:
        assert literal not in results_html

    assert (
        "100 behaviors &times; 5 templates &times; 7 alpha values = 3,500 responses"
        in results_html
    )


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
    write_json(
        repo_root / "data/gemma3_4b/pipeline/neuron_4288_summary.json",
        make_top_neuron_artifact_summary(
            label="L0:N0",
            weight=float(
                tracked_summary["structure"]["top_positive_neurons"][0]["weight"]
            ),
            selected_h_neurons=10,
        ),
    )

    payload = build_classifier_site_payload(repo_root)

    assert payload["selected_h_neuron_structure"] == tracked_summary["structure"]
    assert payload["source_files"][-2] == (
        "data/gemma3_4b/pipeline/classifier_structure_summary.json"
    )
    assert payload["source_files"][-1] == (
        "data/gemma3_4b/pipeline/neuron_4288_summary.json"
    )
    assert payload["top_neuron_artifact_summary"]["target_neuron"]["label"] == "L0:N0"


def test_build_classifier_site_payload_rejects_top_neuron_artifact_slug_drift(
    tmp_path: Path,
):
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
    write_json(
        repo_root / "data/gemma3_4b/pipeline/neuron_4288_summary.json",
        make_top_neuron_artifact_summary(
            label="L0:N0",
            weight=float(
                tracked_summary["structure"]["top_positive_neurons"][0]["weight"]
            ),
            selected_h_neurons=10,
            test_slugs=(
                "single_neuron_auc",
                "distribution_separation",
                "c_sweep_stability",
                "largest_contribution_share",
                "ablation_accuracy_drop",
                "renamed_top10_correlation",
            ),
        ),
    )

    try:
        build_classifier_site_payload(repo_root)
    except ValueError as exc:
        assert "exact renderer slug set" in str(exc)
        assert "max_top10_correlation" in str(exc)
        assert "renamed_top10_correlation" in str(exc)
    else:
        raise AssertionError("Expected top-neuron artifact slug drift to be rejected")


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
    write_json(
        repo_root / "data/gemma3_4b/pipeline/neuron_4288_summary.json",
        make_top_neuron_artifact_summary(
            label="L0:N0",
            weight=9.0,
            selected_h_neurons=10,
        ),
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
    write_json(
        repo_root / "data/gemma3_4b/pipeline/neuron_4288_summary.json",
        make_top_neuron_artifact_summary(
            label="L0:N0",
            weight=9.0,
            selected_h_neurons=10,
        ),
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
