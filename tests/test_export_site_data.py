import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import scripts.build_d7_control_and_ruler_summary as d7_current_state_summary
from scripts.export_site_data import (
    build_bridge_phase3_payload,
    build_d7_comparison_payload,
    build_jailbreak_payload,
    build_payload,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_build_intervention_payload_exports_slope_difference_summaries():
    repo_root = Path(__file__).resolve().parents[1]
    payload = build_payload(repo_root)

    negative_control = payload["negative_control"]
    matched_readout = payload["matched_readout_comparison"]

    assert payload["schema_version"] == 1
    assert (
        "data/gemma3_4b/intervention/faitheval/control/slope_difference_summary.json"
        in payload["provenance"]["source_files"]
    )
    assert (
        "data/gemma3_4b/intervention/faitheval_sae/control/slope_difference_summary.json"
        in payload["provenance"]["source_files"]
    )
    assert negative_control["paired_slope_difference"]["aggregate"]["n_seeds"] == 8
    assert (
        negative_control["paired_slope_difference"]["source_file"]
        == "data/gemma3_4b/intervention/faitheval/control/slope_difference_summary.json"
    )
    assert matched_readout["comparison"] == "neuron_minus_sae"
    assert matched_readout["n_items"] == 1000
    assert (
        matched_readout["source_file"]
        == "data/gemma3_4b/intervention/faitheval_sae/control/slope_difference_summary.json"
    )


def test_build_jailbreak_payload_exports_measurement_anchor_summary():
    repo_root = Path(__file__).resolve().parents[1]
    payload = build_jailbreak_payload(repo_root)

    paired = payload["measurement"]["paired_evaluator_comparison"]
    holdout = payload["measurement"]["strongreject_holdout"]

    assert payload["negative_control"]["status"] == "multi_seed_missing"
    assert paired["binary_v2_slope"]["estimate"] == 2.30
    assert paired["binary_v3_slope"]["ci"]["lower"] == -1.46
    assert paired["substantive_compliance_v3_slope"]["estimate"] == 2.00
    assert holdout["status"] == "tie_with_v3_on_holdout"
    assert holdout["v3_accuracy_pct"] == 96.0
    assert (
        "notes/act3-reports/2026-04-13-v2-v3-paired-evaluator-comparison.md"
        in payload["provenance"]["source_files"]
    )


def test_build_bridge_phase3_payload_exports_externality_break():
    repo_root = Path(__file__).resolve().parents[1]
    payload = build_bridge_phase3_payload(repo_root)

    assert payload["benchmark"] == "triviaqa_bridge_phase3"
    assert payload["effects"]["adjudicated_accuracy_delta_pp"]["estimate"] == -5.8
    assert payload["effects"]["adjudicated_accuracy_delta_pp"]["ci"]["upper"] == -3.0
    assert payload["effects"]["mcnemar_p"] == 0.0002
    assert payload["failure_modes"]["wrong_entity_substitution"]["count"] == 30
    assert payload["failure_modes"]["wrong_entity_substitution"]["share_pct"] == 70.0


def test_build_d7_comparison_payload_exports_benchmark_local_caveat():
    repo_root = Path(__file__).resolve().parents[1]
    payload = build_d7_comparison_payload(repo_root)

    assert payload["claim_status"] == "benchmark_local_supporting_evidence"
    assert (
        payload["caveat"]
        == "D7 is a benchmark-local supporting result on a clean CSV2 v3 current-state "
        "panel: causal is still the strongest completed branch, and the remaining "
        "live caveats are causal token-cap and quality debt plus a small documented "
        "residual CSV2-error set rather than mixed-ruler debt."
    )
    assert (
        "data/gemma3_4b/intervention/jailbreak_d7/full500_canonical/d7_full500_current_state_summary.json"
        in payload["source_files"]
    )
    assert payload["headline"] == (
        "D7 current state: the causal branch is still the strongest completed "
        "full-500 condition, outperforming the available probe and layer-matched "
        "random branches on the current normalized panel."
    )
    assert payload["conditions"]["baseline"]["strict_harmfulness_normalized"][
        "estimate_pct"
    ] == pytest.approx(34.2)
    assert payload["conditions"]["l1"]["strict_harmfulness_normalized"][
        "estimate_pct"
    ] == pytest.approx(36.4)
    assert payload["conditions"]["causal"]["strict_harmfulness_normalized"][
        "estimate_pct"
    ] == pytest.approx(20.0)
    assert payload["paired_vs_baseline"]["causal"]["strict_harmfulness_normalized"][
        "estimate_pp"
    ] == pytest.approx(-14.2)
    assert payload["paired_vs_baseline"]["causal"]["strict_harmfulness_normalized"][
        "estimate"
    ] == pytest.approx(-14.2)
    assert payload["paired_vs_baseline"]["l1"]["strict_harmfulness_normalized"][
        "estimate_pp"
    ] == pytest.approx(2.2)
    assert payload["token_cap"]["causal_hits"] == 112
    assert payload["token_cap"]["causal_share_pct"] == pytest.approx(22.4)
    assert payload["direct_comparisons"]["probe_vs_causal"][
        "strict_harmfulness_normalized"
    ]["estimate_pp"] == pytest.approx(14.8)
    assert (
        payload["conditions"]["baseline"]["csv2_yes"]
        == payload["conditions"]["baseline"]["strict_harmfulness_normalized"]
    )
    assert (
        payload["conditions"]["l1"]["csv2_yes"]
        == payload["conditions"]["l1"]["strict_harmfulness_normalized"]
    )
    assert (
        payload["conditions"]["causal"]["csv2_yes"]
        == payload["conditions"]["causal"]["strict_harmfulness_normalized"]
    )
    assert (
        payload["paired_vs_baseline"]["causal"]["csv2_yes"]
        == payload["paired_vs_baseline"]["causal"]["strict_harmfulness_normalized"]
    )
    assert (
        payload["paired_vs_baseline"]["l1"]["csv2_yes"]
        == payload["paired_vs_baseline"]["l1"]["strict_harmfulness_normalized"]
    )


def test_build_d7_comparison_payload_exports_current_state_namespace():
    repo_root = Path(__file__).resolve().parents[1]
    payload = build_d7_comparison_payload(repo_root)
    current_state = payload["current_state"]

    assert current_state["claim_status"] == "benchmark_local_supporting_clean_panel"
    assert (
        current_state["headline"]
        == "D7 current state: the causal branch is still the strongest completed "
        "full-500 condition, outperforming the available probe and layer-matched "
        "random branches on the current normalized panel."
    )
    assert (
        current_state["caveat"]
        == "Interpret D7 as benchmark-local supporting evidence on a clean CSV2 v3 "
        "panel: the mixed-ruler debt on the current panel has been cleared, and the "
        "remaining evaluator-error debt is now a small documented residual set after "
        "repair. The main live caveat is causal token-cap and quality debt rather "
        "than ruler debt."
    )
    assert (
        current_state["mixed_ruler_status"]["status"]
        == "resolved_clean_v3_panel_small_residual_errors"
    )
    assert current_state["control"]["availability"] == "available_two_seed_panel"
    assert (
        current_state["control"]["seed_1"]["status"] == "complete_scored_error_bearing"
    )
    assert current_state["control"]["status"] == "two_seed_clean_panel"
    assert (
        current_state["control"]["seed_2"]["status"] == "complete_scored_error_bearing"
    )
    assert current_state["probe"]["status"] == "complete_scored_error_bearing"
    assert current_state["probe"]["experiment_row_count"] == 500
    assert current_state["probe"]["included_in_current_claim"] is True
    assert (
        "notes/act3-reports/2026-04-16-d7-full500-two-seed-current-state-audit.md"
        in current_state["source_files"]
    )
    assert current_state["current_panel"]["direct_random_layer_seed1_vs_causal"][
        "strict_harmfulness_normalized"
    ]["estimate_pp"] == pytest.approx(17.2)

    current_panel = current_state["current_panel"]
    assert current_panel["conditions"]["baseline"]["strict_harmfulness_normalized"][
        "estimate_pct"
    ] == pytest.approx(34.2)
    assert current_panel["conditions"]["l1"]["strict_harmfulness_normalized"][
        "estimate_pct"
    ] == pytest.approx(36.4)
    assert current_panel["conditions"]["causal"]["strict_harmfulness_normalized"][
        "estimate_pct"
    ] == pytest.approx(20.0)
    assert current_panel["conditions"]["random_layer_seed1"][
        "strict_harmfulness_normalized"
    ]["estimate_pct"] == pytest.approx(37.2)
    assert current_panel["conditions"]["random_layer_seed2"][
        "strict_harmfulness_normalized"
    ]["estimate_pct"] == pytest.approx(38.4)
    assert current_panel["conditions"]["probe"]["strict_harmfulness_normalized"][
        "estimate_pct"
    ] == pytest.approx(34.8)
    assert current_panel["deltas_vs_baseline"]["l1"]["strict_harmfulness_normalized"][
        "estimate_pp"
    ] == pytest.approx(2.2)
    assert current_panel["deltas_vs_baseline"]["causal"][
        "strict_harmfulness_normalized"
    ]["estimate_pp"] == pytest.approx(-14.2)
    assert current_panel["deltas_vs_baseline"]["random_layer_seed1"][
        "strict_harmfulness_normalized"
    ]["estimate_pp"] == pytest.approx(3.0)
    assert current_panel["deltas_vs_baseline"]["random_layer_seed1"]["binary_harmful"][
        "estimate_pp"
    ] == pytest.approx(6.6)
    assert current_panel["deltas_vs_baseline"]["random_layer_seed2"][
        "strict_harmfulness_normalized"
    ]["estimate_pp"] == pytest.approx(4.2)
    assert current_panel["deltas_vs_baseline"]["random_layer_seed2"]["binary_harmful"][
        "estimate_pp"
    ] == pytest.approx(5.2)
    assert current_panel["deltas_vs_baseline"]["probe"][
        "strict_harmfulness_normalized"
    ]["estimate_pp"] == pytest.approx(0.6)
    assert current_panel["direct_causal_vs_random_layer_seed1"][
        "strict_harmfulness_normalized"
    ]["estimate_pp"] == pytest.approx(-17.2)
    assert current_panel["direct_causal_vs_random_layer_seed1"]["binary_harmful"][
        "estimate_pp"
    ] == pytest.approx(-17.2)
    assert current_panel["direct_causal_vs_probe"]["strict_harmfulness_normalized"][
        "estimate_pp"
    ] == pytest.approx(-14.8)
    assert current_panel["direct_causal_vs_random_layer_seed2"][
        "strict_harmfulness_normalized"
    ]["estimate_pp"] == pytest.approx(-18.4)
    assert current_panel["direct_causal_vs_random_layer_seed2"]["binary_harmful"][
        "estimate_pp"
    ] == pytest.approx(-15.8)
    assert current_panel["direct_probe_vs_random_layer_seed1"][
        "strict_harmfulness_normalized"
    ]["estimate_pp"] == pytest.approx(-2.4)
    assert current_panel["direct_probe_vs_random_layer_seed2"][
        "strict_harmfulness_normalized"
    ]["estimate_pp"] == pytest.approx(-3.6)
    assert current_panel["direct_random_layer_seed2_vs_random_layer_seed1"][
        "strict_harmfulness_normalized"
    ]["estimate_pp"] == pytest.approx(1.2)
    assert current_panel["direct_random_layer_seed2_vs_random_layer_seed1"][
        "strict_harmfulness_normalized"
    ]["ci_pp"]["lower"] == pytest.approx(-2.8)
    assert current_panel["direct_random_layer_seed2_vs_random_layer_seed1"][
        "strict_harmfulness_normalized"
    ]["ci_pp"]["upper"] == pytest.approx(5.0)
    assert current_panel["direct_random_layer_seed1_vs_random_layer_seed2"][
        "strict_harmfulness_normalized"
    ]["estimate_pp"] == pytest.approx(-1.2)

    error_burden = current_state["random_seed1_csv2_error_burden"]
    assert error_burden["count"] == 4
    assert error_burden["types"] == {"invalid_evidence_spans": 3, "parse_failed": 1}
    assert error_burden["clean_row_count"] == 496
    assert error_burden["total_row_count"] == 500

    probe_error_burden = current_state["probe_csv2_error_burden"]
    assert probe_error_burden["count"] == 2
    assert probe_error_burden["types"] == {"invalid_evidence_spans": 2}
    assert probe_error_burden["clean_row_count"] == 498
    assert probe_error_burden["total_row_count"] == 500


def test_available_current_conditions_excludes_incomplete_optional_csv2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    spec = d7_current_state_summary.ConditionSpec(
        key="random_layer_seed2",
        label="causal_random_head_layer_matched/seed_2",
        alpha=4.0,
        experiment_path=tmp_path / "experiment" / "alpha_4.0.jsonl",
        csv2_path=tmp_path / "csv2_evaluation" / "alpha_4.0.jsonl",
    )
    experiment_rows = [
        {"id": "sample-1", "response": "alpha"},
        {"id": "sample-2", "response": "beta"},
    ]
    incomplete_csv2_rows = [
        {"id": "sample-1", "response": "alpha"},
        {"id": "sample-2", "response": "beta", "csv2": {"harmful_binary": "no"}},
    ]
    complete_csv2_rows = [
        {"id": "sample-1", "response": "alpha", "csv2": {"harmful_binary": "no"}},
        {"id": "sample-2", "response": "beta", "csv2": {"harmful_binary": "no"}},
    ]

    _write_jsonl(spec.experiment_path, experiment_rows)
    _write_jsonl(spec.csv2_path, incomplete_csv2_rows)

    monkeypatch.setattr(d7_current_state_summary, "BASE_CURRENT_CONDITIONS", [])
    monkeypatch.setattr(
        d7_current_state_summary,
        "OPTIONAL_CURRENT_CONDITIONS",
        [spec],
    )

    assert d7_current_state_summary._available_current_conditions() == []

    _write_jsonl(spec.csv2_path, complete_csv2_rows)

    assert d7_current_state_summary._available_current_conditions() == [spec]


def test_current_state_summary_counts_parse_failed_v3_rows_separately():
    summary = d7_current_state_summary.build_summary()
    causal_counts = summary["current_panel"]["conditions"]["causal"][
        "csv2_schema_versions"
    ]
    random_seed1_counts = summary["current_panel"]["conditions"]["random_layer_seed1"][
        "csv2_schema_versions"
    ]

    assert "legacy_unversioned" not in causal_counts
    assert causal_counts["csv2_v3_parse_failed"] == 4
    assert causal_counts["csv2_v3"] == 496
    assert "legacy_unversioned" not in random_seed1_counts
    assert random_seed1_counts["csv2_v3_parse_failed"] == 1
    assert random_seed1_counts["csv2_v3"] == 499
