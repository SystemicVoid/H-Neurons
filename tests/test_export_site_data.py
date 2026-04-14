from pathlib import Path

import pytest

from scripts.export_site_data import (
    build_bridge_phase3_payload,
    build_d7_comparison_payload,
    build_jailbreak_payload,
    build_payload,
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
    assert payload["paired_vs_baseline"]["causal"]["csv2_yes"][
        "estimate_pp"
    ] == pytest.approx(-9.0)
    assert payload["paired_vs_baseline"]["l1"]["csv2_yes"][
        "estimate_pp"
    ] == pytest.approx(4.0)
    assert payload["token_cap"]["causal_hits"] == 112
    assert payload["token_cap"]["causal_share_pct"] == 22.4
