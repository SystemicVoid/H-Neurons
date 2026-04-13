from pathlib import Path

from scripts.export_site_data import build_payload


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
