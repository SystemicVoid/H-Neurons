from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from export_simid_prospective_effect_run_gate import (  # noqa: E402
    DEFAULT_OPEN_GATE_DIR,
    DEFAULT_PASSING_OPEN_ANALYSIS,
    PACKAGE_SCHEMA_VERSION,
    build_manifest,
    export_package,
    file_sha256,
    parse_args,
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def export_tmp_gate(tmp_path: Path) -> Path:
    output_dir = tmp_path / "prospective_effect_gate"
    export_package(
        parse_args(
            [
                "--output-dir",
                str(output_dir),
                "--planned-manifest",
                str(tmp_path / "simid_prospective_manifest.json"),
                "--planned-noop-run-dir",
                str(tmp_path / "noop_preflight"),
                "--planned-effect-run-dir",
                str(tmp_path / "effect_run"),
            ]
        )
    )
    return output_dir


def test_effect_run_gate_binds_to_passing_open_calibration_hashes(
    tmp_path: Path,
) -> None:
    output_dir = export_tmp_gate(tmp_path)
    manifest = read_json(output_dir / "effect_run_manifest.json")
    authority = manifest["frozen_open_grading_authority"]

    assert manifest["schema_version"] == PACKAGE_SCHEMA_VERSION
    assert authority["authority_scope"] == (
        "future_simid_open_grading_under_frozen_rubric_only"
    )
    assert authority["passing_analysis"]["path"].endswith(
        "prospective_open_calibration_analysis_opus_4_7_max_run_001.json"
    )
    assert authority["passing_analysis"]["content_sha256"] == file_sha256(
        DEFAULT_PASSING_OPEN_ANALYSIS
    )
    assert authority["review_manifest"]["content_sha256"] == file_sha256(
        DEFAULT_OPEN_GATE_DIR / "review_manifest.json"
    )
    assert authority["rubric"]["content_sha256"] == file_sha256(
        DEFAULT_OPEN_GATE_DIR / "rubric.md"
    )
    assert authority["passing_label_file"]["content_sha256"] == file_sha256(
        DEFAULT_OPEN_GATE_DIR / "prospective_open_labels_opus_4_7_max_run_001.jsonl"
    )
    assert authority["metrics"]["raw_agreement"]["count"] == 138
    assert authority["metrics"]["cohen_kappa"] == 0.8734
    assert authority["policy"]["target"] == "simid_open_correctness_future_grading"
    assert "historical MVP" in authority["claimability_guardrail"]
    assert manifest["exclusions"]["content_sha256"] == file_sha256(
        output_dir / "excluded_effect_sample_ids.jsonl"
    )
    exclusions = (output_dir / "excluded_effect_sample_ids.jsonl").read_text(
        encoding="utf-8"
    )
    assert "historical_mvp_locked_manifest" in exclusions
    assert "also_in_prospective_open_calibration_sample" in exclusions


def test_effect_run_gate_prespecifies_controls_alpha_grid_and_outputs(
    tmp_path: Path,
) -> None:
    output_dir = export_tmp_gate(tmp_path)
    manifest = read_json(output_dir / "effect_run_manifest.json")
    planned = manifest["planned_run"]
    controls = {control["name"] for control in planned["controls"]}

    assert controls == {
        "unhooked_noop_preflight",
        *(f"random_direction_seed{seed}" for seed in range(1, 6)),
        *(f"random_head_seed{seed}" for seed in range(1, 6)),
    }
    assert planned["alpha_grid"]["values"] == [-8.0, 0.0, 4.0, 8.0]
    assert planned["primary_estimand"] == {
        "name": "truthfulqa_paired_open_correctness_delta_alpha8_vs_alpha0",
        "dataset": "truthfulqa",
        "condition": "selected",
        "metric": "adjudicated_open_correct_under_frozen_prospective_rubric",
        "comparison": {"treatment_alpha": 8.0, "baseline_alpha": 0.0},
        "unit": "base_sample_id_with_option_order_replicates_averaged",
        "uncertainty": "paired bootstrap 95% CI at base-item level",
        "minimum_practical_effect_pp": 5.0,
    }
    command = planned["commands"]["effect_run"]["rendered"]
    assert (
        "--conditions selected random_head_seed1 random_head_seed2 "
        "random_head_seed3 random_head_seed4 random_head_seed5 "
        "random_direction_seed1 random_direction_seed2 random_direction_seed3 "
        "random_direction_seed4 random_direction_seed5"
    ) in command
    assert "--alphas -8.0 0.0 4.0 8.0" in command
    assert "--allow-overwrite" not in command
    manifest_command = planned["commands"]["build_manifest"]["rendered"]
    assert "--truthfulqa-leakage-policy heldout_only" in manifest_command
    assert "--min-truthfulqa-rows 450" in manifest_command
    assert "--min-bridge-rows 400" in manifest_command
    assert "--exclude-sample-ids-jsonl" in manifest_command
    analysis_command = planned["commands"]["effect_adjudicated_analysis"]["rendered"]
    assert "--prospective-open-authority-manifest" in analysis_command

    paths = manifest["output_paths"]
    assert paths["package_dir"].endswith("prospective_effect_gate")
    assert paths["planned_manifest"].endswith("simid_prospective_manifest.json")
    assert paths["noop_preflight_run_dir"].endswith("noop_preflight")
    assert paths["effect_run_dir"].endswith("effect_run")


def test_effect_run_gate_preserves_retrospective_diagnostic_only_framing(
    tmp_path: Path,
) -> None:
    output_dir = export_tmp_gate(tmp_path)
    manifest = read_json(output_dir / "effect_run_manifest.json")
    protocol = (output_dir / "protocol.md").read_text(encoding="utf-8")

    assert manifest["source_mvp_run"]["status"] == (
        "diagnostic_only_not_retrospectively_upgraded"
    )
    assert (
        "historical MVP open-correctness improvement"
        in manifest["claimability_gates"]["non_claimable_even_if_passed"]
    )
    assert (
        "prospective_open_authority_not_recorded_in_results"
        in manifest["claimability_gates"]["analysis_blockers"]
    )
    assert (
        "prospective_effect_manifest_reuses_historical_mvp_or_calibration_rows"
        in manifest["claimability_gates"]["analysis_blockers"]
    )
    assert "Historical MVP open metrics" in protocol
    assert "remain diagnostic-only" in protocol
    assert "does not imply historical MVP claimability" in protocol


def test_effect_run_gate_rejects_failed_or_drifted_open_authority(
    tmp_path: Path,
) -> None:
    bad_analysis = tmp_path / "bad_analysis.json"
    payload = read_json(DEFAULT_PASSING_OPEN_ANALYSIS)
    payload["outcome"] = {
        "passed": False,
        "blockers": ["raw_agreement_below_threshold"],
    }
    bad_analysis.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    args = parse_args(
        [
            "--output-dir",
            str(tmp_path / "gate_failed"),
            "--passing-open-analysis",
            str(bad_analysis),
        ]
    )
    with pytest.raises(ValueError, match="did not pass"):
        build_manifest(args)

    mismatched_package = tmp_path / "mismatched_package.json"
    payload = read_json(DEFAULT_PASSING_OPEN_ANALYSIS)
    payload["package_dir"] = "data/not_the_open_gate"
    mismatched_package.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path / "gate_mismatched_package"),
            "--passing-open-analysis",
            str(mismatched_package),
        ]
    )
    with pytest.raises(ValueError, match="does not match open gate"):
        build_manifest(args)

    drifted_analysis = tmp_path / "drifted_analysis.json"
    payload = read_json(DEFAULT_PASSING_OPEN_ANALYSIS)
    payload["label_file_sha256"] = "0" * 64
    drifted_analysis.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path / "gate_drifted"),
            "--passing-open-analysis",
            str(drifted_analysis),
        ]
    )
    with pytest.raises(ValueError, match="label_file_sha256 mismatch"):
        build_manifest(args)
