from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_contract import (  # noqa: E402
    CONTROL_RUN_CONFIG_FILENAME,
    INTERVENTION_RUN_CONFIG_FILENAME,
    INTERVENTION_RUN_CONFIG_SCHEMA_VERSION,
    NEGATIVE_CONTROL_RUN_CONFIG_SCHEMA_VERSION,
    assert_h_neuron_baseline_matches_control_contract,
    assert_or_write_negative_control_run_contract,
    build_run_contract,
    contract_mismatch_paths,
    file_sha256,
    format_mismatch_preview,
    load_h_neuron_baseline_binding,
    optional_file_sha256,
    write_or_validate_run_contract,
)


def _shared_contract_fields() -> dict[str, Any]:
    return {
        "benchmark": "faitheval",
        "model": {
            "key": "mistral24b",
            "path": "mistralai/Mistral-Small-24B-Instruct-2501",
        },
        "classifier": {
            "path": "classifier.pkl",
            "content_sha256": "classifier-sha",
        },
        "sample_selection": {
            "max_samples": None,
            "sample_manifest": {
                "path": "manifest.json",
                "content_sha256": "manifest-sha",
            },
        },
        "schedule": {"alphas": [0.0, 1.0, 3.0]},
        "benchmark_config": {"prompt_style": "standard"},
    }


def _intervention_contract() -> dict[str, Any]:
    return {
        "schema_version": INTERVENTION_RUN_CONFIG_SCHEMA_VERSION,
        **_shared_contract_fields(),
    }


def _negative_control_contract(
    baseline_binding: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": NEGATIVE_CONTROL_RUN_CONFIG_SCHEMA_VERSION,
        "quick": True,
        **_shared_contract_fields(),
        "schedule": {
            "alphas": [0.0, 1.0, 3.0],
            "configs": [
                {
                    "name": "seed_0_unconstrained",
                    "strategy": "unconstrained",
                    "seed": 0,
                }
            ],
        },
        "h_neuron_baseline": baseline_binding,
    }


def test_build_run_contract_preserves_shared_shape_and_classifier_hash(
    tmp_path: Path,
) -> None:
    classifier = tmp_path / "classifier.pkl"
    classifier.write_bytes(b"classifier-v1")
    sample_manifest = {"path": "manifest.json", "content_sha256": "manifest-sha"}

    contract = build_run_contract(
        schema_version=NEGATIVE_CONTROL_RUN_CONFIG_SCHEMA_VERSION,
        benchmark="faitheval",
        after_benchmark_fields={"quick": True},
        model_key="mistral24b",
        model_path="model-path",
        classifier_path=str(classifier),
        max_samples=100,
        sample_manifest=sample_manifest,
        alphas=[0, 1, 3],
        schedule_fields={
            "configs": [
                {
                    "name": "seed_0_unconstrained",
                    "strategy": "unconstrained",
                    "seed": 0,
                }
            ]
        },
        benchmark_config={"prompt_style": "standard"},
        tail_fields={"h_neuron_baseline": None},
    )

    assert contract == {
        "schema_version": NEGATIVE_CONTROL_RUN_CONFIG_SCHEMA_VERSION,
        "benchmark": "faitheval",
        "quick": True,
        "model": {"key": "mistral24b", "path": "model-path"},
        "classifier": {
            "path": str(classifier),
            "content_sha256": hashlib.sha256(b"classifier-v1").hexdigest(),
        },
        "sample_selection": {
            "max_samples": 100,
            "sample_manifest": sample_manifest,
        },
        "schedule": {
            "alphas": [0.0, 1.0, 3.0],
            "configs": [
                {
                    "name": "seed_0_unconstrained",
                    "strategy": "unconstrained",
                    "seed": 0,
                }
            ],
        },
        "benchmark_config": {"prompt_style": "standard"},
        "h_neuron_baseline": None,
    }


def test_file_hash_and_optional_missing_path(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"claim-bearing bytes")

    assert file_sha256(payload) == hashlib.sha256(b"claim-bearing bytes").hexdigest()
    assert optional_file_sha256(str(payload)) == file_sha256(payload)
    assert optional_file_sha256(tmp_path / "missing.bin") is None
    assert optional_file_sha256(None) is None


def test_contract_mismatch_paths_and_preview_preserve_nested_fields() -> None:
    expected = {
        "benchmark_config": {
            "prompt_style": "standard",
            "triviaqa_bridge_manifest_identity": {"content_sha256": "new"},
        },
        "classifier": {"content_sha256": "classifier-v2"},
    }
    observed = {
        "benchmark_config": {"prompt_style": "anti_compliance"},
        "classifier": {},
    }

    mismatches = contract_mismatch_paths(expected, observed)

    assert "benchmark_config.prompt_style" in mismatches
    assert (
        "benchmark_config.triviaqa_bridge_manifest_identity (missing stored value)"
        in mismatches
    )
    assert "classifier.content_sha256 (missing stored value)" in mismatches
    assert format_mismatch_preview([f"field_{idx}" for idx in range(9)]) == (
        "field_0, field_1, field_2, field_3, field_4, field_5, "
        "field_6, field_7, ... (9 total)"
    )


def test_write_or_validate_run_contract_rejects_nested_drift(
    tmp_path: Path,
) -> None:
    contract = {
        "schema_version": NEGATIVE_CONTROL_RUN_CONFIG_SCHEMA_VERSION,
        "benchmark_config": {"prompt_style": "standard"},
    }
    contract_path = write_or_validate_run_contract(
        output_dir=str(tmp_path),
        contract=contract,
        contract_filename=CONTROL_RUN_CONFIG_FILENAME,
        existing_output_paths=[],
        output_kind="negative-control outputs",
        contract_label="negative-control run contract",
        fresh_output_flag="--output_base",
        write_if_missing=True,
    )

    assert json.loads(Path(contract_path).read_text()) == contract
    drifted = {
        "schema_version": NEGATIVE_CONTROL_RUN_CONFIG_SCHEMA_VERSION,
        "benchmark_config": {"prompt_style": "anti_compliance"},
    }

    with pytest.raises(RuntimeError, match=r"benchmark_config\.prompt_style"):
        write_or_validate_run_contract(
            output_dir=str(tmp_path),
            contract=drifted,
            contract_filename=CONTROL_RUN_CONFIG_FILENAME,
            existing_output_paths=[],
            output_kind="negative-control outputs",
            contract_label="negative-control run contract",
            fresh_output_flag="--output_base",
            write_if_missing=True,
        )


def test_negative_control_guard_rejects_configless_output_files(
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "seed_0_unconstrained"
    seed_dir.mkdir()
    (seed_dir / "alpha_0.0.jsonl").write_text('{"id": "old"}\n')

    with pytest.raises(
        RuntimeError, match=r"without control_run_config\.json"
    ) as exc_info:
        assert_or_write_negative_control_run_contract(
            str(tmp_path),
            {"schema_version": NEGATIVE_CONTROL_RUN_CONFIG_SCHEMA_VERSION},
            [0.0],
            [("seed_0_unconstrained", "unconstrained", 0)],
            write_if_missing=True,
        )

    assert "seed_0_unconstrained/alpha_0.0.jsonl" in str(exc_info.value)


def test_h_neuron_baseline_binding_hashes_and_checks_contract(
    tmp_path: Path,
) -> None:
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    results_path = baseline_dir / "results.20260509_000000.json"
    results_path.write_text(
        json.dumps(
            {
                "benchmark": "faitheval",
                "model_key": "mistral24b",
                "model": "mistralai/Mistral-Small-24B-Instruct-2501",
                "classifier": "classifier.pkl",
                "sample_manifest": {
                    "path": "manifest.json",
                    "content_sha256": "manifest-sha",
                },
                "prompt_style": "standard",
                "results": {"0.0": {}, "1.0": {}, "3.0": {}},
            },
            indent=2,
        )
    )
    contract_path = baseline_dir / INTERVENTION_RUN_CONFIG_FILENAME
    contract_path.write_text(json.dumps(_intervention_contract(), indent=2))

    binding = load_h_neuron_baseline_binding(str(baseline_dir), [0.0, 1.0, 3.0])

    assert binding["results_path"] == str(results_path)
    assert binding["results_content_sha256"] == file_sha256(results_path)
    assert binding["contract_path"] == str(contract_path)
    assert binding["contract_content_sha256"] == file_sha256(contract_path)

    contract = _negative_control_contract(binding)
    assert_h_neuron_baseline_matches_control_contract(contract)

    drifted = copy.deepcopy(contract)
    drifted["h_neuron_baseline"]["run_contract"]["benchmark_config"]["prompt_style"] = (
        "anti_compliance"
    )
    with pytest.raises(RuntimeError, match=r"benchmark_config\.prompt_style"):
        assert_h_neuron_baseline_matches_control_contract(drifted)

    contract_path.unlink()
    with pytest.raises(RuntimeError, match=r"missing intervention_run_config\.json"):
        load_h_neuron_baseline_binding(str(baseline_dir), [0.0, 1.0, 3.0])
