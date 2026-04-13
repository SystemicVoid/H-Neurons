from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from jailbreak_measurement_cleanup import (
    _collect_manual_examples,
    _summarize_outcomes,
    build_canary,
    require_canary_pass,
    validate_scored_dir,
    validate_canary,
)


def _write_alpha(path: Path, alpha: float, n_rows: int, *, prefix: str) -> None:
    rows = []
    for idx in range(n_rows):
        rows.append(
            {
                "id": f"{prefix}_{alpha:.1f}_{idx}",
                "alpha": alpha,
                "goal": f"goal {idx}",
                "category": "test",
                "response": f"response {idx}",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_valid_v3(path: Path, alpha: float, n_rows: int, *, prefix: str) -> None:
    rows = []
    for idx in range(n_rows):
        rows.append(
            {
                "id": f"{prefix}_{alpha:.1f}_{idx}",
                "alpha": alpha,
                "goal": f"goal {idx}",
                "category": "test",
                "response": f"response {idx}",
                "csv2": {
                    "schema_version": "csv2_v3",
                    "primary_outcome": "deflection"
                    if idx % 2 == 0
                    else "partial_compliance",
                    "intent_match": 1 if idx % 2 else 0,
                    "harmful_binary": "yes" if idx % 2 else "no",
                    "C": 2 if idx % 2 else 0,
                    "S": 2 if idx % 2 else 0,
                    "V": 1 if idx % 2 else None,
                    "R": bool(idx % 2),
                    "D": 1 if idx % 2 else 0,
                    "P": False,
                    "F": False,
                    "T": False,
                    "span_errors": 0,
                    "validation_errors": [],
                    "validation_warnings": [],
                },
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.fixture()
def canary_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    state_root = tmp_path / "state"
    h_source = tmp_path / "h_source"
    seed_source = tmp_path / "seed_source"
    for alpha in (0.0, 1.0):
        _write_alpha(h_source / f"alpha_{alpha:.1f}.jsonl", alpha, 25, prefix="h")
        _write_alpha(seed_source / f"alpha_{alpha:.1f}.jsonl", alpha, 25, prefix="seed")
    return state_root, h_source, seed_source


def test_build_canary_uses_first_n_records(
    canary_workspace: tuple[Path, Path, Path],
) -> None:
    state_root, h_source, seed_source = canary_workspace

    manifest_path = build_canary(
        state_root,
        h_neuron_source=h_source,
        seed1_source=seed_source,
        alphas=[0.0, 1.0],
        canary_rows=20,
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["canary_rows_per_alpha"] == 20
    h_rows = [
        json.loads(line)
        for line in (state_root / "canary_inputs" / "h_neuron" / "alpha_0.0.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(h_rows) == 20
    assert h_rows[0]["id"] == "h_0.0_0"
    assert h_rows[-1]["id"] == "h_0.0_19"
    assert manifest["jobs"]["h_neuron"]["selected_ids"]["0.0"][-1] == "h_0.0_19"


def test_validate_canary_passes_for_structurally_valid_outputs(
    canary_workspace: tuple[Path, Path, Path],
) -> None:
    state_root, h_source, seed_source = canary_workspace
    build_canary(
        state_root,
        h_neuron_source=h_source,
        seed1_source=seed_source,
        alphas=[0.0, 1.0],
        canary_rows=20,
    )

    for alpha in (0.0, 1.0):
        _write_valid_v3(
            state_root / "canary_v3" / "h_neuron" / f"alpha_{alpha:.1f}.jsonl",
            alpha,
            20,
            prefix="h",
        )
        _write_valid_v3(
            state_root / "canary_v3" / "seed_1_control" / f"alpha_{alpha:.1f}.jsonl",
            alpha,
            20,
            prefix="seed",
        )

    summary = validate_canary(state_root, alphas=[0.0, 1.0])
    assert summary["passed"] is True
    require_canary_pass(state_root)
    assert (state_root / "canary_report.md").exists()


def test_validate_canary_allows_warning_only_validation_issues(
    canary_workspace: tuple[Path, Path, Path],
) -> None:
    state_root, h_source, seed_source = canary_workspace
    build_canary(
        state_root,
        h_neuron_source=h_source,
        seed1_source=seed_source,
        alphas=[0.0],
        canary_rows=20,
    )

    _write_valid_v3(
        state_root / "canary_v3" / "h_neuron" / "alpha_0.0.jsonl",
        0.0,
        20,
        prefix="h",
    )
    _write_valid_v3(
        state_root / "canary_v3" / "seed_1_control" / "alpha_0.0.jsonl",
        0.0,
        20,
        prefix="seed",
    )

    seed_path = state_root / "canary_v3" / "seed_1_control" / "alpha_0.0.jsonl"
    seed_rows = [json.loads(line) for line in seed_path.read_text().splitlines()]
    seed_rows[4]["csv2"]["span_errors"] = 1
    seed_rows[4]["csv2"]["validation_warnings"] = ["invalid_benign_spans:1"]
    with open(seed_path, "w") as f:
        for row in seed_rows:
            f.write(json.dumps(row) + "\n")

    summary = validate_canary(state_root, alphas=[0.0])

    assert summary["passed"] is True
    alpha_summary = summary["jobs"]["seed_1_control"]["alphas"]["0.0"]
    assert alpha_summary["failures"] == []
    assert alpha_summary["warnings"] == ["validation_warnings:4:invalid_benign_spans:1"]
    assert alpha_summary["parse_stats"]["validation_warnings"] == 1
    assert alpha_summary["parse_stats"]["span_errors"] == 1
    require_canary_pass(state_root)


def test_validate_canary_fails_on_missing_csv2(
    canary_workspace: tuple[Path, Path, Path],
) -> None:
    state_root, h_source, seed_source = canary_workspace
    build_canary(
        state_root,
        h_neuron_source=h_source,
        seed1_source=seed_source,
        alphas=[0.0],
        canary_rows=20,
    )
    _write_valid_v3(
        state_root / "canary_v3" / "h_neuron" / "alpha_0.0.jsonl",
        0.0,
        20,
        prefix="h",
    )
    _write_valid_v3(
        state_root / "canary_v3" / "seed_1_control" / "alpha_0.0.jsonl",
        0.0,
        20,
        prefix="seed",
    )
    seed_rows = [
        json.loads(line)
        for line in (state_root / "canary_v3" / "seed_1_control" / "alpha_0.0.jsonl")
        .read_text()
        .splitlines()
    ]
    del seed_rows[3]["csv2"]
    with open(
        state_root / "canary_v3" / "seed_1_control" / "alpha_0.0.jsonl", "w"
    ) as f:
        for row in seed_rows:
            f.write(json.dumps(row) + "\n")

    summary = validate_canary(state_root, alphas=[0.0])
    assert summary["passed"] is False
    with pytest.raises(ValueError, match="Canary did not pass"):
        require_canary_pass(state_root)


def test_validate_canary_fails_on_join_mismatch(
    canary_workspace: tuple[Path, Path, Path],
) -> None:
    state_root, h_source, seed_source = canary_workspace
    build_canary(
        state_root,
        h_neuron_source=h_source,
        seed1_source=seed_source,
        alphas=[0.0],
        canary_rows=20,
    )
    _write_valid_v3(
        state_root / "canary_v3" / "h_neuron" / "alpha_0.0.jsonl",
        0.0,
        20,
        prefix="h",
    )
    _write_valid_v3(
        state_root / "canary_v3" / "seed_1_control" / "alpha_0.0.jsonl",
        0.0,
        20,
        prefix="seed",
    )
    h_rows = [
        json.loads(line)
        for line in (state_root / "canary_v3" / "h_neuron" / "alpha_0.0.jsonl")
        .read_text()
        .splitlines()
    ]
    h_rows[5]["id"] = "wrong_id"
    with open(state_root / "canary_v3" / "h_neuron" / "alpha_0.0.jsonl", "w") as f:
        for row in h_rows:
            f.write(json.dumps(row) + "\n")

    summary = validate_canary(state_root, alphas=[0.0])
    assert summary["passed"] is False
    failures = summary["jobs"]["h_neuron"]["alphas"]["0.0"]["failures"]
    assert "join_key_order_mismatch" in failures


def test_validate_canary_fails_on_schema_and_validation_errors(
    canary_workspace: tuple[Path, Path, Path],
) -> None:
    state_root, h_source, seed_source = canary_workspace
    build_canary(
        state_root,
        h_neuron_source=h_source,
        seed1_source=seed_source,
        alphas=[0.0],
        canary_rows=20,
    )
    _write_valid_v3(
        state_root / "canary_v3" / "h_neuron" / "alpha_0.0.jsonl",
        0.0,
        20,
        prefix="h",
    )
    _write_valid_v3(
        state_root / "canary_v3" / "seed_1_control" / "alpha_0.0.jsonl",
        0.0,
        20,
        prefix="seed",
    )
    h_rows = [
        json.loads(line)
        for line in (state_root / "canary_v3" / "h_neuron" / "alpha_0.0.jsonl")
        .read_text()
        .splitlines()
    ]
    h_rows[0]["csv2"]["schema_version"] = "csv2_v2"
    h_rows[1]["csv2"]["validation_errors"] = ["missing_harmful_evidence"]
    with open(state_root / "canary_v3" / "h_neuron" / "alpha_0.0.jsonl", "w") as f:
        for row in h_rows:
            f.write(json.dumps(row) + "\n")

    summary = validate_canary(state_root, alphas=[0.0])
    assert summary["passed"] is False
    failures = summary["jobs"]["h_neuron"]["alphas"]["0.0"]["failures"]
    assert any(failure.startswith("schema_version:0:") for failure in failures)
    assert any(failure.startswith("validation_errors:1:") for failure in failures)


def test_validate_scored_dir_passes_for_structurally_valid_outputs(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_alpha(input_dir / "alpha_0.0.jsonl", 0.0, 5, prefix="h")
    _write_valid_v3(output_dir / "alpha_0.0.jsonl", 0.0, 5, prefix="h")

    summary = validate_scored_dir(input_dir, output_dir, alphas=[0.0])

    assert summary["passed"] is True
    assert summary["per_alpha"]["0.0"]["failures"] == []


def test_validate_scored_dir_allows_warning_only_validation_issues(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_alpha(input_dir / "alpha_0.0.jsonl", 0.0, 5, prefix="h")
    _write_valid_v3(output_dir / "alpha_0.0.jsonl", 0.0, 5, prefix="h")

    rows = [
        json.loads(line)
        for line in (output_dir / "alpha_0.0.jsonl").read_text().splitlines()
    ]
    rows[1]["csv2"]["span_errors"] = 1
    rows[1]["csv2"]["validation_warnings"] = ["invalid_benign_spans:1"]
    with open(output_dir / "alpha_0.0.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = validate_scored_dir(input_dir, output_dir, alphas=[0.0])

    assert summary["passed"] is True
    assert summary["per_alpha"]["0.0"]["failures"] == []
    assert summary["per_alpha"]["0.0"]["warnings"] == [
        "validation_warnings:1:invalid_benign_spans:1"
    ]
    assert summary["per_alpha"]["0.0"]["parse_stats"]["validation_warnings"] == 1


def test_validate_scored_dir_fails_on_batch_error_placeholders(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_alpha(input_dir / "alpha_0.0.jsonl", 0.0, 5, prefix="h")
    _write_valid_v3(output_dir / "alpha_0.0.jsonl", 0.0, 5, prefix="h")

    rows = [
        json.loads(line)
        for line in (output_dir / "alpha_0.0.jsonl").read_text().splitlines()
    ]
    rows[2]["csv2"] = {"error": "missing_from_batch"}
    with open(output_dir / "alpha_0.0.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = validate_scored_dir(input_dir, output_dir, alphas=[0.0])

    assert summary["passed"] is False
    assert "csv2_error:2:missing_from_batch" in summary["per_alpha"]["0.0"]["failures"]


def test_summarize_outcomes_skips_placeholder_rows() -> None:
    outcomes = _summarize_outcomes(
        [
            {
                "id": "placeholder",
                "alpha": 0.0,
                "response": "pending",
                "csv2": {"error": "missing_from_batch"},
            },
            {
                "id": "real",
                "alpha": 0.0,
                "response": "I refuse.",
                "csv2": {
                    "schema_version": "csv2_v3",
                    "primary_outcome": "refusal",
                    "intent_match": 0,
                    "harmful_binary": "no",
                    "C": 0,
                    "S": 0,
                    "V": None,
                    "R": True,
                    "D": 1,
                    "P": False,
                    "F": False,
                    "T": False,
                    "span_errors": 0,
                    "validation_errors": [],
                    "validation_warnings": [],
                },
            },
        ]
    )

    assert outcomes["error:missing_from_batch"] == 1
    assert outcomes["primary_outcome:refusal"] == 1
    assert outcomes["harmful_binary:no"] == 1
    assert "primary_outcome:deflection" not in outcomes


def test_collect_manual_examples_skips_placeholder_rows() -> None:
    examples = _collect_manual_examples(
        [
            {
                "id": "placeholder",
                "alpha": 0.0,
                "response": "pending",
                "csv2": {"error": "parse_failed"},
            },
            {
                "id": "real",
                "alpha": 0.0,
                "response": "I refuse to help with that request.",
                "csv2": {
                    "schema_version": "csv2_v3",
                    "primary_outcome": "refusal",
                    "intent_match": 0,
                    "harmful_binary": "no",
                    "C": 0,
                    "S": 0,
                    "V": None,
                    "R": True,
                    "D": 1,
                    "P": False,
                    "F": False,
                    "T": False,
                    "span_errors": 0,
                    "validation_errors": [],
                    "validation_warnings": [],
                },
            },
        ]
    )

    assert [example["id"] for example in examples] == ["real"]
