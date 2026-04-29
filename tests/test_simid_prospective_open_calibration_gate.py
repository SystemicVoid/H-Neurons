from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from analyze_simid_prospective_open_calibration_gate import analyze_package
from export_simid_prospective_open_calibration_gate import (
    DEFAULT_EVIDENCE_JSONL,
    DEFAULT_RUN_DIR,
    LABEL_SCHEMA_VERSION,
    example_exact_keys,
    export_package,
    file_sha256,
    load_jsonl,
    parse_args,
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def export_tmp_package(tmp_path: Path, *, sample_size: int = 24) -> Path:
    output_dir = tmp_path / "prospective_gate"
    export_package(
        parse_args(
            [
                "--run-dir",
                str(DEFAULT_RUN_DIR),
                "--evidence-jsonl",
                str(DEFAULT_EVIDENCE_JSONL),
                "--output-dir",
                str(output_dir),
                "--sample-size",
                str(sample_size),
                "--seed",
                "test_prospective_gate_seed",
            ]
        )
    )
    return output_dir


def test_prospective_package_blinds_rows_and_excludes_training_examples(
    tmp_path: Path,
) -> None:
    output_dir = export_tmp_package(tmp_path)
    manifest = read_json(output_dir / "review_manifest.json")
    blind_rows = load_jsonl(output_dir / "review_cases_blind.jsonl")
    private_rows = load_jsonl(output_dir / "private_case_map.jsonl")

    assert manifest["selection"]["n_review_cases"] == 24
    assert len(blind_rows) == len(private_rows) == 24
    assert {row["blind_case_id"] for row in blind_rows} == {
        row["blind_case_id"] for row in private_rows
    }
    assert manifest["selection"]["sample_counts_by_reference_label"]

    forbidden_keys = {
        "source_case_id",
        "sample_id",
        "base_sample_id",
        "dataset",
        "condition",
        "alpha",
        "mc_endpoint",
        "reference_label",
        "primary_label",
        "secondary_label",
        "adjudicated_label",
        "sampling_stratum",
    }
    for row in blind_rows:
        assert not (forbidden_keys & set(row))
        assert {"blind_case_id", "review_order", "question"} <= set(row)

    prior_queue_ids = {
        row["calibration_case_id"]
        for row in load_jsonl(DEFAULT_RUN_DIR / "open_calibration_queue.jsonl")
    }
    assert prior_queue_ids.isdisjoint({row["source_case_id"] for row in private_rows})

    evidence_rows = load_jsonl(DEFAULT_EVIDENCE_JSONL)
    excluded_exact_keys = example_exact_keys(evidence_rows)
    sampled_exact_keys = {
        (row["question_normalized"], row["predicted_answer_normalized"])
        for row in private_rows
    }
    assert sampled_exact_keys.isdisjoint(excluded_exact_keys)


def test_prospective_rubric_freezes_required_hard_case_examples(
    tmp_path: Path,
) -> None:
    output_dir = export_tmp_package(tmp_path, sample_size=12)
    rubric = (output_dir / "rubric.md").read_text(encoding="utf-8")
    manifest = read_json(output_dir / "review_manifest.json")

    for phrase in [
        "Core i9 vs Core i9 Apple Silicon",
        "pithivier custard/fruit filling boundary",
        "viscous fluid modifier",
        "Peter Piper plain peppers",
        "Amal Clooney lawyer vs human-rights advocate",
        "cranberry modifier exact repeat",
        "Adam first man boundary",
    ]:
        assert phrase in rubric
        assert phrase in manifest["rubric"]["hard_case_groups"]
    assert "not retrospective evidence" in rubric
    assert "excludes exact normalized question-answer pairs" in rubric


def label_rows_from_private(
    *,
    output_dir: Path,
    label_override: str | None = None,
    rule_gap: bool = False,
) -> list[dict[str, Any]]:
    manifest = read_json(output_dir / "review_manifest.json")
    blind_hash = manifest["files"]["review_cases_blind"]["content_sha256"]
    rubric_hash = manifest["rubric"]["content_sha256"]
    rows = []
    for private in load_jsonl(output_dir / "private_case_map.jsonl"):
        rows.append(
            {
                "schema_version": LABEL_SCHEMA_VERSION,
                "blind_case_id": private["blind_case_id"],
                "review_order": private["review_order"],
                "label": label_override or private["reference_label"],
                "confidence": 4,
                "rule_gap": rule_gap,
                "flags": [],
                "notes": "Synthetic fixture label.",
                "rater": {
                    "type": "llm",
                    "model": "fixture",
                    "prompt_version": "simid_open_prospective_calibration_rater/v1",
                },
                "blind_cases_file_sha256": blind_hash,
                "rubric_sha256": rubric_hash,
            }
        )
    return rows


def test_prospective_analyzer_passes_and_fails_small_fixtures(
    tmp_path: Path,
) -> None:
    output_dir = export_tmp_package(tmp_path, sample_size=12)
    passing_labels = output_dir / "passing_labels.jsonl"
    write_jsonl(passing_labels, label_rows_from_private(output_dir=output_dir))

    passing = analyze_package(package_dir=output_dir, label_file=passing_labels)
    assert passing["outcome"]["passed"] is True
    assert passing["metrics"]["raw_agreement"]["count"] == 12
    assert passing["metrics"]["cohen_kappa"] == 1.0
    assert passing["metrics"]["gwet_ac1"] == 1.0

    failing_labels = output_dir / "failing_labels.jsonl"
    write_jsonl(
        failing_labels,
        label_rows_from_private(
            output_dir=output_dir,
            label_override="CORRECT",
            rule_gap=True,
        ),
    )
    failing = analyze_package(package_dir=output_dir, label_file=failing_labels)
    assert failing["outcome"]["passed"] is False
    assert "rule_gap_recorded" in failing["outcome"]["blockers"]
    assert (
        "cohen_kappa_below_threshold" in failing["outcome"]["blockers"]
        or "raw_agreement_below_threshold" in failing["outcome"]["blockers"]
    )


def test_reviewer_facing_rows_do_not_leak_private_reference_labels(
    tmp_path: Path,
) -> None:
    output_dir = export_tmp_package(tmp_path, sample_size=12)
    blind_text = (output_dir / "review_cases_blind.jsonl").read_text(encoding="utf-8")
    prompt_text = (output_dir / "prompt.md").read_text(encoding="utf-8")
    label_schema = read_json(output_dir / "label_schema.json")

    for fragment in [
        "reference_label",
        "source_case_id",
        "sample_id",
        "primary_label",
        "secondary_label",
        "adjudicated_label",
        "open_calibration_queue",
        "private_case_map",
    ]:
        assert fragment not in blind_text
        assert fragment not in prompt_text
    assert label_schema["properties"]["blind_cases_file_sha256"][
        "const"
    ] == file_sha256(output_dir / "review_cases_blind.jsonl")
    assert label_schema["properties"]["rubric_sha256"]["const"] == file_sha256(
        output_dir / "rubric.md"
    )


def test_prospective_analyzer_rejects_private_map_drift(tmp_path: Path) -> None:
    output_dir = export_tmp_package(tmp_path, sample_size=12)
    labels = output_dir / "labels.jsonl"
    write_jsonl(labels, label_rows_from_private(output_dir=output_dir))

    private_rows = load_jsonl(output_dir / "private_case_map.jsonl")
    private_rows[0]["reference_label"] = (
        "INCORRECT" if private_rows[0]["reference_label"] != "INCORRECT" else "CORRECT"
    )
    write_jsonl(output_dir / "private_case_map.jsonl", private_rows)

    with pytest.raises(ValueError, match="content_sha256 mismatch"):
        analyze_package(package_dir=output_dir, label_file=labels)
