from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from export_simid_open_review_package import (
    DEFAULT_AGREEMENT_SAMPLE_SIZE,
    DEFAULT_LLM_BATCH_SIZE,
    DEFAULT_REVIEW_SEED,
    DEFAULT_RUN_DIR,
    LLM_BATCH_LABEL_SCHEMA_VERSION,
    LLM_BATCH_ROOT_DIR,
    PRIMARY_SECONDARY_DISAGREEMENT,
    STRATIFIED_AGREEMENT_SAMPLE,
    blind_review_row,
    export_package,
    file_sha256,
    load_calibration_cases,
    parse_args,
    priority_tags,
    selected_review_cases,
)
from review_package import (
    LLM_BLIND_MAP_SCHEMA_VERSION,
    SIMID_OPEN_REVIEW_SCHEMAS,
)
from validate_simid_open_review_labels import main as validate_labels_main


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def synthetic_llm_label_rows(
    batch_dir: Path,
    *,
    blind_cases_file_sha256: str | None = None,
) -> list[dict[str, Any]]:
    batch_hash = blind_cases_file_sha256 or file_sha256(
        batch_dir / "review_cases_blind.jsonl"
    )
    label_rows = []
    for row in load_jsonl(batch_dir / "review_cases_blind.jsonl"):
        label_rows.append(
            {
                "schema_version": LLM_BATCH_LABEL_SCHEMA_VERSION,
                "blind_case_id": row["blind_case_id"],
                "review_order": row["review_order"],
                "label": "INCORRECT",
                "confidence": 3,
                "rule_gap": False,
                "flags": [],
                "notes": "Synthetic validator fixture.",
                "rater": {
                    "type": "llm",
                    "model": "claude-opus-4.7",
                    "prompt_version": "simid_open_independent_rater/v1",
                },
                "blind_cases_file_sha256": batch_hash,
            }
        )
    return label_rows


def write_synthetic_llm_labels(output_dir: Path) -> None:
    for batch_dir in sorted((output_dir / LLM_BATCH_ROOT_DIR).iterdir()):
        if not batch_dir.is_dir():
            continue
        write_jsonl(
            batch_dir / "opus_4_7_labels.jsonl",
            synthetic_llm_label_rows(batch_dir),
        )


def test_review_package_selection_includes_all_disagreements() -> None:
    cases = load_calibration_cases(DEFAULT_RUN_DIR)
    selected = selected_review_cases(
        cases,
        agreement_sample_size=DEFAULT_AGREEMENT_SAMPLE_SIZE,
        seed=DEFAULT_REVIEW_SEED,
    )

    disagreement_ids = {
        case.case_id for case in cases if case.primary_label != case.secondary_label
    }
    selected_disagreement_ids = {
        case.case_id
        for review_set, case in selected
        if review_set == PRIMARY_SECONDARY_DISAGREEMENT
    }
    selected_agreement_ids = {
        case.case_id
        for review_set, case in selected
        if review_set == STRATIFIED_AGREEMENT_SAMPLE
    }

    assert len(disagreement_ids) == 34
    assert selected_disagreement_ids == disagreement_ids
    assert len(selected_agreement_ids) == DEFAULT_AGREEMENT_SAMPLE_SIZE
    assert len({case.case_id for _, case in selected}) == len(selected)
    assert any(
        review_set == STRATIFIED_AGREEMENT_SAMPLE
        for review_set, _ in selected[: len(disagreement_ids)]
    )
    assert any(
        review_set == PRIMARY_SECONDARY_DISAGREEMENT
        for review_set, _ in selected[len(disagreement_ids) :]
    )


def test_review_package_prioritizes_requested_boundary_families() -> None:
    cases = load_calibration_cases(DEFAULT_RUN_DIR)
    selected = selected_review_cases(
        cases,
        agreement_sample_size=DEFAULT_AGREEMENT_SAMPLE_SIZE,
        seed=DEFAULT_REVIEW_SEED,
    )
    selected_cases = [case for _, case in selected]
    selected_tags = {tag for case in selected_cases for tag in priority_tags(case)}

    assert any(case.dataset == "triviaqa_bridge" for case in selected_cases)
    assert "bridge_partial_entity_or_modifier" in selected_tags
    assert (
        "truthfulqa_non_answer_boundary" in selected_tags
        or "truthfulqa_qualified_answer_boundary" in selected_tags
    )


def test_blind_review_rows_do_not_expose_existing_labels() -> None:
    cases = load_calibration_cases(DEFAULT_RUN_DIR)
    selected = selected_review_cases(
        cases,
        agreement_sample_size=2,
        seed=DEFAULT_REVIEW_SEED,
    )

    row = blind_review_row(
        selected[0][1],
        review_order=1,
    )

    forbidden_keys = {
        "review_set",
        "sample_id",
        "base_sample_id",
        "dataset",
        "condition",
        "alpha",
        "mc_endpoint",
        "priority_tags",
        "primary_label",
        "secondary_label",
        "adjudicated_label",
        "deterministic_open_grade",
        "primary_judge_grade",
        "primary_effective_open_grade",
        "primary_open_adjudication",
        "sample_source",
        "sampling_stratum",
    }
    assert not (forbidden_keys & set(row))
    assert {"question", "gold_aliases", "predicted_answer"} <= set(row)


def test_exported_static_ui_is_blind(tmp_path: Path) -> None:
    output_dir = tmp_path / "review_package"
    manifest = export_package(
        parse_args(
            [
                "--run-dir",
                str(DEFAULT_RUN_DIR),
                "--output-dir",
                str(output_dir),
            ]
        )
    )

    assert not (output_dir / "review_audit_context.jsonl").exists()
    assert (output_dir / "adjudication_rule.md").exists()
    assert (output_dir / "label_schema.json").exists()
    assert (
        file_sha256(output_dir / "review_cases_blind.jsonl")
        == manifest["blind_cases_file_sha256"]
    )

    index_html = (output_dir / "index.html").read_text(encoding="utf-8")
    forbidden_fragments = {
        "primary_label",
        "secondary_label",
        "adjudicated_label",
        "deterministic_open_grade",
        "sample_id",
        "base_sample_id",
        "sample_source",
        "sampling_stratum",
        "mc_endpoint",
        "review_audit_context",
        "Reveal Context",
        "Import Perspective",
        "package_manifest_sha256",
    }
    assert not any(fragment in index_html for fragment in forbidden_fragments)

    blind_text = (output_dir / "review_cases_blind.jsonl").read_text(encoding="utf-8")
    assert "\ufffd" not in blind_text


def test_exported_llm_batches_are_small_complete_and_blind(tmp_path: Path) -> None:
    output_dir = tmp_path / "review_package"
    manifest = export_package(
        parse_args(
            [
                "--run-dir",
                str(DEFAULT_RUN_DIR),
                "--output-dir",
                str(output_dir),
                "--llm-batch-size",
                str(DEFAULT_LLM_BATCH_SIZE),
            ]
        )
    )

    batch_root = output_dir / LLM_BATCH_ROOT_DIR
    batch_dirs = sorted(path for path in batch_root.iterdir() if path.is_dir())
    all_blind_ids: set[str] = set()
    all_review_orders: set[int] = set()

    assert len(batch_dirs) == 10
    assert len(manifest["llm_batching"]["batches"]) == 10
    for batch_dir in batch_dirs:
        rows = load_jsonl(batch_dir / "review_cases_blind.jsonl")
        assert 1 <= len(rows) <= DEFAULT_LLM_BATCH_SIZE
        assert (batch_dir / "adjudication_rule.md").exists()
        assert (batch_dir / "label_schema.json").exists()
        assert (batch_dir / "prompt.md").exists()
        for row in rows:
            assert "blind_case_id" in row
            assert "calibration_case_id" not in row
            all_blind_ids.add(str(row["blind_case_id"]))
            all_review_orders.add(int(row["review_order"]))

    assert len(all_blind_ids) == manifest["selection"]["n_review_cases"]
    assert all_review_orders == set(
        range(1, manifest["selection"]["n_review_cases"] + 1)
    )

    private_map = load_jsonl(output_dir / "llm_blind_case_map.jsonl")
    assert {str(row["blind_case_id"]) for row in private_map} == all_blind_ids
    assert all("calibration_case_id" in row for row in private_map)


def test_review_package_schema_constants_bind_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "review_package"
    manifest = export_package(
        parse_args(
            [
                "--run-dir",
                str(DEFAULT_RUN_DIR),
                "--output-dir",
                str(output_dir),
            ]
        )
    )

    assert manifest["schema_version"] == SIMID_OPEN_REVIEW_SCHEMAS.package
    assert (
        manifest["label_schema"]["schema_version"]
        == SIMID_OPEN_REVIEW_SCHEMAS.human_label
    )

    blind_rows = load_jsonl(output_dir / "review_cases_blind.jsonl")
    assert {row["schema_version"] for row in blind_rows} == {
        SIMID_OPEN_REVIEW_SCHEMAS.blind_case
    }

    map_rows = load_jsonl(output_dir / "llm_blind_case_map.jsonl")
    assert {row["schema_version"] for row in map_rows} == {LLM_BLIND_MAP_SCHEMA_VERSION}

    first_batch = output_dir / LLM_BATCH_ROOT_DIR / "batch_001"
    batch_rows = load_jsonl(first_batch / "review_cases_blind.jsonl")
    assert {row["schema_version"] for row in batch_rows} == {
        SIMID_OPEN_REVIEW_SCHEMAS.llm_batch_case
    }

    label_schema = json.loads((output_dir / "label_schema.json").read_text())
    assert (
        label_schema["properties"]["schema_version"]["const"]
        == SIMID_OPEN_REVIEW_SCHEMAS.human_label
    )
    batch_schema = json.loads((first_batch / "label_schema.json").read_text())
    assert (
        batch_schema["properties"]["schema_version"]["const"]
        == SIMID_OPEN_REVIEW_SCHEMAS.llm_batch_label
    )


def test_llm_batch_tree_does_not_expose_calibration_metadata(tmp_path: Path) -> None:
    output_dir = tmp_path / "review_package"
    export_package(
        parse_args(
            [
                "--run-dir",
                str(DEFAULT_RUN_DIR),
                "--output-dir",
                str(output_dir),
            ]
        )
    )

    forbidden_fragments = {
        "primary_label",
        "secondary_label",
        "adjudicated_label",
        "deterministic_open_grade",
        "review_set",
        "sample_source",
        "source_calibration_claimability",
        "cohen_kappa",
        "open_calibration_queue",
        "open_calibration_secondary_labels",
        "open_calibration_adjudications",
        "open_calibration_summary",
        "calibration_case_id",
    }
    batch_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((output_dir / LLM_BATCH_ROOT_DIR).rglob("*"))
        if path.is_file()
    )
    assert not any(fragment in batch_text for fragment in forbidden_fragments)


def test_llm_batch_prompts_are_disk_first_and_batch_specific(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "review_package"
    export_package(
        parse_args(
            [
                "--run-dir",
                str(DEFAULT_RUN_DIR),
                "--output-dir",
                str(output_dir),
            ]
        )
    )

    controller_prompt = (output_dir / "opus_4_7_independent_rater_prompt.md").read_text(
        encoding="utf-8"
    )
    assert "Exactly 100" not in controller_prompt
    assert "Return newline-delimited JSON only" not in controller_prompt
    assert "Do not print generated JSONL rows" in controller_prompt

    first_batch = output_dir / LLM_BATCH_ROOT_DIR / "batch_001"
    batch_prompt = (first_batch / "prompt.md").read_text(encoding="utf-8")
    assert "Write newline-delimited JSON to `opus_4_7_labels.jsonl`" in batch_prompt
    assert "Exactly 10 JSONL rows" in batch_prompt
    assert "calibration_case_id" not in batch_prompt

    batch_schema = json.loads((first_batch / "label_schema.json").read_text())
    assert batch_schema["properties"]["blind_cases_file_sha256"][
        "const"
    ] == file_sha256(first_batch / "review_cases_blind.jsonl")


def test_llm_batch_labels_validate_and_merge(tmp_path: Path) -> None:
    output_dir = tmp_path / "review_package"
    manifest = export_package(
        parse_args(
            [
                "--run-dir",
                str(DEFAULT_RUN_DIR),
                "--output-dir",
                str(output_dir),
            ]
        )
    )

    write_synthetic_llm_labels(output_dir)

    output_file = output_dir / "opus_4_7_labels.jsonl"
    assert (
        validate_labels_main(
            [
                "--package-dir",
                str(output_dir),
                "--output",
                str(output_file),
            ]
        )
        == 0
    )
    merged = load_jsonl(output_file)
    assert len(merged) == manifest["selection"]["n_review_cases"]
    assert "calibration_case_id" in merged[0]
    assert "blind_case_id" not in merged[0]
    assert {row["blind_cases_file_sha256"] for row in merged} == {
        manifest["blind_cases_file_sha256"]
    }


def test_llm_batch_validator_rejects_duplicate_blind_case_ids(tmp_path: Path) -> None:
    output_dir = tmp_path / "review_package"
    export_package(
        parse_args(
            [
                "--run-dir",
                str(DEFAULT_RUN_DIR),
                "--output-dir",
                str(output_dir),
            ]
        )
    )

    first_batch = output_dir / LLM_BATCH_ROOT_DIR / "batch_001"
    rows = load_jsonl(first_batch / "review_cases_blind.jsonl")
    rows[1]["blind_case_id"] = rows[0]["blind_case_id"]
    write_jsonl(first_batch / "review_cases_blind.jsonl", rows)
    label_file = first_batch / "opus_4_7_labels.jsonl"
    write_jsonl(label_file, synthetic_llm_label_rows(first_batch))

    with pytest.raises(ValueError, match="duplicate blind_case_id"):
        validate_labels_main(
            [
                "--label-file",
                str(label_file),
            ]
        )


def test_llm_batch_validator_rejects_label_package_hash_mismatch(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "review_package"
    export_package(
        parse_args(
            [
                "--run-dir",
                str(DEFAULT_RUN_DIR),
                "--output-dir",
                str(output_dir),
            ]
        )
    )

    first_batch = output_dir / LLM_BATCH_ROOT_DIR / "batch_001"
    label_file = first_batch / "opus_4_7_labels.jsonl"
    write_jsonl(
        label_file,
        synthetic_llm_label_rows(first_batch, blind_cases_file_sha256="0" * 64),
    )

    with pytest.raises(ValueError, match="blind_cases_file_sha256 mismatch"):
        validate_labels_main(
            [
                "--label-file",
                str(label_file),
                "--batch-dir",
                str(first_batch),
            ]
        )


def test_llm_label_merge_requires_private_map_binding(tmp_path: Path) -> None:
    output_dir = tmp_path / "review_package"
    export_package(
        parse_args(
            [
                "--run-dir",
                str(DEFAULT_RUN_DIR),
                "--output-dir",
                str(output_dir),
            ]
        )
    )
    write_synthetic_llm_labels(output_dir)

    map_file = output_dir / "llm_blind_case_map.jsonl"
    map_rows = load_jsonl(map_file)
    write_jsonl(map_file, map_rows[:-1])

    with pytest.raises(ValueError, match="missing map row"):
        validate_labels_main(
            [
                "--package-dir",
                str(output_dir),
                "--output",
                str(output_dir / "opus_4_7_labels.jsonl"),
            ]
        )
