from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from export_simid_open_review_package import (
    DEFAULT_AGREEMENT_SAMPLE_SIZE,
    DEFAULT_REVIEW_SEED,
    DEFAULT_RUN_DIR,
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
