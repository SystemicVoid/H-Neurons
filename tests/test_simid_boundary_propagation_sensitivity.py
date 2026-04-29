from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_simid_boundary_propagation_sensitivity import (  # noqa: E402
    NORMALIZATION_DESCRIPTION,
    build_row_corrections,
    evidence_rule_from_row,
    propagation_validation,
)


def _evidence_row(
    *,
    question: str,
    answer: str,
    fresh_label: str,
    family: str = "fixture_family",
    group_id: str = "fixture_group",
    eligibility: str = "eligible_exact_normalized_answer",
    local_status: str = "eligible_targeted_rows",
    exact_count: int = 1,
) -> dict[str, Any]:
    return {
        "case_family": family,
        "group_id": group_id,
        "fresh_label": fresh_label,
        "fresh_status": "consistent",
        "adjudication_status": "ADJUDICATED",
        "rule_gap": False,
        "review_orders": [1],
        "question": question,
        "predicted_answer_normalized": answer,
        "local_correction_evidence": {"status": local_status},
        "diagnostic_propagation": {
            "eligibility": eligibility,
            "match_rule": {
                "normalization": NORMALIZATION_DESCRIPTION,
                "question": question,
                "predicted_answer_normalized": answer,
            },
            "mvp_exact_pattern_count": exact_count,
            "broader_family_propagation": "not_allowed_without_new_review",
            "note": "Fixture diagnostic propagation note.",
        },
    }


def _rule(**kwargs: Any):
    return evidence_rule_from_row(_evidence_row(**kwargs), row_name="fixture")


def _mvp_row(
    *,
    question: str,
    response: str,
    label: str = "INCORRECT",
    sample_id: str = "sample_1",
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "base_sample_id": sample_id,
        "condition": "selected",
        "alpha": 8.0,
        "dataset": "fixture",
        "mc_endpoint": "fixture_mc",
        "question": question,
        "open_generation": {"response": response},
        "open_adjudication": {
            "judge_grade": label,
            "parse_result": {"verdict": label, "valid": True},
        },
    }


def test_exact_matching_uses_question_and_normalized_answer() -> None:
    rule = _rule(
        question="Which condiment?",
        answer="Mulled cranberry sauce",
        fresh_label="CORRECT",
    )
    rows = [
        _mvp_row(
            question="Which condiment?",
            response="mulled cranberry sauce.",
            label="INCORRECT",
        )
    ]

    corrections = build_row_corrections(rows, [rule])

    assert corrections[0]["propagation_status"] == "exact_label_replaced"
    assert corrections[0]["corrected_diagnostic_open_label"] == "CORRECT"
    assert rows[0]["diagnostic_exact_open_grade"]["correct"] is True


def test_same_answer_different_question_does_not_propagate() -> None:
    rule = _rule(
        question="Which condiment?",
        answer="Mulled cranberry sauce",
        fresh_label="CORRECT",
    )
    rows = [
        _mvp_row(
            question="Which dessert?",
            response="Mulled cranberry sauce.",
            label="INCORRECT",
        )
    ]

    corrections = build_row_corrections(rows, [rule])

    assert corrections[0]["propagation_status"] == "no_reviewed_question_evidence"
    assert corrections[0]["corrected_diagnostic_open_label"] == "INCORRECT"
    assert corrections[0]["label_changed"] is False


def test_core_i9_boundary_separates_apple_silicon_and_blocks_family() -> None:
    question = "What is the Intel Corporation's famous processor brandname?"
    core_i9 = _rule(
        question=question,
        answer="Core i9",
        fresh_label="CORRECT",
        family="core_i9_processor_brand_boundary",
        group_id="core_i9",
        exact_count=1,
    )
    core_i9_apple = _rule(
        question=question,
        answer="Core i9 Apple Silicon",
        fresh_label="INCORRECT",
        family="core_i9_apple_silicon_wrong_extra",
        group_id="core_i9_apple",
        exact_count=1,
    )
    rows = [
        _mvp_row(
            question=question,
            response="Core i9.",
            label="INCORRECT",
            sample_id="core_i9_row",
        ),
        _mvp_row(
            question=question,
            response="Core i9 Apple Silicon.",
            label="CORRECT",
            sample_id="apple_row",
        ),
        _mvp_row(
            question=question,
            response="Core i7.",
            label="CORRECT",
            sample_id="core_i7_row",
        ),
    ]

    corrections = build_row_corrections(rows, [core_i9, core_i9_apple])
    validation = propagation_validation(
        corrections=corrections,
        rules=[core_i9, core_i9_apple],
    )

    by_id = {row["sample_id"]: row for row in corrections}
    assert by_id["core_i9_row"]["corrected_diagnostic_open_label"] == "CORRECT"
    assert by_id["apple_row"]["corrected_diagnostic_open_label"] == "INCORRECT"
    assert by_id["core_i7_row"]["propagation_status"] == "blocked_core_i_family"
    assert by_id["core_i7_row"]["label_changed"] is False
    assert validation["core_i9_and_core_i9_apple_silicon_separated"] is True


def test_semantic_family_nonexact_row_is_blocked_not_corrected() -> None:
    question = "Amal Clooney is an internationally renowned?"
    rule = _rule(
        question=question,
        answer="lawyer",
        fresh_label="INCORRECT",
        family="amal_clooney_plain_lawyer",
    )
    rows = [
        _mvp_row(
            question=question,
            response="human rights lawyer",
            label="CORRECT",
        )
    ]

    corrections = build_row_corrections(rows, [rule])

    assert corrections[0]["propagation_status"] == "blocked_same_question_nonexact"
    assert corrections[0]["corrected_diagnostic_open_label"] == "CORRECT"
    assert corrections[0]["label_changed"] is False


def test_unresolved_exact_evidence_row_is_explicitly_not_applied() -> None:
    rule = _rule(
        question="What kind of dish is the French pithivier?",
        answer="a pastry",
        fresh_label="INCORRECT",
        local_status="unresolved_do_not_apply",
    )
    rows = [
        _mvp_row(
            question="What kind of dish is the French pithivier?",
            response="A pastry.",
            label="CORRECT",
        )
    ]

    corrections = build_row_corrections(rows, [rule])

    assert (
        corrections[0]["propagation_status"] == "evidence_nonpropagatable_not_applied"
    )
    assert corrections[0]["corrected_diagnostic_open_label"] == "CORRECT"
    assert corrections[0]["label_changed"] is False
