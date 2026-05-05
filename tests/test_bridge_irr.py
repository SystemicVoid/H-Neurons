import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bridge_irr import (  # noqa: E402
    build_blinded_case_artifacts,
    build_pending_status_payload,
    build_rater_b_summary_provenance,
    build_case_id,
    ensure_progress_files_compatible,
    extract_discordant_cases,
    finalize_bridge_irr,
    gwet_ac1,
    load_adjudication_progress,
    load_label_progress,
)
import prepare_bridge_irr_queue  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_extract_discordant_cases_identifies_damage_and_rescue(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.jsonl"
    comparison_path = tmp_path / "comparison.jsonl"
    _write_jsonl(
        baseline_path,
        [
            {
                "id": "q1",
                "question": "Question 1",
                "ground_truth_aliases": ["Alpha"],
                "response": "Alpha",
                "compliance": True,
                "attempted": True,
                "triviaqa_bridge_grade": "CORRECT",
            },
            {
                "id": "q2",
                "question": "Question 2",
                "ground_truth_aliases": ["Bravo"],
                "response": "No idea",
                "compliance": False,
                "attempted": False,
                "triviaqa_bridge_grade": "NOT_ATTEMPTED",
            },
            {
                "id": "q3",
                "question": "Question 3",
                "ground_truth_aliases": ["Charlie"],
                "response": "Wrong",
                "compliance": False,
                "attempted": True,
                "triviaqa_bridge_grade": "INCORRECT",
            },
        ],
    )
    _write_jsonl(
        comparison_path,
        [
            {
                "id": "q1",
                "question": "Question 1",
                "ground_truth_aliases": ["Alpha"],
                "response": "Nearby wrong entity",
                "compliance": False,
                "attempted": True,
                "triviaqa_bridge_grade": "INCORRECT",
            },
            {
                "id": "q2",
                "question": "Question 2",
                "ground_truth_aliases": ["Bravo"],
                "response": "Bravo",
                "compliance": True,
                "attempted": True,
                "triviaqa_bridge_grade": "CORRECT",
            },
            {
                "id": "q3",
                "question": "Question 3",
                "ground_truth_aliases": ["Charlie"],
                "response": "Still wrong",
                "compliance": False,
                "attempted": True,
                "triviaqa_bridge_grade": "INCORRECT",
            },
        ],
    )

    cases = extract_discordant_cases(
        baseline_path,
        comparison_path,
        comparison_name="comparison",
    )

    assert len(cases) == 2
    assert {case["transition"] for case in cases} == {
        "right_to_wrong",
        "wrong_to_right",
    }

    case_by_id = {case["question_id"]: case for case in cases}
    assert case_by_id["q1"]["incorrect_condition"] == "comparison"
    assert case_by_id["q1"]["incorrect_response"] == "Nearby wrong entity"
    assert case_by_id["q2"]["incorrect_condition"] == "baseline"
    assert case_by_id["q2"]["incorrect_grade"] == "NOT_ATTEMPTED"


def test_build_blinded_case_artifacts_produces_queue_and_key() -> None:
    cases = [
        {
            "question_id": "q1",
            "question": "Question 1",
            "gold_aliases": ["Alpha"],
            "transition": "right_to_wrong",
            "incorrect_condition": "comparison",
            "correct_condition": "baseline",
            "baseline_response": "Alpha",
            "comparison_response": "Nearby wrong entity",
            "incorrect_response": "Nearby wrong entity",
            "paired_correct_response": "Alpha",
            "baseline_grade": "CORRECT",
            "comparison_grade": "INCORRECT",
            "incorrect_grade": "INCORRECT",
            "paired_correct_grade": "CORRECT",
        }
    ]

    queue_rows, key_rows = build_blinded_case_artifacts(cases, split="test", seed=42)

    assert len(queue_rows) == 1
    assert len(key_rows) == 1
    assert queue_rows[0]["case_id"] == key_rows[0]["case_id"]
    assert queue_rows[0]["incorrect_response"] == "Nearby wrong entity"
    assert "baseline_response" not in queue_rows[0]
    assert key_rows[0]["transition"] == "right_to_wrong"


def test_build_blinded_case_artifacts_keeps_case_ids_stable_across_seeds() -> None:
    cases = [
        {
            "question_id": "q1",
            "question": "Question 1",
            "gold_aliases": ["Alpha"],
            "transition": "right_to_wrong",
            "incorrect_condition": "comparison",
            "correct_condition": "baseline",
            "baseline_response": "Alpha",
            "comparison_response": "Nearby wrong entity",
            "incorrect_response": "Nearby wrong entity",
            "paired_correct_response": "Alpha",
            "baseline_grade": "CORRECT",
            "comparison_grade": "INCORRECT",
            "incorrect_grade": "INCORRECT",
            "paired_correct_grade": "CORRECT",
        },
        {
            "question_id": "q2",
            "question": "Question 2",
            "gold_aliases": ["Bravo"],
            "transition": "wrong_to_right",
            "incorrect_condition": "baseline",
            "correct_condition": "comparison",
            "baseline_response": "Wrong",
            "comparison_response": "Bravo",
            "incorrect_response": "Wrong",
            "paired_correct_response": "Bravo",
            "baseline_grade": "INCORRECT",
            "comparison_grade": "CORRECT",
            "incorrect_grade": "INCORRECT",
            "paired_correct_grade": "CORRECT",
        },
    ]

    _, key_rows_seed_1 = build_blinded_case_artifacts(cases, split="test", seed=1)
    _, key_rows_seed_2 = build_blinded_case_artifacts(cases, split="test", seed=2)

    case_ids_seed_1 = {row["question_id"]: row["case_id"] for row in key_rows_seed_1}
    case_ids_seed_2 = {row["question_id"]: row["case_id"] for row in key_rows_seed_2}

    assert case_ids_seed_1 == case_ids_seed_2
    assert case_ids_seed_1["q1"] == build_case_id(split="test", question_id="q1")


def test_ensure_progress_files_compatible_rejects_stale_case_ids(
    tmp_path: Path,
) -> None:
    expected_case_ids = {
        build_case_id(split="test", question_id="q1"),
        build_case_id(split="test", question_id="q3"),
    }
    stale_case_id = build_case_id(split="test", question_id="q2")

    rater_a_path = tmp_path / "rater_a_progress.jsonl"
    rater_b_path = tmp_path / "rater_b_progress.jsonl"
    adjudication_path = tmp_path / "adjudication_progress.jsonl"
    _write_jsonl(
        rater_a_path,
        [
            {
                "case_id": stale_case_id,
                "label": "formal_refusal",
                "confidence": "medium",
                "notes": "",
            }
        ],
    )

    with pytest.raises(ValueError, match="different question"):
        ensure_progress_files_compatible(
            expected_case_ids=expected_case_ids,
            rater_a_path=rater_a_path,
            rater_b_path=rater_b_path,
            adjudication_path=adjudication_path,
        )


def test_build_pending_status_payload_keeps_external_paths_absolute(
    tmp_path: Path,
) -> None:
    rule_path = tmp_path / "adjudication_rule.md"
    rule_path.write_text("# frozen\n", encoding="utf-8")
    status = build_pending_status_payload(
        dev_cases=[],
        test_cases=[],
        output_dir=tmp_path,
        adjudication_rule={
            "path": str(rule_path.resolve()).replace("\\", "/"),
            "git_commit": "abc123",
            "content_sha256": "deadbeef",
        },
    )

    assert status["files"]["summary"] == str(
        (tmp_path / "bridge_irr_summary.json").resolve()
    ).replace("\\", "/")
    assert status["adjudication_rule"]["git_commit"] == "abc123"
    assert status["files"]["rater_b_provenance"] == str(
        (tmp_path / "rater_b_provenance.json").resolve()
    ).replace("\\", "/")


def test_prepare_bridge_irr_queue_can_skip_dev_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_path = tmp_path / "baseline.jsonl"
    comparison_path = tmp_path / "comparison.jsonl"
    rule_path = tmp_path / "adjudication_rule.md"
    output_dir = tmp_path / "irr"
    rule_path.write_text("# frozen\n", encoding="utf-8")
    _write_jsonl(
        baseline_path,
        [
            {
                "id": "q1",
                "question": "Question 1",
                "ground_truth_aliases": ["Alpha"],
                "response": "Alpha",
                "compliance": True,
                "attempted": True,
                "triviaqa_bridge_grade": "CORRECT",
            }
        ],
    )
    _write_jsonl(
        comparison_path,
        [
            {
                "id": "q1",
                "question": "Question 1",
                "ground_truth_aliases": ["Alpha"],
                "response": "Wrong entity",
                "compliance": False,
                "attempted": True,
                "triviaqa_bridge_grade": "INCORRECT",
            }
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_bridge_irr_queue.py",
            "--skip_dev",
            "--test_baseline",
            str(baseline_path),
            "--test_comparison",
            str(comparison_path),
            "--output_dir",
            str(output_dir),
            "--adjudication_rule_path",
            str(rule_path),
        ],
    )

    prepare_bridge_irr_queue.main()

    status = json.loads((output_dir / "bridge_irr_status.json").read_text())
    assert status["counts"]["dev"]["discordant_total"] == 0
    assert status["counts"]["test"]["discordant_total"] == 1
    assert (output_dir / "dev_calibration_queue.jsonl").read_text() == ""


def test_gwet_ac1_is_one_for_perfect_agreement() -> None:
    labels = [
        "wrong_entity_substitution",
        "evasion_or_factual_denial",
        "answer_dilution",
        "formal_refusal",
    ]
    assert gwet_ac1(labels, labels) == pytest.approx(1.0)


def test_build_rater_b_summary_provenance_falls_back_to_progress_rows() -> None:
    rows = {
        "bridge_test_case_001": {
            "case_id": "bridge_test_case_001",
            "label": "wrong_entity_substitution",
            "confidence": "high",
            "notes": "",
            "model": "gpt-4o-2024-11-20",
            "prompt_hash": "prompt123",
            "rubric_version": "bridge_incorrect_response_v1",
        }
    }

    provenance = build_rater_b_summary_provenance(rows)

    assert provenance == {
        "model_snapshot": "gpt-4o-2024-11-20",
        "prompt_hash": "prompt123",
        "rubric_version": "bridge_incorrect_response_v1",
    }


def test_finalize_bridge_irr_computes_summary(tmp_path: Path) -> None:
    queue_rows = [
        {"case_id": "bridge_test_case_001"},
        {"case_id": "bridge_test_case_002"},
        {"case_id": "bridge_test_case_003"},
    ]
    key_rows = [
        {
            "case_id": "bridge_test_case_001",
            "question_id": "q1",
            "transition": "right_to_wrong",
            "incorrect_condition": "comparison",
            "correct_condition": "baseline",
        },
        {
            "case_id": "bridge_test_case_002",
            "question_id": "q2",
            "transition": "right_to_wrong",
            "incorrect_condition": "comparison",
            "correct_condition": "baseline",
        },
        {
            "case_id": "bridge_test_case_003",
            "question_id": "q3",
            "transition": "wrong_to_right",
            "incorrect_condition": "baseline",
            "correct_condition": "comparison",
        },
    ]

    rater_a_path = tmp_path / "rater_a.jsonl"
    rater_b_path = tmp_path / "rater_b.jsonl"
    adjudication_path = tmp_path / "adjudication.jsonl"
    _write_jsonl(
        rater_a_path,
        [
            {
                "case_id": "bridge_test_case_001",
                "label": "wrong_entity_substitution",
                "confidence": "high",
                "notes": "",
            },
            {
                "case_id": "bridge_test_case_002",
                "label": "formal_refusal",
                "confidence": "medium",
                "notes": "",
            },
            {
                "case_id": "bridge_test_case_003",
                "label": "wrong_entity_substitution",
                "confidence": "medium",
                "notes": "",
            },
        ],
    )
    _write_jsonl(
        rater_b_path,
        [
            {
                "case_id": "bridge_test_case_001",
                "label": "wrong_entity_substitution",
                "confidence": "high",
                "notes": "",
            },
            {
                "case_id": "bridge_test_case_002",
                "label": "evasion_or_factual_denial",
                "confidence": "medium",
                "notes": "",
            },
            {
                "case_id": "bridge_test_case_003",
                "label": "wrong_entity_substitution",
                "confidence": "low",
                "notes": "",
            },
        ],
    )
    _write_jsonl(
        adjudication_path,
        [
            {
                "case_id": "bridge_test_case_002",
                "label": "evasion_or_factual_denial",
                "notes": "Settled after review.",
                "rule_gap": True,
            }
        ],
    )

    summary = finalize_bridge_irr(
        queue_rows=queue_rows,
        key_rows=key_rows,
        rater_a_rows=load_label_progress(rater_a_path),
        rater_b_rows=load_label_progress(rater_b_path),
        adjudication_rows=load_adjudication_progress(adjudication_path),
        adjudication_rule={
            "path": "data/judge_validation/bridge_irr/adjudication_rule.md",
            "git_commit": "freeze123",
            "content_sha256": "rulehash456",
        },
        rater_b_provenance={
            "model_snapshot": "gpt-4o-2024-11-20",
            "prompt_hash": "prompt789",
            "rubric_version": "bridge_incorrect_response_v1",
            "git_commit": "prov321",
        },
    )

    assert summary["status"] == "adjudicated"
    assert summary["irr"]["n_cases"] == 3
    assert summary["irr"]["n_disagreements"] == 1
    assert summary["adjudication"]["n_disagreements"] == 1
    assert summary["adjudication"]["rule_gap_cases"]["count"] == 1
    assert summary["adjudicated_rows"][1]["rule_gap"] is True
    assert summary["disagreements"][0]["rule_gap"] is True
    assert summary["provenance"]["adjudication_rule"]["git_commit"] == "freeze123"
    assert summary["provenance"]["rater_b"]["model_snapshot"] == "gpt-4o-2024-11-20"
    assert summary["provenance"]["rater_b"]["prompt_hash"] == "prompt789"
    assert summary["paper_claims"]["wrong_entity_substitution_r2w"]["count"] == 1
    assert (
        summary["direction_summaries"]["right_to_wrong"]["categories"][
            "evasion_or_factual_denial"
        ]["count"]
        == 1
    )
    assert (
        summary["direction_summaries"]["wrong_to_right"]["categories"][
            "wrong_entity_substitution"
        ]["count"]
        == 1
    )
