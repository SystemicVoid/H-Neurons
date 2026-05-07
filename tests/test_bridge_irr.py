import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
KAPPA = "\u03ba"
RIGHT_ARROW = "\u2192"
EM_DASH = "\u2014"

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


def _markdown_rows_by_first_cell(path: Path) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or cells[0] in {"---", "Claim", "Category", "Metric"}:
            continue
        rows[cells[0]] = cells
    return rows


def _extract_main_bridge_claims() -> dict[str, str]:
    text = (ROOT / "paper/icml/main.tex").read_text(encoding="utf-8")
    patterns = {
        "raw_agreement": r"Raw agreement was 96\.5\\%",
        "cohen_kappa": r"Cohen's \$\\kappa = 0\.90\$",
        "gwet_ac1": r"Gwet's AC1 = 0\.96",
        "wrong_entity_substitution": r"31/43 = 72\\%",
        "evasion_or_factual_denial": r"evasion/factual denial \(9/43, 21\\%\)",
        "answer_dilution": r"answer dilution \(3/43, 7\\%\)",
        "formal_refusal": r"formal refusal \(0/43\)",
    }
    return {
        name: pattern for name, pattern in patterns.items() if re.search(pattern, text)
    }


def test_paper_bridge_taxonomy_provenance_matches_irr_summary() -> None:
    summary = json.loads(
        (ROOT / "data/judge_validation/bridge_irr/bridge_irr_summary.json").read_text(
            encoding="utf-8"
        )
    )
    categories = summary["direction_summaries"]["right_to_wrong"]["categories"]
    expected_categories = {
        "wrong_entity_substitution": {
            "count": 31,
            "denominator": 43,
            "share": "72.1%",
            "ci": "[57.3, 83.3]",
        },
        "evasion_or_factual_denial": {
            "count": 9,
            "denominator": 43,
            "share": "20.9%",
            "ci": "[11.4, 35.2]",
        },
        "answer_dilution": {
            "count": 3,
            "denominator": 43,
            "share": "7.0%",
            "ci": "[2.4, 18.6]",
        },
        "formal_refusal": {
            "count": 0,
            "denominator": 43,
            "share": "0.0%",
            "ci": "[0.0, 8.2]",
        },
    }
    for key, expected in expected_categories.items():
        source = categories[key]
        assert source["count"] == expected["count"]
        assert source["denominator"] == expected["denominator"]
        assert f"{source['share_pct']:.1f}%" == expected["share"]
        assert (
            f"[{source['share_ci_pct']['lower']:.1f}, "
            f"{source['share_ci_pct']['upper']:.1f}]" == expected["ci"]
        )

    irr = summary["irr"]
    assert irr["raw_agreement"]["count"] == 55
    assert irr["raw_agreement"]["n"] == 57
    assert f"{irr['raw_agreement']['estimate_pct']:.1f}%" == "96.5%"
    assert round(irr["cohen_kappa"], 2) == 0.90
    assert round(irr["gwet_ac1"], 2) == 0.96

    paper_rows = _markdown_rows_by_first_cell(ROOT / "paper/icml/number_provenance.md")
    supplement_rows = _markdown_rows_by_first_cell(
        ROOT / "paper/icml/supplement/number_provenance.md"
    )
    support_rows = _markdown_rows_by_first_cell(
        ROOT / "paper/icml/supplement/support/externality_summary.md"
    )
    manifest_rows = _markdown_rows_by_first_cell(
        ROOT / "paper/icml/supplement/failure_coding_manifest.md"
    )

    paper_expected_rows = {
        f"Wrong-entity substitution (adjudicated, R{RIGHT_ARROW}W)": (
            "31 / 43 (72.1%)",
            "[57.3, 83.3] (Wilson)",
        ),
        f"Evasion / factual denial (adjudicated, R{RIGHT_ARROW}W)": (
            "9 / 43 (20.9%)",
            "[11.4, 35.2] (Wilson)",
        ),
        f"Answer dilution / verbosity (adjudicated, R{RIGHT_ARROW}W)": (
            "3 / 43 (7.0%)",
            "[2.4, 18.6] (Wilson)",
        ),
        f"Formal refusal among right{RIGHT_ARROW}wrong flips (adjudicated)": (
            "0 / 43 (0.0%)",
            "[0.0, 8.2] (Wilson)",
        ),
        "Dual-rater raw agreement (all 57 discordant)": (
            "55 / 57 (96.5%)",
            "[88.1, 99.0] (Wilson)",
        ),
        f"Cohen's {KAPPA} (A vs B, 4-category)": ("0.90", EM_DASH),
        "Gwet's AC1 (A vs B, 4-category)": ("0.96", EM_DASH),
    }
    for claim, (value, interval) in paper_expected_rows.items():
        assert paper_rows[claim][1:3] == [value, interval]

    supplement_expected_rows = {
        "Wrong-entity substitution": ("31 / 43 (72.1%)", "[57.3, 83.3] (Wilson)"),
        "Evasion / factual denial": ("9 / 43 (20.9%)", "[11.4, 35.2] (Wilson)"),
        "Answer dilution / verbosity": ("3 / 43 (7.0%)", "[2.4, 18.6] (Wilson)"),
        f"Formal refusal among right{RIGHT_ARROW}wrong flips": (
            "0 / 43 (0.0%)",
            "[0.0, 8.2] (Wilson)",
        ),
        "Dual-rater raw agreement (all 57 discordant)": (
            "55 / 57 (96.5%)",
            "[88.1, 99.0] (Wilson)",
        ),
        f"Cohen's {KAPPA} (A vs B, 4-category)": ("0.90", EM_DASH),
        "Gwet's AC1 (A vs B, 4-category)": ("0.96", EM_DASH),
    }
    for claim, (value, interval) in supplement_expected_rows.items():
        assert supplement_rows[claim][1:3] == [value, interval]

    support_expected_rows = {
        "Wrong-entity substitution": ("31", "72.1%", "[57.3, 83.3]"),
        "Evasion / factual denial": ("9", "20.9%", "[11.4, 35.2]"),
        "Answer dilution / verbosity": ("3", "7.0%", "[2.4, 18.6]"),
        "Formal refusal": ("0", "0.0%", "[0.0, 8.2]"),
    }
    for category, expected_cells in support_expected_rows.items():
        assert support_rows[category][1:4] == list(expected_cells)
        assert manifest_rows[category][2:5] == list(expected_cells)

    assert _extract_main_bridge_claims().keys() == {
        "raw_agreement",
        "cohen_kappa",
        "gwet_ac1",
        "wrong_entity_substitution",
        "evasion_or_factual_denial",
        "answer_dilution",
        "formal_refusal",
    }


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


def test_extract_discordant_cases_rejects_unjudged_generation_rows(
    tmp_path: Path,
) -> None:
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
                "deterministic_correct": True,
                "match_tier": "exact",
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
                "deterministic_correct": False,
                "match_tier": "no_match",
            }
        ],
    )

    with pytest.raises(ValueError, match="requires judged bridge outputs") as exc_info:
        extract_discordant_cases(
            baseline_path,
            comparison_path,
            comparison_name="comparison",
        )

    message = str(exc_info.value)
    assert str(baseline_path) in message
    assert "q1" in message
    assert "compliance" in message


def test_prepare_bridge_irr_queue_rejects_partial_judge_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
                "deterministic_correct": True,
                "match_tier": "exact",
                "compliance": True,
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
                "deterministic_correct": False,
                "match_tier": "no_match",
                "compliance": False,
                "judge": "ERROR",
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

    with pytest.raises(ValueError, match="requires judged bridge outputs") as exc_info:
        prepare_bridge_irr_queue.main()

    message = str(exc_info.value)
    assert str(comparison_path) in message
    assert "q1" in message
    assert "ERROR" in message


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
