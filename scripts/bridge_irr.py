"""Bridge inter-rater reliability workflow helpers.

This module materializes blinded coding queues for the TriviaQA bridge
discordant cases and finalizes dual-rater annotations into a machine-readable
summary artifact.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score

from uncertainty import build_rate_summary, wilson_interval

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DEV_BASELINE = (
    ROOT / "data/gemma3_4b/intervention/triviaqa_bridge/dev_experiment/alpha_1.0.jsonl"
)
DEFAULT_DEV_COMPARISON = (
    ROOT
    / "data/gemma3_4b/intervention/triviaqa_bridge_iti_e0_paperfaithful_k12_first-3-tokens/experiment/alpha_8.0.jsonl"
)
DEFAULT_TEST_BASELINE = (
    ROOT / "data/gemma3_4b/intervention/triviaqa_bridge/test_experiment/alpha_1.0.jsonl"
)
DEFAULT_TEST_COMPARISON = (
    ROOT
    / "data/gemma3_4b/intervention/triviaqa_bridge_iti_e0_paperfaithful_k12_first-3-tokens/test_experiment/alpha_8.0.jsonl"
)
DEFAULT_OUTPUT_DIR = ROOT / "data/judge_validation/bridge_irr"

RUBRIC_VERSION = "bridge_incorrect_response_v1"
LABELS = (
    "wrong_entity_substitution",
    "evasion_or_factual_denial",
    "answer_dilution",
    "formal_refusal",
)
CONFIDENCE_LEVELS = ("low", "medium", "high")
TRIVIAQA_BRIDGE_FINAL_GRADES = {"CORRECT", "INCORRECT", "NOT_ATTEMPTED"}
TRIVIAQA_BRIDGE_AUDIT_TYPES = {"nonmatch", "match_audit"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def status_path(path: Path) -> str:
    resolved_path = path.resolve()
    resolved_root = ROOT.resolve()
    try:
        display_path = resolved_path.relative_to(resolved_root)
    except ValueError:
        display_path = resolved_path
    return str(display_path).replace("\\", "/")


def git_head_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def sha256_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_adjudication_rule_metadata(rule_path: Path) -> dict[str, str]:
    if not rule_path.exists():
        raise FileNotFoundError(f"Adjudication rule not found: {rule_path}")
    return {
        "path": status_path(rule_path),
        "git_commit": git_head_commit(),
        "content_sha256": sha256_digest(rule_path.read_text(encoding="utf-8")),
    }


def bridge_correct(row: dict[str, Any]) -> bool:
    compliance = row.get("compliance")
    if not isinstance(compliance, bool):
        row_id = row.get("id", "<unknown>")
        raise ValueError(
            "Bridge IRR requires judged bridge rows; "
            f"id={row_id!r} has non-boolean or missing compliance"
        )
    return compliance


def bridge_judgement_errors(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    compliance = row.get("compliance")
    if not isinstance(compliance, bool):
        errors.append("missing or non-boolean compliance")

    grade = row.get("triviaqa_bridge_grade")
    if grade is not None and grade not in TRIVIAQA_BRIDGE_FINAL_GRADES:
        errors.append(f"non-final triviaqa_bridge_grade={grade!r}")

    judge = row.get("judge")
    if judge is not None and judge not in TRIVIAQA_BRIDGE_FINAL_GRADES:
        errors.append(f"non-final judge={judge!r}")

    audit_type = row.get("judge_audit_type")
    if audit_type is not None and audit_type not in TRIVIAQA_BRIDGE_AUDIT_TYPES:
        errors.append(f"unknown judge_audit_type={audit_type!r}")

    has_final_grade = grade in TRIVIAQA_BRIDGE_FINAL_GRADES
    if has_final_grade and isinstance(compliance, bool):
        expected_compliance = grade == "CORRECT"
        if compliance != expected_compliance:
            errors.append(
                f"compliance does not match final triviaqa_bridge_grade {grade!r}"
            )
    elif row.get("deterministic_correct") is True:
        if compliance is False:
            errors.append(
                "unaudited deterministic match has compliance=false; "
                "expected final judge/audit fields for an overturned match"
            )
    else:
        errors.append("missing final triviaqa_bridge_grade")

    return errors


def validate_bridge_rows_judged(
    path: Path,
    rows_by_id: dict[str, dict[str, Any]],
) -> None:
    offenders: list[str] = []
    for row_id, row in sorted(rows_by_id.items()):
        errors = bridge_judgement_errors(row)
        if errors:
            offenders.append(f"{row_id}: {', '.join(errors)}")

    if not offenders:
        return

    preview = "; ".join(offenders[:5])
    if len(offenders) > 5:
        preview += f"; ... ({len(offenders)} total)"
    raise ValueError(
        "Bridge IRR requires judged bridge outputs before queue preparation. "
        f"{path}: offending ids: {preview}. Run "
        "scripts/evaluate_intervention.py --benchmark triviaqa_bridge and ensure "
        "the batch audit completed without ERROR records."
    )


def bridge_grade_label(row: dict[str, Any], *, default_if_incorrect: str) -> str:
    grade = row.get("triviaqa_bridge_grade")
    if isinstance(grade, str) and grade:
        return grade
    return "CORRECT" if bridge_correct(row) else default_if_incorrect


def load_bridge_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row["id"])
        if row_id in rows_by_id:
            raise ValueError(f"Duplicate bridge id in {path}: {row_id}")
        rows_by_id[row_id] = row
    validate_bridge_rows_judged(path, rows_by_id)
    return rows_by_id


def extract_discordant_cases(
    baseline_path: Path,
    comparison_path: Path,
    *,
    comparison_name: str,
) -> list[dict[str, Any]]:
    baseline_rows = load_bridge_rows(baseline_path)
    comparison_rows = load_bridge_rows(comparison_path)

    if set(baseline_rows) != set(comparison_rows):
        missing_in_comparison = sorted(set(baseline_rows) - set(comparison_rows))
        missing_in_baseline = sorted(set(comparison_rows) - set(baseline_rows))
        raise ValueError(
            "Bridge pair parity failed: "
            f"missing_in_comparison={missing_in_comparison[:5]} "
            f"missing_in_baseline={missing_in_baseline[:5]}"
        )

    cases: list[dict[str, Any]] = []
    for row_id in sorted(baseline_rows):
        baseline = baseline_rows[row_id]
        comparison = comparison_rows[row_id]
        baseline_correct = bridge_correct(baseline)
        comparison_correct = bridge_correct(comparison)
        if baseline_correct == comparison_correct:
            continue

        transition = "right_to_wrong" if baseline_correct else "wrong_to_right"
        incorrect_condition = "comparison" if baseline_correct else "baseline"
        incorrect_row = comparison if baseline_correct else baseline
        correct_row = baseline if baseline_correct else comparison

        cases.append(
            {
                "question_id": row_id,
                "question": str(baseline["question"]),
                "gold_aliases": list(baseline.get("ground_truth_aliases") or []),
                "transition": transition,
                "incorrect_condition": incorrect_condition,
                "correct_condition": (
                    "baseline"
                    if incorrect_condition == "comparison"
                    else comparison_name
                ),
                "baseline_response": str(baseline["response"]),
                "comparison_response": str(comparison["response"]),
                "incorrect_response": str(incorrect_row["response"]),
                "paired_correct_response": str(correct_row["response"]),
                "baseline_grade": bridge_grade_label(
                    baseline, default_if_incorrect="INCORRECT"
                ),
                "comparison_grade": bridge_grade_label(
                    comparison, default_if_incorrect="INCORRECT"
                ),
                "incorrect_grade": bridge_grade_label(
                    incorrect_row,
                    default_if_incorrect=(
                        "NOT_ATTEMPTED"
                        if not incorrect_row.get("attempted", True)
                        else "INCORRECT"
                    ),
                ),
                "paired_correct_grade": bridge_grade_label(
                    correct_row, default_if_incorrect="INCORRECT"
                ),
                "baseline_correct": baseline_correct,
                "comparison_correct": comparison_correct,
            }
        )
    return cases


def build_case_id(*, split: str, question_id: str) -> str:
    digest = hashlib.sha256(f"{split}:{question_id}".encode("utf-8")).hexdigest()[:12]
    return f"bridge_{split}_case_{digest}"


def build_blinded_case_artifacts(
    cases: list[dict[str, Any]],
    *,
    split: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = [dict(case) for case in cases]
    random.Random(seed).shuffle(shuffled)

    queue_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for case in shuffled:
        case_id = build_case_id(split=split, question_id=str(case["question_id"]))
        if case_id in seen_case_ids:
            raise ValueError(
                "Duplicate blinded case id generated for "
                f"split={split!r}, question_id={case['question_id']!r}"
            )
        seen_case_ids.add(case_id)
        queue_rows.append(
            {
                "case_id": case_id,
                "split": split,
                "rubric_version": RUBRIC_VERSION,
                "question": case["question"],
                "gold_aliases": case["gold_aliases"],
                "incorrect_response": case["incorrect_response"],
                "paired_correct_response": case["paired_correct_response"],
                "coding_target": "incorrect_response",
                "instructions": (
                    "Label the incorrect response only. Use the paired correct "
                    "response and gold aliases only to distinguish substitution "
                    "from dilution."
                ),
            }
        )
        key_rows.append(
            {
                "case_id": case_id,
                "question_id": case["question_id"],
                "transition": case["transition"],
                "incorrect_condition": case["incorrect_condition"],
                "correct_condition": case["correct_condition"],
                "baseline_response": case["baseline_response"],
                "comparison_response": case["comparison_response"],
                "incorrect_response": case["incorrect_response"],
                "paired_correct_response": case["paired_correct_response"],
                "baseline_grade": case["baseline_grade"],
                "comparison_grade": case["comparison_grade"],
                "incorrect_grade": case["incorrect_grade"],
                "paired_correct_grade": case["paired_correct_grade"],
            }
        )
    return queue_rows, key_rows


def case_key(row: dict[str, Any]) -> str:
    return str(row["case_id"])


def index_rows_by_case(
    rows: list[dict[str, Any]], *, row_kind: str
) -> dict[str, dict[str, Any]]:
    by_case: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = case_key(row)
        if key in by_case:
            raise ValueError(f"Duplicate {row_kind} row for {key}")
        by_case[key] = row
    return by_case


def _load_progress_with_validator(
    path: Path, validator: Any
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    by_case: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = case_key(row)
        if key in by_case:
            raise ValueError(f"Duplicate label row for {key} in {path}")
        validator(row)
        by_case[key] = row
    return by_case


def load_label_progress(path: Path) -> dict[str, dict[str, Any]]:
    return _load_progress_with_validator(path, validate_label_row)


def validate_label_row(row: dict[str, Any]) -> None:
    label = row.get("label")
    confidence = row.get("confidence")
    if label not in LABELS:
        raise ValueError(f"Invalid bridge label: {label!r}")
    if confidence not in CONFIDENCE_LEVELS:
        raise ValueError(f"Invalid bridge confidence: {confidence!r}")
    notes = row.get("notes", "")
    if not isinstance(notes, str):
        raise ValueError("Bridge notes must be a string")


def validate_adjudication_row(row: dict[str, Any]) -> None:
    label = row.get("label")
    notes = row.get("notes", "")
    rule_gap = row.get("rule_gap", False)
    if label not in LABELS:
        raise ValueError(f"Invalid adjudicated label: {label!r}")
    if not isinstance(notes, str):
        raise ValueError("Adjudication notes must be a string")
    if not isinstance(rule_gap, bool):
        raise ValueError("Adjudication rule_gap must be a boolean")


def load_adjudication_progress(path: Path) -> dict[str, dict[str, Any]]:
    return _load_progress_with_validator(path, validate_adjudication_row)


def ensure_progress_files_compatible(
    *,
    expected_case_ids: set[str],
    rater_a_path: Path,
    rater_b_path: Path,
    adjudication_path: Path,
) -> None:
    existing_progress = {
        "rater_a_progress": load_label_progress(rater_a_path),
        "rater_b_progress": load_label_progress(rater_b_path),
        "adjudication_progress": load_adjudication_progress(adjudication_path),
    }
    for ledger_name, rows_by_case in existing_progress.items():
        stale_case_ids = sorted(set(rows_by_case) - expected_case_ids)
        if stale_case_ids:
            raise ValueError(
                f"{ledger_name} contains case_ids that are not present in the "
                "regenerated queue. Refusing to reuse existing labels because "
                f"they may target a different question: {stale_case_ids[:5]}"
            )


def summarize_case_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    transitions = Counter(str(case["transition"]) for case in cases)
    return {
        "discordant_total": len(cases),
        "right_to_wrong": transitions.get("right_to_wrong", 0),
        "wrong_to_right": transitions.get("wrong_to_right", 0),
    }


def build_pending_status_payload(
    *,
    dev_cases: list[dict[str, Any]],
    test_cases: list[dict[str, Any]],
    output_dir: Path,
    adjudication_rule: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "awaiting_labels",
        "rubric_version": RUBRIC_VERSION,
        "label_set": list(LABELS),
        "adjudication_rule": adjudication_rule,
        "files": {
            "adjudication_rule": adjudication_rule["path"],
            "dev_calibration_queue": status_path(
                output_dir / "dev_calibration_queue.jsonl"
            ),
            "test_queue_blinded": status_path(output_dir / "test_queue_blinded.jsonl"),
            "test_queue_key": status_path(output_dir / "test_queue_key.jsonl"),
            "rater_a_progress": status_path(output_dir / "rater_a_progress.jsonl"),
            "rater_b_progress": status_path(output_dir / "rater_b_progress.jsonl"),
            "adjudication_progress": status_path(
                output_dir / "adjudication_progress.jsonl"
            ),
            "rater_b_provenance": status_path(output_dir / "rater_b_provenance.json"),
            "summary": status_path(output_dir / "bridge_irr_summary.json"),
        },
        "counts": {
            "dev": summarize_case_counts(dev_cases),
            "test": summarize_case_counts(test_cases),
        },
    }


def gwet_ac1(labels_a: list[str], labels_b: list[str]) -> float:
    if len(labels_a) != len(labels_b):
        raise ValueError("AC1 requires equal-length label sequences")
    n_items = len(labels_a)
    if n_items == 0:
        return 0.0

    if set(labels_a) == set(labels_b) and len(set(labels_a)) == 1:
        return 1.0

    category_count = len(LABELS)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n_items

    counts = Counter(labels_a)
    counts.update(labels_b)
    expected = 0.0
    for label in LABELS:
        p_label = counts.get(label, 0) / (2.0 * n_items)
        expected += p_label * (1.0 - p_label) / (category_count - 1.0)

    if abs(1.0 - expected) < 1e-12:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def confusion_matrix_counts(
    labels_a: list[str], labels_b: list[str]
) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {
        label_a: {label_b: 0 for label_b in LABELS} for label_a in LABELS
    }
    for label_a, label_b in zip(labels_a, labels_b):
        matrix[label_a][label_b] += 1
    return matrix


def distribution_summary(labels: list[str]) -> dict[str, int]:
    counts = Counter(labels)
    return {label: counts.get(label, 0) for label in LABELS}


def direction_category_summary(
    adjudicated_rows: list[dict[str, Any]], transition: str
) -> dict[str, Any]:
    subset = [row for row in adjudicated_rows if row["transition"] == transition]
    total = len(subset)
    categories: dict[str, Any] = {}
    for label in LABELS:
        count = sum(row["label"] == label for row in subset)
        rate = build_rate_summary(count, total)
        categories[label] = {
            "count": count,
            "denominator": total,
            "share": rate["estimate"],
            "share_pct": round(rate["estimate"] * 100.0, 1),
            "share_ci": rate["ci"],
            "share_ci_pct": {
                "lower": round(rate["ci"]["lower"] * 100.0, 1),
                "upper": round(rate["ci"]["upper"] * 100.0, 1),
                "level": rate["ci"]["level"],
                "method": rate["ci"]["method"],
            },
        }
    return {
        "n": total,
        "categories": categories,
    }


def _single_string_value(
    rows: dict[str, dict[str, Any]],
    *,
    field: str,
    field_name: str,
) -> str | None:
    values = {
        str(value).strip()
        for row in rows.values()
        if (value := row.get(field)) is not None and str(value).strip()
    }
    if not values:
        return None
    if len(values) != 1:
        raise ValueError(f"Expected one {field_name}, found {sorted(values)}")
    return next(iter(values))


def build_rater_b_summary_provenance(
    rater_b_rows: dict[str, dict[str, Any]],
    *,
    provenance_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_from_rows = _single_string_value(
        rater_b_rows,
        field="model",
        field_name="rater B model snapshot",
    )
    prompt_hash_from_rows = _single_string_value(
        rater_b_rows,
        field="prompt_hash",
        field_name="rater B prompt hash",
    )
    rubric_from_rows = _single_string_value(
        rater_b_rows,
        field="rubric_version",
        field_name="rater B rubric version",
    )

    if provenance_payload is None:
        model_snapshot = model_from_rows
        prompt_hash = prompt_hash_from_rows
        provenance_commit = None
        rubric_version = rubric_from_rows or RUBRIC_VERSION
    else:
        model_snapshot = (
            str(provenance_payload.get("model", "")).strip() or model_from_rows
        )
        prompt_hash = (
            str(provenance_payload.get("prompt_hash", "")).strip()
            or prompt_hash_from_rows
        )
        provenance_commit = (
            str(provenance_payload.get("git_commit", "")).strip() or None
        )
        rubric_version = (
            str(provenance_payload.get("rubric_version", "")).strip()
            or rubric_from_rows
            or RUBRIC_VERSION
        )

    if not model_snapshot:
        raise ValueError("Could not determine Rater B model snapshot")
    if not prompt_hash:
        raise ValueError("Could not determine Rater B prompt hash")
    if model_from_rows and model_snapshot != model_from_rows:
        raise ValueError("Rater B provenance model does not match progress rows")
    if prompt_hash_from_rows and prompt_hash != prompt_hash_from_rows:
        raise ValueError("Rater B provenance prompt hash does not match progress rows")
    if rubric_from_rows and rubric_version != rubric_from_rows:
        raise ValueError(
            "Rater B provenance rubric version does not match progress rows"
        )

    payload: dict[str, Any] = {
        "model_snapshot": model_snapshot,
        "prompt_hash": prompt_hash,
        "rubric_version": rubric_version,
    }
    if provenance_commit:
        payload["git_commit"] = provenance_commit
    return payload


def finalize_bridge_irr(
    *,
    queue_rows: list[dict[str, Any]],
    key_rows: list[dict[str, Any]],
    rater_a_rows: dict[str, dict[str, Any]],
    rater_b_rows: dict[str, dict[str, Any]],
    adjudication_rows: dict[str, dict[str, Any]],
    adjudication_rule: dict[str, Any],
    rater_b_provenance: dict[str, Any],
) -> dict[str, Any]:
    queue_by_case = index_rows_by_case(queue_rows, row_kind="queue")
    key_by_case = index_rows_by_case(key_rows, row_kind="key")
    expected_cases = set(queue_by_case)
    if expected_cases != set(key_by_case):
        raise ValueError("Queue/key case parity failed")
    if set(rater_a_rows) != expected_cases or set(rater_b_rows) != expected_cases:
        raise ValueError("Both raters must label every queued test case")

    disagreements = {
        case_id
        for case_id in expected_cases
        if rater_a_rows[case_id]["label"] != rater_b_rows[case_id]["label"]
    }
    if set(adjudication_rows) != disagreements:
        raise ValueError(
            "Adjudication ledger must contain exactly the disagreement cases"
        )

    sorted_case_ids = sorted(expected_cases)
    labels_a = [str(rater_a_rows[case_id]["label"]) for case_id in sorted_case_ids]
    labels_b = [str(rater_b_rows[case_id]["label"]) for case_id in sorted_case_ids]

    agreement_count = sum(a == b for a, b in zip(labels_a, labels_b))
    agreement_summary = build_rate_summary(agreement_count, len(sorted_case_ids))
    kappa = float(cohen_kappa_score(labels_a, labels_b, labels=list(LABELS)))
    ac1 = float(gwet_ac1(labels_a, labels_b))

    adjudicated_rows_out: list[dict[str, Any]] = []
    disagreements_out: list[dict[str, Any]] = []
    for case_id in sorted_case_ids:
        label_a = str(rater_a_rows[case_id]["label"])
        label_b = str(rater_b_rows[case_id]["label"])
        rule_gap = False
        if case_id in adjudication_rows:
            adjudicated = adjudication_rows[case_id]
            final_label = str(adjudicated["label"])
            source = "adjudication"
            rule_gap = bool(adjudicated.get("rule_gap", False))
            disagreements_out.append(
                {
                    "case_id": case_id,
                    "transition": key_by_case[case_id]["transition"],
                    "rater_a_label": label_a,
                    "rater_b_label": label_b,
                    "adjudicated_label": final_label,
                    "adjudication_notes": adjudicated.get("notes", ""),
                    "rule_gap": rule_gap,
                }
            )
        else:
            final_label = label_a
            source = "consensus"

        adjudicated_rows_out.append(
            {
                "case_id": case_id,
                "question_id": key_by_case[case_id]["question_id"],
                "transition": key_by_case[case_id]["transition"],
                "incorrect_condition": key_by_case[case_id]["incorrect_condition"],
                "correct_condition": key_by_case[case_id]["correct_condition"],
                "label": final_label,
                "label_source": source,
                "rule_gap": rule_gap,
                "rater_a_label": label_a,
                "rater_b_label": label_b,
                "rater_a_confidence": rater_a_rows[case_id]["confidence"],
                "rater_b_confidence": rater_b_rows[case_id]["confidence"],
                "rater_a_notes": rater_a_rows[case_id].get("notes", ""),
                "rater_b_notes": rater_b_rows[case_id].get("notes", ""),
            }
        )

    r2w_summary = direction_category_summary(adjudicated_rows_out, "right_to_wrong")
    w2r_summary = direction_category_summary(adjudicated_rows_out, "wrong_to_right")
    wrong_entity_r2w = r2w_summary["categories"]["wrong_entity_substitution"]
    rule_gap_count = sum(
        1 for row in adjudicated_rows_out if bool(row.get("rule_gap", False))
    )
    disagreement_count = len(disagreements_out)
    rule_gap_summary = build_rate_summary(rule_gap_count, disagreement_count)

    summary = {
        "schema_version": 1,
        "status": "adjudicated",
        "rubric_version": RUBRIC_VERSION,
        "label_set": list(LABELS),
        "provenance": {
            "adjudication_rule": adjudication_rule,
            "rater_b": rater_b_provenance,
        },
        "n_cases": len(sorted_case_ids),
        "irr": {
            "n_cases": len(sorted_case_ids),
            "raw_agreement": {
                "count": agreement_count,
                "n": len(sorted_case_ids),
                "estimate": agreement_summary["estimate"],
                "estimate_pct": round(agreement_summary["estimate"] * 100.0, 1),
                "ci": agreement_summary["ci"],
                "ci_pct": {
                    "lower": round(agreement_summary["ci"]["lower"] * 100.0, 1),
                    "upper": round(agreement_summary["ci"]["upper"] * 100.0, 1),
                    "level": agreement_summary["ci"]["level"],
                    "method": agreement_summary["ci"]["method"],
                },
            },
            "cohen_kappa": round(kappa, 4),
            "gwet_ac1": round(ac1, 4),
            "confusion_matrix": confusion_matrix_counts(labels_a, labels_b),
            "rater_a_distribution": distribution_summary(labels_a),
            "rater_b_distribution": distribution_summary(labels_b),
            "n_disagreements": len(disagreements_out),
        },
        "direction_summaries": {
            "right_to_wrong": r2w_summary,
            "wrong_to_right": w2r_summary,
        },
        "paper_claims": {
            "wrong_entity_substitution_r2w": wrong_entity_r2w,
        },
        "adjudication": {
            "n_consensus_cases": len(sorted_case_ids) - len(disagreements_out),
            "n_disagreements": len(disagreements_out),
            "rule_gap_cases": {
                "count": rule_gap_count,
                "n": disagreement_count,
                "estimate": rule_gap_summary["estimate"],
                "estimate_pct": round(rule_gap_summary["estimate"] * 100.0, 1),
                "ci": rule_gap_summary["ci"],
                "ci_pct": {
                    "lower": round(rule_gap_summary["ci"]["lower"] * 100.0, 1),
                    "upper": round(rule_gap_summary["ci"]["upper"] * 100.0, 1),
                    "level": rule_gap_summary["ci"]["level"],
                    "method": rule_gap_summary["ci"]["method"],
                },
            },
        },
        "adjudicated_rows": adjudicated_rows_out,
        "disagreements": disagreements_out,
    }
    return summary


def empty_progress_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")


def share_ci_pct(count: int, total: int) -> dict[str, Any]:
    interval = wilson_interval(count, total)
    return {
        "lower": round(interval.lower * 100.0, 1),
        "upper": round(interval.upper * 100.0, 1),
        "level": interval.level,
        "method": interval.method,
    }
