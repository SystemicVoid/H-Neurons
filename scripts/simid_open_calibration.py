from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score

from artifact_io import file_sha256
from bridge_irr import build_adjudication_rule_metadata
from uncertainty import build_rate_summary


SIMID_OPEN_FINAL_GRADE_LABELS = ("CORRECT", "INCORRECT", "NOT_ATTEMPTED")
SIMID_OPEN_FINAL_GRADES = frozenset(SIMID_OPEN_FINAL_GRADE_LABELS)
SIMID_OPEN_CALIBRATION_QUEUE_SCHEMA_VERSION = "simid_open_calibration_queue/v1"
SECONDARY_SCHEMA_VERSION = "simid_open_calibration_secondary_label/v1"
ADJUDICATION_SCHEMA_VERSION = "simid_open_calibration_adjudication/v1"
PROMPT_VERSION = "simid_open_calibration_rule/v1"

OPEN_CORRECTNESS_CLAIM_CONTRACT = (
    "diagnostic_only_until_judge_calibration_evidence_recorded"
)
OPEN_CORRECTNESS_CALIBRATED_CLAIM_CONTRACT = (
    "claimable_with_recorded_judge_calibration_evidence"
)
OPEN_CORRECTNESS_PROSPECTIVE_AUTHORITY_CLAIM_CONTRACT = (
    "claimable_with_recorded_prospective_open_authority_and_external_labels"
)
OPEN_CORRECTNESS_JUDGE_BLOCKER = "calibration_evidence_not_recorded"
OPEN_CORRECTNESS_PROSPECTIVE_AUTHORITY_BLOCKER = (
    "prospective_open_authority_not_recorded"
)
OPEN_CORRECTNESS_PROSPECTIVE_LABELS_BLOCKER = (
    "prospective_open_external_labels_not_loaded"
)
OPEN_CORRECTNESS_PROSPECTIVE_LABELS_WITHOUT_AUTHORITY_BLOCKER = (
    "prospective_open_external_labels_without_authority"
)
OPEN_CORRECTNESS_NO_JUDGE_BLOCKER = "judge_adjudication_not_loaded"
OPEN_CORRECTNESS_MIXED_SOURCE_BLOCKER = "mixed_adjudication_and_deterministic_sources"
OPEN_CORRECTNESS_INVALID_JUDGE_BLOCKER = "judge_verdict_unknown_or_error"
OPEN_CORRECTNESS_CALIBRATION_SCHEMA_VERSION = "simid_open_calibration/v1"
OPEN_CORRECTNESS_CALIBRATION_TARGET = "simid_open_correctness"
OPEN_CORRECTNESS_CALIBRATION_INVALID_BLOCKER = "calibration_evidence_invalid"
OPEN_CORRECTNESS_CALIBRATION_FAILED_BLOCKER = "calibration_evidence_failed_thresholds"
OPEN_CORRECTNESS_CALIBRATION_RULE_GAP_BLOCKER = "calibration_rule_gap_recorded"
OPEN_CORRECTNESS_CALIBRATION_N_CASES_BLOCKER = "n_cases_below_minimum"
OPEN_CORRECTNESS_CALIBRATION_MIN_CASES = 100
OPEN_CORRECTNESS_CALIBRATION_MIN_KAPPA = 0.80
OPEN_CORRECTNESS_CALIBRATION_MIN_AC1 = 0.80


class CalibrationQueueLaunchError(ValueError):
    """Raised when the calibration queue fails pre-launch sanity checks."""


def load_jsonl(path: Path, *, required: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_no}")
            rows.append(payload)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def case_ids_sha256(case_ids: list[str]) -> str:
    payload = json.dumps(sorted(case_ids), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def queue_row_sha256(row: dict[str, Any]) -> str:
    payload = json.dumps(
        row,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def queue_rows_sha256(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        rows,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def input_file_metadata(path: Path) -> dict[str, Any]:
    return {"path": str(path), "content_sha256": file_sha256(path)}


def deterministic_open_correct(row: dict[str, Any]) -> bool:
    return bool(row["open_grade"]["correct"])


def deterministic_open_attempted(row: dict[str, Any]) -> bool:
    return bool(row["open_grade"]["attempted"])


def adjudication_verdict(adjudication: dict[str, Any] | None) -> str | None:
    if not adjudication:
        return None
    raw = adjudication.get("judge_grade")
    if raw is None:
        parse_result = adjudication.get("parse_result")
        if isinstance(parse_result, dict):
            raw = parse_result.get("verdict")
    if raw is None:
        return None
    return str(raw).strip().upper()


def effective_open_grade(row: dict[str, Any]) -> dict[str, Any]:
    """Return the analysis-grade view of SIMID open correctness."""
    external_label = row.get("prospective_open_label")
    if isinstance(external_label, dict):
        verdict = str(external_label.get("label") or "").strip().upper()
        if verdict in SIMID_OPEN_FINAL_GRADES:
            return {
                "correct": verdict == "CORRECT",
                "attempted": verdict in {"CORRECT", "INCORRECT"},
                "source": "external_blind_label",
                "judge_grade": verdict,
                "parse_valid": True,
                "claimable": True,
                "claimability_blocker": None,
            }
        return {
            "correct": False,
            "attempted": False,
            "source": "external_blind_label_invalid",
            "judge_grade": verdict or None,
            "parse_valid": False,
            "claimable": False,
            "claimability_blocker": OPEN_CORRECTNESS_INVALID_JUDGE_BLOCKER,
        }
    adjudication = row.get("open_adjudication")
    verdict = adjudication_verdict(
        adjudication if isinstance(adjudication, dict) else None
    )
    if verdict is None:
        if isinstance(adjudication, dict):
            return {
                "correct": False,
                "attempted": False,
                "source": "adjudication_unknown",
                "judge_grade": None,
                "parse_valid": False,
                "claimable": False,
                "claimability_blocker": OPEN_CORRECTNESS_INVALID_JUDGE_BLOCKER,
            }
        return {
            "correct": deterministic_open_correct(row),
            "attempted": deterministic_open_attempted(row),
            "source": "deterministic_alias",
            "judge_grade": None,
            "parse_valid": False,
            "claimable": False,
            "claimability_blocker": OPEN_CORRECTNESS_NO_JUDGE_BLOCKER,
        }
    if verdict in SIMID_OPEN_FINAL_GRADES:
        return {
            "correct": verdict == "CORRECT",
            "attempted": verdict in {"CORRECT", "INCORRECT"},
            "source": "adjudication",
            "judge_grade": verdict,
            "parse_valid": True,
            "claimable": False,
            "claimability_blocker": OPEN_CORRECTNESS_JUDGE_BLOCKER,
        }
    return {
        "correct": False,
        "attempted": False,
        "source": "adjudication_unknown",
        "judge_grade": verdict,
        "parse_valid": False,
        "claimable": False,
        "claimability_blocker": OPEN_CORRECTNESS_INVALID_JUDGE_BLOCKER,
    }


def effective_open_correct(row: dict[str, Any]) -> bool:
    return bool(effective_open_grade(row)["correct"])


def effective_open_attempted(row: dict[str, Any]) -> bool:
    return bool(effective_open_grade(row)["attempted"])


def row_case_id(row: dict[str, Any]) -> str:
    case_id = row.get("calibration_case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"Missing calibration_case_id in row: {row!r}")
    return case_id


def index_by_case(
    rows: list[dict[str, Any]], *, row_kind: str
) -> dict[str, dict[str, Any]]:
    by_case: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = row_case_id(row)
        if case_id in by_case:
            raise ValueError(f"Duplicate {row_kind} row for {case_id}")
        by_case[case_id] = row
    return by_case


def grade_from_row(row: dict[str, Any], *, row_kind: str) -> str:
    raw = (
        row.get("judge_grade")
        or row.get("primary_judge_grade")
        or row.get("label")
        or row.get("verdict")
    )
    if raw is None and isinstance(row.get("primary_effective_open_grade"), dict):
        raw = row["primary_effective_open_grade"].get("judge_grade")
    grade = str(raw).strip().upper() if raw is not None else ""
    if grade not in SIMID_OPEN_FINAL_GRADES:
        raise ValueError(f"Invalid {row_kind} SIMID open grade: {raw!r}")
    return grade


def nested_str(row: dict[str, Any], *path: str) -> str | None:
    current: Any = row
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if current is None:
        return None
    return str(current)


def validate_row_queue_hash(
    row: dict[str, Any],
    *,
    row_kind: str,
    case_id: str,
    queue_row: dict[str, Any],
) -> None:
    expected_hash = row.get("queue_row_sha256")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError(
            f"{row_kind} row for {case_id} is missing queue_row_sha256; cannot "
            "verify the current calibration queue row"
        )
    observed_hash = queue_row_sha256(queue_row)
    if expected_hash != observed_hash:
        raise ValueError(
            f"{row_kind} row for {case_id} was produced for a different "
            "calibration queue row"
        )


def validate_row_rule_hash(
    row: dict[str, Any],
    *,
    row_kind: str,
    case_id: str,
    expected_rule_hash: str,
) -> None:
    if nested_str(row, "rule", "content_sha256") != expected_rule_hash:
        raise ValueError(
            f"{row_kind} row for {case_id} does not match the current frozen rule hash"
        )


def validate_row_output_metadata(
    row: dict[str, Any],
    *,
    row_kind: str,
    case_id: str,
    expected_schema_version: str,
    actor_key: str,
    expected_model: str,
) -> None:
    observed_schema_version = row.get("schema_version")
    if observed_schema_version != expected_schema_version:
        raise ValueError(
            f"{row_kind} row for {case_id} has schema_version "
            f"{observed_schema_version!r}, expected {expected_schema_version!r}"
        )
    actor_type = nested_str(row, actor_key, "type")
    if actor_type != "llm":
        raise ValueError(
            f"{row_kind} row for {case_id} has {actor_key}.type "
            f"{actor_type!r}, expected 'llm'"
        )
    observed_model = nested_str(row, actor_key, "model")
    if observed_model != expected_model:
        raise ValueError(
            f"{row_kind} row for {case_id} was produced by {observed_model!r}, "
            f"expected {expected_model!r}"
        )
    observed_prompt_version = nested_str(row, actor_key, "prompt_version")
    if observed_prompt_version != PROMPT_VERSION:
        raise ValueError(
            f"{row_kind} row for {case_id} has prompt_version "
            f"{observed_prompt_version!r}, expected {PROMPT_VERSION!r}"
        )


def validate_secondary_queue_context(
    secondary_by_case: dict[str, dict[str, Any]],
    *,
    queue_by_case: dict[str, dict[str, Any]],
) -> None:
    for case_id, row in secondary_by_case.items():
        if case_id not in queue_by_case:
            raise ValueError(
                f"Existing secondary-rater row for {case_id} is not present in "
                "the current calibration queue"
            )
        validate_row_queue_hash(
            row,
            row_kind="Existing secondary-rater",
            case_id=case_id,
            queue_row=queue_by_case[case_id],
        )


def validate_existing_adjudication_context(
    existing: dict[str, dict[str, Any]],
    *,
    queue_by_case: dict[str, dict[str, Any]],
    secondary_labels: dict[str, str],
) -> None:
    for case_id, row in existing.items():
        queue_row = queue_by_case[case_id]
        validate_row_queue_hash(
            row,
            row_kind="Existing adjudication",
            case_id=case_id,
            queue_row=queue_row,
        )
        expected_primary = grade_from_row(queue_row, row_kind="primary queue")
        expected_secondary = secondary_labels[case_id]
        if row.get("primary_grade") != expected_primary:
            raise ValueError(
                f"Existing adjudication row for {case_id} has primary_grade "
                f"{row.get('primary_grade')!r}, expected {expected_primary!r}"
            )
        if row.get("secondary_grade") != expected_secondary:
            raise ValueError(
                f"Existing adjudication row for {case_id} has secondary_grade "
                f"{row.get('secondary_grade')!r}, expected {expected_secondary!r}"
            )


def validate_secondary_evidence_context(
    secondary_by_case: dict[str, dict[str, Any]],
    *,
    queue_by_case: dict[str, dict[str, Any]],
    expected_rule_hash: str,
    expected_model: str,
) -> None:
    for case_id, row in secondary_by_case.items():
        validate_row_output_metadata(
            row,
            row_kind="Secondary-rater",
            case_id=case_id,
            expected_schema_version=SECONDARY_SCHEMA_VERSION,
            actor_key="rater",
            expected_model=expected_model,
        )
        validate_row_queue_hash(
            row,
            row_kind="Secondary-rater",
            case_id=case_id,
            queue_row=queue_by_case[case_id],
        )
        validate_row_rule_hash(
            row,
            row_kind="Secondary-rater",
            case_id=case_id,
            expected_rule_hash=expected_rule_hash,
        )


def validate_adjudication_evidence_context(
    adjudication_by_case: dict[str, dict[str, Any]],
    *,
    queue_by_case: dict[str, dict[str, Any]],
    primary_labels_by_case: dict[str, str],
    secondary_labels_by_case: dict[str, str],
    expected_rule_hash: str,
    expected_model: str,
) -> None:
    for case_id, row in adjudication_by_case.items():
        validate_row_output_metadata(
            row,
            row_kind="Adjudication",
            case_id=case_id,
            expected_schema_version=ADJUDICATION_SCHEMA_VERSION,
            actor_key="adjudicator",
            expected_model=expected_model,
        )
        validate_row_queue_hash(
            row,
            row_kind="Adjudication",
            case_id=case_id,
            queue_row=queue_by_case[case_id],
        )
        validate_row_rule_hash(
            row,
            row_kind="Adjudication",
            case_id=case_id,
            expected_rule_hash=expected_rule_hash,
        )
        expected_primary = primary_labels_by_case[case_id]
        if row.get("primary_grade") != expected_primary:
            raise ValueError(
                f"Adjudication row for {case_id} has primary_grade "
                f"{row.get('primary_grade')!r}, expected {expected_primary!r}"
            )
        expected_secondary = secondary_labels_by_case[case_id]
        if row.get("secondary_grade") != expected_secondary:
            raise ValueError(
                f"Adjudication row for {case_id} has secondary_grade "
                f"{row.get('secondary_grade')!r}, expected {expected_secondary!r}"
            )


def validate_calibration_queue_for_launch(
    queue_rows: list[dict[str, Any]],
    *,
    min_cases: int = OPEN_CORRECTNESS_CALIBRATION_MIN_CASES,
    allow_undersized: bool = False,
) -> None:
    """Refuse to launch secondary labeling on a non-claimable calibration queue."""
    failures: list[str] = []
    if not allow_undersized and len(queue_rows) < min_cases:
        failures.append(
            f"n_cases below policy threshold: {len(queue_rows)} < {min_cases}"
        )

    missing_case_rows = [
        idx
        for idx, row in enumerate(queue_rows, start=1)
        if not isinstance(row.get("calibration_case_id"), str)
        or not row.get("calibration_case_id", "").strip()
    ]
    if missing_case_rows:
        failures.append(
            "missing calibration_case_id values at queue rows "
            + ", ".join(map(str, missing_case_rows[:5]))
            + (" ..." if len(missing_case_rows) > 5 else "")
        )
    case_ids = [
        str(row.get("calibration_case_id", ""))
        for row in queue_rows
        if isinstance(row.get("calibration_case_id"), str)
        and row.get("calibration_case_id", "").strip()
    ]
    duplicates = sorted(
        {case_id for case_id in case_ids if case_ids.count(case_id) > 1 and case_id}
    )
    if duplicates:
        failures.append(
            "duplicate calibration_case_id values: "
            + ", ".join(duplicates[:5])
            + (" ..." if len(duplicates) > 5 else "")
        )

    audit_n = sum(
        1
        for row in queue_rows
        if row.get("sample_source") == "audit_queue_disagreement"
    )
    stratified_n = sum(
        1
        for row in queue_rows
        if str(row.get("sample_source", "")).startswith("stratified_panel")
    )
    if audit_n == 0:
        failures.append(
            "audit_queue_disagreement_n == 0 (no rows with "
            "sample_source == 'audit_queue_disagreement')"
        )
    if stratified_n == 0:
        failures.append(
            "stratified_panel_n == 0 (no rows with sample_source starting "
            "with 'stratified_panel')"
        )

    grades: set[str] = set()
    invalid_grades: list[str] = []
    for idx, row in enumerate(queue_rows, start=1):
        try:
            grades.add(grade_from_row(row, row_kind="primary queue"))
        except ValueError as exc:
            case_id = row.get("calibration_case_id") or f"row {idx}"
            invalid_grades.append(f"{case_id}: {exc}")
    if invalid_grades:
        failures.append(
            "invalid primary grade row(s): "
            + "; ".join(invalid_grades[:5])
            + (" ..." if len(invalid_grades) > 5 else "")
        )
    if len(grades) < 2:
        failures.append(
            "fewer than 2 distinct primary grades in queue "
            f"(found {sorted(grades) or 'none'}); kappa would be undefined"
        )

    if failures:
        raise CalibrationQueueLaunchError(
            "Calibration queue is not claimable; refusing to launch secondary "
            "labeling. Failures:\n  - " + "\n  - ".join(failures)
        )


def agreement_kappa(labels_a: list[str], labels_b: list[str]) -> float | None:
    if not labels_a or not labels_b or len(set(labels_a) | set(labels_b)) == 1:
        return None
    score = float(
        cohen_kappa_score(
            labels_a, labels_b, labels=list(SIMID_OPEN_FINAL_GRADE_LABELS)
        )
    )
    if not math.isfinite(score):
        return None
    return score


def gwet_ac1_for_labels(
    labels_a: list[str],
    labels_b: list[str],
    *,
    labels: tuple[str, ...],
) -> float:
    if len(labels_a) != len(labels_b):
        raise ValueError("AC1 requires equal-length label sequences")
    n_items = len(labels_a)
    if n_items == 0:
        return 0.0

    if set(labels_a) == set(labels_b) and len(set(labels_a)) == 1:
        return 1.0

    observed = sum(a == b for a, b in zip(labels_a, labels_b, strict=True)) / n_items
    counts = Counter(labels_a)
    counts.update(labels_b)
    expected = 0.0
    category_count = len(labels)
    for label in labels:
        p_label = counts.get(label, 0) / (2.0 * n_items)
        expected += p_label * (1.0 - p_label) / (category_count - 1.0)

    if abs(1.0 - expected) < 1e-12:
        return 1.0
    return float((observed - expected) / (1.0 - expected))


def format_optional_metric(value: Any) -> str:
    return "not_recorded" if value is None else f"{float(value):.4f}"


def sample_source_counts(queue_rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("sample_source", "unknown")) for row in queue_rows)
    audit_n = counts.get("audit_queue_disagreement", 0)
    stratified_n = sum(
        count
        for source, count in counts.items()
        if source.startswith("stratified_panel")
    )
    return {
        "audit_queue_disagreement_n": audit_n,
        "stratified_panel_n": stratified_n,
        "counts_by_source": dict(sorted(counts.items())),
    }


def _nested_value(payload: dict[str, Any], *path: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _maybe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def open_calibration_policy() -> dict[str, Any]:
    return {
        "schema_version": OPEN_CORRECTNESS_CALIBRATION_SCHEMA_VERSION,
        "target": OPEN_CORRECTNESS_CALIBRATION_TARGET,
        "requires_pre_frozen_rule": True,
        "requires_audit_queue_disagreement_sample": True,
        "requires_stratified_panel": True,
        "min_cases": OPEN_CORRECTNESS_CALIBRATION_MIN_CASES,
        "min_cohen_kappa": OPEN_CORRECTNESS_CALIBRATION_MIN_KAPPA,
        "min_gwet_ac1": OPEN_CORRECTNESS_CALIBRATION_MIN_AC1,
        "max_rule_gap_cases": 0,
    }


def _calibration_rule_is_pre_frozen(rule: Any) -> bool:
    if not isinstance(rule, dict):
        return False
    if rule.get("pre_frozen") is True:
        return True
    return all(
        isinstance(rule.get(field), str) and bool(str(rule.get(field)).strip())
        for field in ("path", "content_sha256", "git_commit")
    )


def normalize_open_calibration_summary(
    payload: dict[str, Any],
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    """Validate and normalize L4-style calibration evidence for open grades."""
    schema_version = str(payload.get("schema_version", "")).strip()
    status = str(payload.get("status", "")).strip()
    target = str(payload.get("target", "")).strip()
    sampling_raw = payload.get("sampling")
    sampling: dict[str, Any] = sampling_raw if isinstance(sampling_raw, dict) else {}
    rule = payload.get("rule")
    if not isinstance(rule, dict):
        rule = _nested_value(payload, "provenance", "adjudication_rule")

    n_cases = _maybe_int(
        payload.get("n_cases") or _nested_value(payload, "irr", "n_cases")
    )
    audit_n = _maybe_int(
        _nested_value(sampling, "audit_queue_disagreement_n")
        or _nested_value(sampling, "audit_queue_disagreements_n")
    )
    stratified_n = _maybe_int(_nested_value(sampling, "stratified_panel_n"))
    cohen_kappa = _maybe_float(_nested_value(payload, "irr", "cohen_kappa"))
    gwet_ac1 = _maybe_float(_nested_value(payload, "irr", "gwet_ac1"))
    raw_agreement_count = _maybe_int(
        _nested_value(payload, "irr", "raw_agreement", "count")
    )
    raw_agreement_n = _maybe_int(_nested_value(payload, "irr", "raw_agreement", "n"))
    rule_gap_count = _maybe_int(
        _nested_value(payload, "adjudication", "rule_gap_cases", "count")
    )
    rule_gap_n = _maybe_int(
        _nested_value(payload, "adjudication", "rule_gap_cases", "n")
    )

    blockers: list[str] = []
    if schema_version != OPEN_CORRECTNESS_CALIBRATION_SCHEMA_VERSION:
        blockers.append("schema_version_mismatch")
    if status != "adjudicated":
        blockers.append("status_not_adjudicated")
    if target != OPEN_CORRECTNESS_CALIBRATION_TARGET:
        blockers.append("target_mismatch")
    if not _calibration_rule_is_pre_frozen(rule):
        blockers.append("pre_frozen_rule_not_recorded")
    if n_cases is None or n_cases <= 0:
        blockers.append("n_cases_not_recorded")
    elif n_cases < OPEN_CORRECTNESS_CALIBRATION_MIN_CASES:
        blockers.append(OPEN_CORRECTNESS_CALIBRATION_N_CASES_BLOCKER)
    if audit_n is None or audit_n <= 0:
        blockers.append("audit_queue_disagreement_sample_not_recorded")
    if stratified_n is None or stratified_n <= 0:
        blockers.append("stratified_panel_not_recorded")
    if cohen_kappa is None:
        blockers.append("cohen_kappa_not_recorded")
    elif cohen_kappa < OPEN_CORRECTNESS_CALIBRATION_MIN_KAPPA:
        blockers.append("cohen_kappa_below_threshold")
    if gwet_ac1 is None:
        blockers.append("gwet_ac1_not_recorded")
    elif gwet_ac1 < OPEN_CORRECTNESS_CALIBRATION_MIN_AC1:
        blockers.append("gwet_ac1_below_threshold")
    if rule_gap_count is None or rule_gap_n is None:
        blockers.append("rule_gap_not_recorded")
    elif rule_gap_count > 0:
        blockers.append(OPEN_CORRECTNESS_CALIBRATION_RULE_GAP_BLOCKER)

    if raw_agreement_count is None or raw_agreement_n is None:
        raw_agreement = _nested_value(payload, "irr", "raw_agreement")
    else:
        raw_agreement = build_rate_summary(raw_agreement_count, raw_agreement_n)

    failure_blocker: str | None
    if not blockers:
        failure_blocker = None
    elif OPEN_CORRECTNESS_CALIBRATION_RULE_GAP_BLOCKER in blockers:
        failure_blocker = OPEN_CORRECTNESS_CALIBRATION_RULE_GAP_BLOCKER
    elif any(
        blocker == OPEN_CORRECTNESS_CALIBRATION_N_CASES_BLOCKER
        or blocker.endswith("_below_threshold")
        for blocker in blockers
    ):
        failure_blocker = OPEN_CORRECTNESS_CALIBRATION_FAILED_BLOCKER
    else:
        failure_blocker = OPEN_CORRECTNESS_CALIBRATION_INVALID_BLOCKER

    return {
        "schema_version": schema_version,
        "status": status,
        "target": target,
        "path": str(path) if path is not None else None,
        "sampling": {
            "n_cases": n_cases,
            "audit_queue_disagreement_n": audit_n,
            "stratified_panel_n": stratified_n,
        },
        "rule": rule if isinstance(rule, dict) else None,
        "irr": {
            "n_cases": n_cases,
            "raw_agreement": raw_agreement,
            "cohen_kappa": cohen_kappa,
            "gwet_ac1": gwet_ac1,
        },
        "adjudication": {
            "rule_gap_cases": {
                "count": rule_gap_count,
                "n": rule_gap_n,
            },
        },
        "claimability": {
            "passed": not blockers,
            "blockers": blockers,
            "blocker": failure_blocker,
            "policy": open_calibration_policy(),
        },
    }


def validate_open_calibration_summary_binding(
    payload: dict[str, Any],
    *,
    run_dir: Path,
    path: Path,
) -> None:
    queue_path = run_dir / "open_calibration_queue.jsonl"
    if not queue_path.exists():
        return
    queue_ids = [str(row["calibration_case_id"]) for row in load_jsonl(queue_path)]
    adjudicated_rows = payload.get("adjudicated_rows")
    if not isinstance(adjudicated_rows, list):
        raise ValueError(
            f"Calibration summary {path} cannot be bound to {queue_path}: "
            "adjudicated_rows is not recorded"
        )
    summary_ids = [
        str(row.get("calibration_case_id"))
        for row in adjudicated_rows
        if isinstance(row, dict)
    ]
    if sorted(summary_ids) != sorted(queue_ids):
        raise ValueError(
            f"Calibration summary {path} case IDs do not exactly match {queue_path}"
        )
    n_cases = _maybe_int(payload.get("n_cases"))
    if n_cases != len(queue_ids):
        raise ValueError(
            f"Calibration summary {path} n_cases={n_cases!r} does not match "
            f"{len(queue_ids)} queue rows in {queue_path}"
        )
    expected_digest = case_ids_sha256(queue_ids)
    observed_digest = payload.get("queue_case_ids_sha256")
    if observed_digest is not None and observed_digest != expected_digest:
        raise ValueError(
            f"Calibration summary {path} queue_case_ids_sha256 does not match "
            f"{queue_path}"
        )
    expected_content_digest = file_sha256(queue_path)
    observed_content_digest = _nested_value(
        payload,
        "inputs",
        "queue",
        "content_sha256",
    )
    if not isinstance(observed_content_digest, str) or not observed_content_digest:
        raise ValueError(
            f"Calibration summary {path} cannot be bound to {queue_path}: "
            "inputs.queue.content_sha256 is not recorded"
        )
    if observed_content_digest != expected_content_digest:
        raise ValueError(
            f"Calibration summary {path} inputs.queue.content_sha256 does not "
            f"match {queue_path}"
        )


def load_open_calibration_summary(
    path: Path | None,
    *,
    run_dir: Path | None = None,
) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected SIMID open calibration JSON object in {path}")
    if run_dir is not None:
        validate_open_calibration_summary_binding(
            payload,
            run_dir=run_dir,
            path=path,
        )
    return normalize_open_calibration_summary(payload, path=path)


def finalize_open_calibration(
    *,
    queue_rows: list[dict[str, Any]],
    secondary_rows: list[dict[str, Any]],
    adjudication_rows: list[dict[str, Any]],
    rule_path: Path,
    primary_rater_name: str,
    secondary_rater_name: str,
    secondary_model: str = "gpt-5.5",
    adjudicator_model: str = "gpt-5.5",
) -> dict[str, Any]:
    queue_by_case = index_by_case(queue_rows, row_kind="queue")
    secondary_by_case = index_by_case(secondary_rows, row_kind="secondary-rater")
    adjudication_by_case = index_by_case(adjudication_rows, row_kind="adjudication")
    expected_rule_hash = file_sha256(rule_path)

    expected_cases = set(queue_by_case)
    if set(secondary_by_case) != expected_cases:
        raise ValueError("Secondary rater labels must cover every calibration case")
    validate_secondary_evidence_context(
        secondary_by_case,
        queue_by_case=queue_by_case,
        expected_rule_hash=expected_rule_hash,
        expected_model=secondary_model,
    )

    sorted_case_ids = sorted(expected_cases)
    primary_labels_by_case = {
        case_id: grade_from_row(queue_by_case[case_id], row_kind="primary queue")
        for case_id in sorted_case_ids
    }
    secondary_labels_by_case = {
        case_id: grade_from_row(secondary_by_case[case_id], row_kind="secondary-rater")
        for case_id in sorted_case_ids
    }
    primary_labels = [primary_labels_by_case[case_id] for case_id in sorted_case_ids]
    secondary_labels = [
        secondary_labels_by_case[case_id] for case_id in sorted_case_ids
    ]
    disagreement_ids = {
        case_id
        for case_id, primary, secondary in zip(
            sorted_case_ids, primary_labels, secondary_labels, strict=True
        )
        if primary != secondary
    }
    if set(adjudication_by_case) != disagreement_ids:
        raise ValueError(
            "Adjudication labels must cover exactly the primary/secondary disagreements"
        )
    validate_adjudication_evidence_context(
        adjudication_by_case,
        queue_by_case=queue_by_case,
        primary_labels_by_case=primary_labels_by_case,
        secondary_labels_by_case=secondary_labels_by_case,
        expected_rule_hash=expected_rule_hash,
        expected_model=adjudicator_model,
    )

    agreement_count = len(sorted_case_ids) - len(disagreement_ids)
    raw_agreement = build_rate_summary(agreement_count, len(sorted_case_ids))
    rule_gap_count = sum(
        1 for row in adjudication_rows if bool(row.get("rule_gap", False))
    )
    rule_gap_summary = build_rate_summary(rule_gap_count, len(disagreement_ids))

    adjudicated_rows: list[dict[str, Any]] = []
    disagreement_rows: list[dict[str, Any]] = []
    for case_id, primary, secondary in zip(
        sorted_case_ids, primary_labels, secondary_labels, strict=True
    ):
        queue_row = queue_by_case[case_id]
        adjudication = adjudication_by_case.get(case_id)
        if adjudication is None:
            final_grade = primary
            label_source = "consensus"
            rule_gap = False
        else:
            final_grade = grade_from_row(adjudication, row_kind="adjudication")
            label_source = "adjudication"
            rule_gap = bool(adjudication.get("rule_gap", False))
            disagreement_rows.append(
                {
                    "calibration_case_id": case_id,
                    "sample_source": queue_row.get("sample_source"),
                    "primary_grade": primary,
                    "secondary_grade": secondary,
                    "adjudicated_grade": final_grade,
                    "rule_gap": rule_gap,
                    "adjudication_notes": adjudication.get("notes", ""),
                }
            )
        adjudicated_rows.append(
            {
                "calibration_case_id": case_id,
                "sample_source": queue_row.get("sample_source"),
                "sample_id": queue_row.get("sample_id"),
                "condition": queue_row.get("condition"),
                "alpha": queue_row.get("alpha"),
                "dataset": queue_row.get("dataset"),
                "primary_grade": primary,
                "secondary_grade": secondary,
                "final_grade": final_grade,
                "label_source": label_source,
                "rule_gap": rule_gap,
            }
        )

    source_counts = sample_source_counts(queue_rows)
    kappa = agreement_kappa(primary_labels, secondary_labels)
    summary: dict[str, Any] = {
        "schema_version": OPEN_CORRECTNESS_CALIBRATION_SCHEMA_VERSION,
        "status": "adjudicated",
        "target": OPEN_CORRECTNESS_CALIBRATION_TARGET,
        "calibration_queue_schema_version": SIMID_OPEN_CALIBRATION_QUEUE_SCHEMA_VERSION,
        "n_cases": len(sorted_case_ids),
        "queue_case_ids_sha256": case_ids_sha256(sorted_case_ids),
        "sampling": source_counts,
        "rule": build_adjudication_rule_metadata(rule_path),
        "raters": {
            "primary": primary_rater_name,
            "secondary": secondary_rater_name,
            "secondary_model": secondary_model,
            "adjudicator_model": (
                adjudicator_model if len(disagreement_ids) > 0 else None
            ),
            "prompt_version": PROMPT_VERSION,
        },
        "irr": {
            "n_cases": len(sorted_case_ids),
            "raw_agreement": raw_agreement,
            "cohen_kappa": round(kappa, 4) if kappa is not None else None,
            "gwet_ac1": round(
                gwet_ac1_for_labels(
                    primary_labels,
                    secondary_labels,
                    labels=SIMID_OPEN_FINAL_GRADE_LABELS,
                ),
                4,
            ),
            "n_disagreements": len(disagreement_ids),
        },
        "adjudication": {
            "n_disagreements": len(disagreement_ids),
            "rule_gap_cases": rule_gap_summary,
        },
        "adjudicated_rows": adjudicated_rows,
        "disagreements": disagreement_rows,
    }
    summary["claimability"] = normalize_open_calibration_summary(summary)[
        "claimability"
    ]
    return summary
