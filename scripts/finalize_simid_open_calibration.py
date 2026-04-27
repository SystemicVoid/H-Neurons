#!/usr/bin/env python3
"""Finalize SIMID open-correctness calibration labels."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score

from analyze_simid import (
    OPEN_CORRECTNESS_CALIBRATION_SCHEMA_VERSION,
    OPEN_CORRECTNESS_CALIBRATION_TARGET,
    SIMID_OPEN_CALIBRATION_QUEUE_SCHEMA_VERSION,
    SIMID_OPEN_FINAL_GRADE_LABELS,
    SIMID_OPEN_FINAL_GRADES,
    gwet_ac1_for_labels,
    normalize_open_calibration_summary,
)
from bridge_irr import build_adjudication_rule_metadata
from uncertainty import build_rate_summary
from utils import (
    finish_run_provenance,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_ids_sha256(case_ids: list[str]) -> str:
    payload = json.dumps(sorted(case_ids), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def input_file_metadata(path: Path) -> dict[str, Any]:
    return {"path": str(path), "content_sha256": file_sha256(path)}


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


def finalize_open_calibration(
    *,
    queue_rows: list[dict[str, Any]],
    secondary_rows: list[dict[str, Any]],
    adjudication_rows: list[dict[str, Any]],
    rule_path: Path,
    primary_rater_name: str,
    secondary_rater_name: str,
) -> dict[str, Any]:
    queue_by_case = index_by_case(queue_rows, row_kind="queue")
    secondary_by_case = index_by_case(secondary_rows, row_kind="secondary-rater")
    adjudication_by_case = index_by_case(adjudication_rows, row_kind="adjudication")

    expected_cases = set(queue_by_case)
    if set(secondary_by_case) != expected_cases:
        raise ValueError("Secondary rater labels must cover every calibration case")

    sorted_case_ids = sorted(expected_cases)
    primary_labels = [
        grade_from_row(queue_by_case[case_id], row_kind="primary queue")
        for case_id in sorted_case_ids
    ]
    secondary_labels = [
        grade_from_row(secondary_by_case[case_id], row_kind="secondary-rater")
        for case_id in sorted_case_ids
    ]
    disagreement_ids = {
        case_id
        for case_id, primary, secondary in zip(
            sorted_case_ids, primary_labels, secondary_labels
        )
        if primary != secondary
    }
    if set(adjudication_by_case) != disagreement_ids:
        raise ValueError(
            "Adjudication labels must cover exactly the primary/secondary disagreements"
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
        sorted_case_ids, primary_labels, secondary_labels
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-path", type=Path, required=True)
    parser.add_argument("--secondary-rater-path", type=Path, required=True)
    parser.add_argument("--adjudication-path", type=Path, required=True)
    parser.add_argument("--rule-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-rater-name", default="simid_open_primary_judge")
    parser.add_argument("--secondary-rater-name", default="simid_open_secondary_rater")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output.exists() and not args.allow_overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing SIMID open calibration summary: "
            f"{args.output}. Pass --allow-overwrite to replace it explicitly."
        )
    input_metadata = {
        "queue": input_file_metadata(args.queue_path),
        "secondary_rater": input_file_metadata(args.secondary_rater_path),
        "adjudication": (
            input_file_metadata(args.adjudication_path)
            if args.adjudication_path.exists()
            else {"path": str(args.adjudication_path), "content_sha256": None}
        ),
        "rule": input_file_metadata(args.rule_path),
    }
    provenance = start_run_provenance(
        args,
        primary_target=args.output,
        output_targets=[args.output],
        extra={
            "simid_schema": OPEN_CORRECTNESS_CALIBRATION_SCHEMA_VERSION,
            "inputs": input_metadata,
        },
    )
    status = "completed"
    extra: dict[str, Any] = {}
    try:
        summary = finalize_open_calibration(
            queue_rows=load_jsonl(args.queue_path),
            secondary_rows=load_jsonl(args.secondary_rater_path),
            adjudication_rows=load_jsonl(args.adjudication_path, required=False),
            rule_path=args.rule_path,
            primary_rater_name=args.primary_rater_name,
            secondary_rater_name=args.secondary_rater_name,
        )
        summary["inputs"] = input_metadata
        write_json(args.output, summary)
        extra = {
            "n_cases": summary["n_cases"],
            "claimability": summary["claimability"],
            "queue_case_ids_sha256": summary["queue_case_ids_sha256"],
        }
        print(
            "Finalized SIMID open calibration: "
            f"agreement={summary['irr']['raw_agreement']['estimate']:.1%} "
            f"kappa={format_optional_metric(summary['irr']['cohen_kappa'])} "
            f"AC1={format_optional_metric(summary['irr']['gwet_ac1'])}"
        )
    except BaseException as exc:
        status = provenance_status_for_exception(exc)
        extra["error"] = provenance_error_message(exc)
        raise
    finally:
        finish_run_provenance(provenance, status, extra)


if __name__ == "__main__":
    main()
