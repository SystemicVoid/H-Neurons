#!/usr/bin/env python3
"""Analyze returned labels for a prospective SIMID open calibration gate."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score

from export_simid_prospective_open_calibration_gate import (
    LABEL_SCHEMA_VERSION,
    PACKAGE_SCHEMA_VERSION,
    PRIVATE_CASE_SCHEMA_VERSION,
    ROOT,
    VALID_FLAGS,
    file_sha256,
)
from simid_open_calibration import SIMID_OPEN_FINAL_GRADE_LABELS, gwet_ac1_for_labels
from uncertainty import build_rate_summary
from utils import (
    finish_run_provenance,
    format_path_for_metadata,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)


DEFAULT_PACKAGE_DIR = (
    ROOT / "data/gemma3_4b/intervention/"
    "simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/"
    "mvp_20260427_calibration/human_review_package/"
    "prospective_open_calibration_gate_20260429"
)
SUMMARY_SCHEMA_VERSION = "simid_prospective_open_calibration_analysis/v1"
VALID_LABELS = set(SIMID_OPEN_FINAL_GRADE_LABELS)
ALLOWED_LABEL_KEYS = {
    "schema_version",
    "blind_case_id",
    "review_order",
    "label",
    "confidence",
    "rule_gap",
    "flags",
    "notes",
    "rater",
    "blind_cases_file_sha256",
    "rubric_sha256",
    "labeled_at_utc",
}
FORBIDDEN_LABEL_KEYS = {
    "source_case_id",
    "calibration_case_id",
    "sample_id",
    "reference_label",
    "primary_label",
    "secondary_label",
    "adjudicated_label",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing analysis output JSON.",
    )
    return parser.parse_args(argv)


def relpath(path: Path) -> str:
    return format_path_for_metadata(path, root=ROOT)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(payload)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def require_string(row: dict[str, Any], key: str, *, row_name: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{row_name}: invalid {key}")
    return value


def require_int(row: dict[str, Any], key: str, *, row_name: str) -> int:
    value = row.get(key)
    if type(value) is not int:
        raise ValueError(f"{row_name}: invalid {key}")
    return value


def validate_flags(row: dict[str, Any], *, row_name: str) -> list[str]:
    flags = row.get("flags")
    if not isinstance(flags, list):
        raise ValueError(f"{row_name}: flags must be a list")
    if len(flags) != len(set(flags)):
        raise ValueError(f"{row_name}: duplicate flags")
    parsed: list[str] = []
    for flag in flags:
        if not isinstance(flag, str) or flag not in VALID_FLAGS:
            raise ValueError(f"{row_name}: invalid flag {flag!r}")
        parsed.append(flag)
    return parsed


def index_private_rows(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        row_name = f"private_case_map.jsonl:{index}"
        if row.get("schema_version") != PRIVATE_CASE_SCHEMA_VERSION:
            raise ValueError(f"{row_name}: invalid schema_version")
        blind_case_id = require_string(row, "blind_case_id", row_name=row_name)
        if blind_case_id in indexed:
            raise ValueError(f"{row_name}: duplicate blind_case_id")
        label = row.get("reference_label")
        if label not in VALID_LABELS:
            raise ValueError(f"{row_name}: invalid reference_label {label!r}")
        require_int(row, "review_order", row_name=row_name)
        indexed[blind_case_id] = row
    return indexed


def validate_label_rows(
    *,
    label_rows: list[dict[str, Any]],
    private_by_id: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
    label_file: Path,
) -> list[dict[str, Any]]:
    blind_hash = str(manifest["files"]["review_cases_blind"]["content_sha256"])
    rubric_hash = str(manifest["rubric"]["content_sha256"])
    seen: set[str] = set()
    for index, row in enumerate(label_rows, start=1):
        row_name = f"{label_file}:{index}"
        unexpected = set(row) - ALLOWED_LABEL_KEYS
        if unexpected:
            raise ValueError(f"{row_name}: unexpected keys {sorted(unexpected)}")
        leaked = FORBIDDEN_LABEL_KEYS & set(row)
        if leaked:
            raise ValueError(f"{row_name}: leaked private keys {sorted(leaked)}")
        if row.get("schema_version") != LABEL_SCHEMA_VERSION:
            raise ValueError(f"{row_name}: invalid schema_version")
        blind_case_id = require_string(row, "blind_case_id", row_name=row_name)
        if blind_case_id not in private_by_id:
            raise ValueError(f"{row_name}: blind_case_id not in package")
        if blind_case_id in seen:
            raise ValueError(f"{row_name}: duplicate blind_case_id")
        seen.add(blind_case_id)
        expected = private_by_id[blind_case_id]
        review_order = require_int(row, "review_order", row_name=row_name)
        if review_order != expected.get("review_order"):
            raise ValueError(f"{row_name}: review_order mismatch")
        if row.get("label") not in VALID_LABELS:
            raise ValueError(f"{row_name}: invalid label {row.get('label')!r}")
        confidence = require_int(row, "confidence", row_name=row_name)
        if not 1 <= confidence <= 5:
            raise ValueError(f"{row_name}: confidence outside 1-5")
        if type(row.get("rule_gap")) is not bool:
            raise ValueError(f"{row_name}: invalid rule_gap")
        validate_flags(row, row_name=row_name)
        if not isinstance(row.get("notes"), str):
            raise ValueError(f"{row_name}: invalid notes")
        if not isinstance(row.get("rater"), dict):
            raise ValueError(f"{row_name}: invalid rater")
        if row.get("blind_cases_file_sha256") != blind_hash:
            raise ValueError(f"{row_name}: blind_cases_file_sha256 mismatch")
        if row.get("rubric_sha256") != rubric_hash:
            raise ValueError(f"{row_name}: rubric_sha256 mismatch")
    expected_ids = set(private_by_id)
    if seen != expected_ids:
        missing = sorted(expected_ids - seen)[:5]
        extra = sorted(seen - expected_ids)[:5]
        raise ValueError(
            f"{label_file}: incomplete label coverage "
            f"(missing={missing}, extra={extra})"
        )
    return sorted(label_rows, key=lambda row: int(row["review_order"]))


def optional_kappa(
    reference_labels: list[str], returned_labels: list[str]
) -> float | None:
    if len(set(reference_labels) | set(returned_labels)) <= 1:
        return None
    score = float(
        cohen_kappa_score(
            reference_labels,
            returned_labels,
            labels=list(SIMID_OPEN_FINAL_GRADE_LABELS),
        )
    )
    if not math.isfinite(score):
        return None
    return score


def labels_sha256(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def label_count_dict(labels: Iterable[str]) -> dict[str, int]:
    return dict(Counter(labels))


def manifest_path(path_value: Any, *, row_name: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{row_name}: missing path")
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def validate_manifest_bound_file(
    entry: dict[str, Any],
    *,
    row_name: str,
) -> None:
    path = manifest_path(entry.get("path"), row_name=row_name)
    expected_hash = require_string(entry, "content_sha256", row_name=row_name)
    observed_hash = file_sha256(path)
    if observed_hash != expected_hash:
        raise ValueError(
            f"{row_name}: content_sha256 mismatch for {path} "
            f"(expected {expected_hash}, observed {observed_hash})"
        )


def validate_package_file_integrity(manifest: dict[str, Any]) -> None:
    rubric = manifest.get("rubric")
    if not isinstance(rubric, dict):
        raise ValueError("review_manifest.json: missing rubric metadata")
    validate_manifest_bound_file(rubric, row_name="review_manifest.json:rubric")

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("review_manifest.json: missing files metadata")
    for key in (
        "index",
        "review_cases_blind",
        "private_case_map",
        "label_schema",
        "prompt",
        "readme",
    ):
        entry = files.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"review_manifest.json: missing files.{key}")
        validate_manifest_bound_file(
            entry,
            row_name=f"review_manifest.json:files.{key}",
        )


def build_analysis_summary(
    *,
    package_dir: Path,
    label_file: Path,
    manifest: dict[str, Any],
    private_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    private_by_id = index_private_rows(private_rows)
    sorted_label_rows = validate_label_rows(
        label_rows=label_rows,
        private_by_id=private_by_id,
        manifest=manifest,
        label_file=label_file,
    )
    reference_labels = [
        str(private_by_id[str(row["blind_case_id"])]["reference_label"])
        for row in sorted_label_rows
    ]
    returned_labels = [str(row["label"]) for row in sorted_label_rows]
    agreement_count = sum(
        1
        for reference, returned in zip(reference_labels, returned_labels)
        if reference == returned
    )
    raw_agreement = build_rate_summary(agreement_count, len(returned_labels))
    kappa = optional_kappa(reference_labels, returned_labels)
    ac1 = gwet_ac1_for_labels(
        reference_labels,
        returned_labels,
        labels=SIMID_OPEN_FINAL_GRADE_LABELS,
    )
    rule_gap_count = sum(1 for row in sorted_label_rows if bool(row["rule_gap"]))
    policy = dict(manifest["analysis_policy"])
    blockers: list[str] = []
    if manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION:
        blockers.append("package_schema_version_mismatch")
    if len(returned_labels) < int(policy["min_cases"]):
        blockers.append("n_cases_below_minimum")
    if raw_agreement["estimate"] < float(policy["min_raw_agreement"]):
        blockers.append("raw_agreement_below_threshold")
    if kappa is None or kappa < float(policy["min_cohen_kappa"]):
        blockers.append("cohen_kappa_below_threshold")
    if ac1 < float(policy["min_gwet_ac1"]):
        blockers.append("gwet_ac1_below_threshold")
    if rule_gap_count > int(policy["max_rule_gap_cases"]):
        blockers.append("rule_gap_recorded")

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "package_dir": relpath(package_dir),
        "label_file": relpath(label_file),
        "label_file_sha256": file_sha256(label_file),
        "label_rows_canonical_sha256": labels_sha256(sorted_label_rows),
        "n_cases": len(returned_labels),
        "policy": policy,
        "metrics": {
            "raw_agreement": raw_agreement,
            "cohen_kappa": round(kappa, 4) if kappa is not None else None,
            "gwet_ac1": round(ac1, 4),
            "rule_gap_count": rule_gap_count,
            "reference_label_counts": label_count_dict(reference_labels),
            "returned_label_counts": label_count_dict(returned_labels),
        },
        "outcome": {
            "passed": not blockers,
            "blockers": blockers,
        },
        "claimability_guardrail": (
            "This is prospective grading-reliability evidence for future SIMID "
            "open-response use only. It does not upgrade historical MVP "
            "open-correctness metrics by itself."
        ),
    }


def analyze_package(*, package_dir: Path, label_file: Path) -> dict[str, Any]:
    manifest = load_json(package_dir / "review_manifest.json")
    validate_package_file_integrity(manifest)
    private_rows = load_jsonl(package_dir / "private_case_map.jsonl")
    label_rows = load_jsonl(label_file)
    return build_analysis_summary(
        package_dir=package_dir,
        label_file=label_file,
        manifest=manifest,
        private_rows=private_rows,
        label_rows=label_rows,
    )


def output_targets(output: Path | None) -> list[str | os.PathLike[str]]:
    return [] if output is None else [output]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    package_dir = args.package_dir.resolve()
    label_file = args.labels.resolve()
    output = args.output.resolve() if args.output is not None else None
    if output is not None and output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite")

    provenance = None
    status = "completed"
    extra: dict[str, Any] = {}
    if output is not None:
        provenance = start_run_provenance(
            args,
            primary_target=output,
            output_targets=output_targets(output),
            extra={
                "simid_schema": SUMMARY_SCHEMA_VERSION,
                "package_dir": relpath(package_dir),
                "label_file": relpath(label_file),
            },
        )
    try:
        summary = analyze_package(package_dir=package_dir, label_file=label_file)
        if output is None:
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            write_json(output, summary)
            print(
                "Analyzed prospective SIMID open calibration labels: "
                f"passed={summary['outcome']['passed']} -> {relpath(output)}"
            )
        extra = {
            "passed": summary["outcome"]["passed"],
            "blockers": summary["outcome"]["blockers"],
            "n_cases": summary["n_cases"],
        }
    except BaseException as exc:
        status = provenance_status_for_exception(exc)
        extra["error"] = provenance_error_message(exc)
        raise
    finally:
        finish_run_provenance(provenance, status, extra)


if __name__ == "__main__":
    main()
