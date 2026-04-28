#!/usr/bin/env python3
"""Validate and merge SIMID open independent-rater label files."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PACKAGE_DIR = (
    ROOT / "data/gemma3_4b/intervention/"
    "simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens/"
    "mvp_20260427_calibration/human_review_package"
)
LLM_BATCH_ROOT_DIR = "llm_blind_batches"
LLM_BATCH_LABEL_SCHEMA_VERSION = "simid_open_llm_blind_rater_label/v1"
FINAL_LABEL_SCHEMA_VERSION = "simid_open_independent_rater_label/v1"
VALID_LABELS = {"CORRECT", "INCORRECT", "NOT_ATTEMPTED"}
VALID_FLAGS = {
    "bridge_partial_entity_or_modifier",
    "truthfulqa_non_answer_boundary",
    "truthfulqa_qualified_answer_boundary",
    "wrong_extra_answer",
    "multiple_candidates_no_commitment",
    "alias_too_broad_or_too_narrow",
    "malformed_case",
    "other_boundary",
}
BATCH_LABEL_KEYS = {
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
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate SIMID open blind batch labels and optionally merge."
    )
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument(
        "--label-file",
        type=Path,
        help="Validate one batch label file. Defaults to all expected batches.",
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        help="Batch directory for --label-file. Defaults to label file parent.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write merged calibration-ID labels after validating all batches.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing --output file.",
    )
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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
    expect(len(flags) == len(set(flags)), f"{row_name}: duplicate flags")
    validated_flags: list[str] = []
    for flag in flags:
        expect(isinstance(flag, str), f"{row_name}: non-string flag")
        expect(flag in VALID_FLAGS, f"{row_name}: invalid flag {flag!r}")
        validated_flags.append(flag)
    return validated_flags


def expected_batch_cases(batch_dir: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(batch_dir / "review_cases_blind.jsonl")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        blind_case_id = require_string(
            row,
            "blind_case_id",
            row_name=f"{batch_dir.name} input",
        )
        expect(blind_case_id not in by_id, f"{batch_dir}: duplicate blind_case_id")
        by_id[blind_case_id] = row
    return by_id


def validate_batch_file(label_file: Path, batch_dir: Path) -> list[dict[str, Any]]:
    expected = expected_batch_cases(batch_dir)
    expected_hash = file_sha256(batch_dir / "review_cases_blind.jsonl")
    label_rows = load_jsonl(label_file)
    seen: set[str] = set()

    for index, row in enumerate(label_rows, start=1):
        row_name = f"{label_file}:{index}"
        expect(set(row) <= BATCH_LABEL_KEYS, f"{row_name}: unexpected keys")
        expect(
            "calibration_case_id" not in row,
            f"{row_name}: calibration_case_id is not allowed in blind batch labels",
        )
        expect(
            row.get("schema_version") == LLM_BATCH_LABEL_SCHEMA_VERSION,
            f"{row_name}: invalid schema_version",
        )
        blind_case_id = require_string(row, "blind_case_id", row_name=row_name)
        expect(
            blind_case_id in expected,
            f"{row_name}: blind_case_id not present in batch input",
        )
        expect(blind_case_id not in seen, f"{row_name}: duplicate blind_case_id")
        seen.add(blind_case_id)

        review_order = require_int(row, "review_order", row_name=row_name)
        expect(
            review_order == expected[blind_case_id].get("review_order"),
            f"{row_name}: review_order does not match batch input",
        )
        label = row.get("label")
        expect(label in VALID_LABELS, f"{row_name}: invalid label {label!r}")
        confidence = require_int(row, "confidence", row_name=row_name)
        expect(1 <= confidence <= 5, f"{row_name}: confidence outside 1-5")
        expect(type(row.get("rule_gap")) is bool, f"{row_name}: invalid rule_gap")
        validate_flags(row, row_name=row_name)
        expect(isinstance(row.get("notes"), str), f"{row_name}: invalid notes")
        rater = row.get("rater")
        if not isinstance(rater, dict):
            raise ValueError(f"{row_name}: invalid rater")
        expect(rater.get("type") == "llm", f"{row_name}: rater.type must be llm")
        expect(isinstance(rater.get("model"), str), f"{row_name}: missing model")
        expect(
            isinstance(rater.get("prompt_version"), str),
            f"{row_name}: missing prompt_version",
        )
        expect(
            row.get("blind_cases_file_sha256") == expected_hash,
            f"{row_name}: blind_cases_file_sha256 mismatch",
        )

    expect(
        seen == set(expected),
        f"{label_file}: expected {len(expected)} labels, got {len(seen)}",
    )
    return label_rows


def load_manifest_batches(package_dir: Path) -> list[dict[str, Any]]:
    manifest = load_json(package_dir / "review_manifest.json")
    batching = manifest.get("llm_batching")
    if not isinstance(batching, dict):
        raise ValueError("review_manifest.json missing llm_batching")
    batches = batching.get("batches")
    expect(isinstance(batches, list), "review_manifest.json missing batch list")
    return [batch for batch in batches if isinstance(batch, dict)]


def expected_batch_label_files(package_dir: Path) -> list[tuple[Path, Path]]:
    files: list[tuple[Path, Path]] = []
    for batch in load_manifest_batches(package_dir):
        label_path = ROOT / str(batch["label_part_file"])
        case_path = ROOT / str(batch["case_file"])
        files.append((label_path, case_path.parent))
    return files


def load_blind_case_map(package_dir: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(package_dir / "llm_blind_case_map.jsonl")
    by_blind_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        blind_case_id = require_string(row, "blind_case_id", row_name="blind map")
        expect(blind_case_id not in by_blind_id, "duplicate blind_case_id in map")
        require_string(row, "calibration_case_id", row_name="blind map")
        require_int(row, "review_order", row_name="blind map")
        by_blind_id[blind_case_id] = row
    return by_blind_id


def merged_label_rows(
    *,
    package_dir: Path,
    batch_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    manifest = load_json(package_dir / "review_manifest.json")
    full_hash = require_string(
        manifest,
        "blind_cases_file_sha256",
        row_name="review_manifest.json",
    )
    case_map = load_blind_case_map(package_dir)
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for row in batch_rows:
        blind_case_id = require_string(row, "blind_case_id", row_name="batch label")
        expect(blind_case_id in case_map, f"missing map row for {blind_case_id}")
        expect(blind_case_id not in seen, f"duplicate merged label {blind_case_id}")
        seen.add(blind_case_id)
        map_row = case_map[blind_case_id]
        merged.append(
            {
                "schema_version": FINAL_LABEL_SCHEMA_VERSION,
                "calibration_case_id": map_row["calibration_case_id"],
                "review_order": map_row["review_order"],
                "label": row["label"],
                "confidence": row["confidence"],
                "rule_gap": row["rule_gap"],
                "flags": row["flags"],
                "notes": row["notes"],
                "rater": row["rater"],
                "blind_cases_file_sha256": full_hash,
            }
        )
    expect(seen == set(case_map), f"merged labels cover {len(seen)} of {len(case_map)}")
    merged.sort(key=lambda item: int(item["review_order"]))
    return merged


def validate_one(args: argparse.Namespace) -> int:
    label_file = args.label_file.resolve()
    batch_dir = (
        args.batch_dir.resolve()
        if args.batch_dir is not None
        else label_file.parent.resolve()
    )
    rows = validate_batch_file(label_file, batch_dir)
    print(f"Validated {len(rows)} rows: {label_file}")
    return 0


def validate_all(args: argparse.Namespace) -> int:
    package_dir = args.package_dir.resolve()
    all_rows: list[dict[str, Any]] = []
    for label_file, batch_dir in expected_batch_label_files(package_dir):
        expect(label_file.exists(), f"Missing batch label file: {label_file}")
        all_rows.extend(validate_batch_file(label_file, batch_dir))

    print(f"Validated {len(all_rows)} blind batch labels.")
    if args.output is not None:
        output = args.output.resolve()
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite")
        merged = merged_label_rows(package_dir=package_dir, batch_rows=all_rows)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(output, merged)
        print(f"Wrote {len(merged)} merged labels: {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.label_file is not None:
        return validate_one(args)
    return validate_all(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
