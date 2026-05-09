"""Shared helpers for blinded review-package artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from artifact_io import (
    file_sha256,
    jsonl_rows_sha256,
    read_jsonl_strict,
    write_canonical_json,
    write_jsonl_rows,
)
from utils import format_path_for_metadata


ROOT = Path(__file__).resolve().parent.parent

REVIEW_PACKAGE_SCHEMA_VERSION = "simid_open_independent_review_package/v1"
BLIND_CASE_SCHEMA_VERSION = "simid_open_blind_review_case/v1"
HUMAN_LABEL_SCHEMA_VERSION = "simid_open_independent_rater_label/v1"
LLM_BATCH_ROOT_DIR = "llm_blind_batches"
LLM_BATCH_CASE_SCHEMA_VERSION = "simid_open_llm_blind_review_case/v1"
LLM_BATCH_LABEL_SCHEMA_VERSION = "simid_open_llm_blind_rater_label/v1"
LLM_BLIND_MAP_SCHEMA_VERSION = "simid_open_llm_blind_case_map/v1"
VALID_LABELS = ("CORRECT", "INCORRECT", "NOT_ATTEMPTED")
VALID_FLAGS = (
    "bridge_partial_entity_or_modifier",
    "truthfulqa_non_answer_boundary",
    "truthfulqa_qualified_answer_boundary",
    "wrong_extra_answer",
    "multiple_candidates_no_commitment",
    "alias_too_broad_or_too_narrow",
    "malformed_case",
    "other_boundary",
)
HUMAN_LABEL_REQUIRED_FIELDS = (
    "schema_version",
    "calibration_case_id",
    "review_order",
    "label",
    "confidence",
    "rule_gap",
    "flags",
    "notes",
    "rater",
    "blind_cases_file_sha256",
)
LLM_BATCH_LABEL_REQUIRED_FIELDS = (
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
)
BATCH_LABEL_KEYS = frozenset(LLM_BATCH_LABEL_REQUIRED_FIELDS)


@dataclass(frozen=True)
class ReviewPackageSchemas:
    package: str
    blind_case: str
    human_label: str
    llm_batch_case: str
    llm_batch_label: str
    llm_blind_map: str


SIMID_OPEN_REVIEW_SCHEMAS = ReviewPackageSchemas(
    package=REVIEW_PACKAGE_SCHEMA_VERSION,
    blind_case=BLIND_CASE_SCHEMA_VERSION,
    human_label=HUMAN_LABEL_SCHEMA_VERSION,
    llm_batch_case=LLM_BATCH_CASE_SCHEMA_VERSION,
    llm_batch_label=LLM_BATCH_LABEL_SCHEMA_VERSION,
    llm_blind_map=LLM_BLIND_MAP_SCHEMA_VERSION,
)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def jsonl_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    return jsonl_rows_sha256(rows)


def byte_sha256_for_rows(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_canonical_json(
        path, payload, pretty=True, ensure_ascii=False, sort_keys=False
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    write_jsonl_rows(path, rows, ensure_ascii=False, sort_keys=False)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl_strict(path)


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_string(row: Mapping[str, Any], key: str, *, row_name: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{row_name}: invalid {key}")
    return value


def require_int(row: Mapping[str, Any], key: str, *, row_name: str) -> int:
    value = row.get(key)
    if type(value) is not int:
        raise ValueError(f"{row_name}: invalid {key}")
    return value


def validate_flags(
    row: Mapping[str, Any],
    *,
    row_name: str,
    valid_flags: Iterable[str] = VALID_FLAGS,
) -> list[str]:
    valid_flag_set = set(valid_flags)
    flags = row.get("flags")
    if not isinstance(flags, list):
        raise ValueError(f"{row_name}: flags must be a list")
    expect(len(flags) == len(set(flags)), f"{row_name}: duplicate flags")
    validated_flags: list[str] = []
    for flag in flags:
        expect(isinstance(flag, str), f"{row_name}: non-string flag")
        expect(flag in valid_flag_set, f"{row_name}: invalid flag {flag!r}")
        validated_flags.append(flag)
    return validated_flags


def blind_case_id_for(calibration_case_id: str, *, seed: str) -> str:
    return f"simid_open_blind_{stable_hash(f'{seed}:llm_blind:{calibration_case_id}')[:16]}"


def llm_blind_case_row(
    blind_row: Mapping[str, Any],
    *,
    seed: str,
) -> dict[str, Any]:
    return {
        "schema_version": LLM_BATCH_CASE_SCHEMA_VERSION,
        "review_order": blind_row["review_order"],
        "blind_case_id": blind_case_id_for(
            str(blind_row["calibration_case_id"]),
            seed=seed,
        ),
        "question": blind_row["question"],
        "gold_aliases": blind_row["gold_aliases"],
        "predicted_answer": blind_row["predicted_answer"],
    }


def llm_blind_case_map_row(
    blind_row: Mapping[str, Any],
    *,
    seed: str,
) -> dict[str, Any]:
    return {
        "schema_version": LLM_BLIND_MAP_SCHEMA_VERSION,
        "review_order": blind_row["review_order"],
        "blind_case_id": blind_case_id_for(
            str(blind_row["calibration_case_id"]),
            seed=seed,
        ),
        "calibration_case_id": blind_row["calibration_case_id"],
    }


def batch_dir_name(batch_index: int) -> str:
    return f"batch_{batch_index:03d}"


def batched_review_rows(
    rows: list[dict[str, Any]], *, batch_size: int
) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("--llm-batch-size must be positive")
    return [
        rows[index : index + batch_size] for index in range(0, len(rows), batch_size)
    ]


def llm_batch_records(
    *,
    output_dir: Path,
    llm_blind_rows: list[dict[str, Any]],
    batch_size: int,
    repo_root: Path = ROOT,
    batch_root_dir: str = LLM_BATCH_ROOT_DIR,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for batch_index, rows in enumerate(
        batched_review_rows(llm_blind_rows, batch_size=batch_size),
        start=1,
    ):
        if not rows:
            continue
        start = int(rows[0]["review_order"])
        end = int(rows[-1]["review_order"])
        batch_id = batch_dir_name(batch_index)
        batch_dir = f"{batch_root_dir}/{batch_id}"
        case_file = f"{batch_dir}/review_cases_blind.jsonl"
        prompt_file = f"{batch_dir}/prompt.md"
        label_part_file = f"{batch_dir}/opus_4_7_labels.jsonl"
        records.append(
            {
                "batch_id": batch_id,
                "review_order_start": start,
                "review_order_end": end,
                "n_cases": len(rows),
                "case_file": format_path_for_metadata(
                    output_dir / case_file,
                    root=repo_root,
                ),
                "case_file_sha256": byte_sha256_for_rows(rows),
                "prompt_file": format_path_for_metadata(
                    output_dir / prompt_file,
                    root=repo_root,
                ),
                "label_part_file": format_path_for_metadata(
                    output_dir / label_part_file,
                    root=repo_root,
                ),
            }
        )
    return records


def human_label_schema_summary(
    *,
    schema_version: str = HUMAN_LABEL_SCHEMA_VERSION,
    valid_labels: Iterable[str] = VALID_LABELS,
    valid_flags: Iterable[str] = VALID_FLAGS,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "required_fields": list(HUMAN_LABEL_REQUIRED_FIELDS),
        "valid_labels": list(valid_labels),
        "valid_flags": list(valid_flags),
    }


def build_label_schema(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "simid_open_independent_rater_label.schema.json",
        "title": "SIMID Open Independent Rater Label",
        "type": "object",
        "additionalProperties": False,
        "required": manifest["label_schema"]["required_fields"],
        "properties": {
            "schema_version": {"const": HUMAN_LABEL_SCHEMA_VERSION},
            "calibration_case_id": {
                "type": "string",
                "pattern": "^simid_open_cal_[0-9a-f]+$",
            },
            "review_order": {"type": "integer", "minimum": 1},
            "label": {
                "type": "string",
                "enum": list(VALID_LABELS),
            },
            "confidence": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
            },
            "rule_gap": {"type": "boolean"},
            "flags": {
                "type": "array",
                "items": {"type": "string", "enum": list(VALID_FLAGS)},
                "uniqueItems": True,
            },
            "notes": {"type": "string"},
            "rater": {
                "type": "object",
                "additionalProperties": True,
                "required": ["type"],
                "properties": {
                    "type": {"type": "string"},
                    "id": {"type": "string"},
                    "model": {"type": "string"},
                    "prompt_version": {"type": "string"},
                },
            },
            "labeled_at_local": {"type": "string"},
            "blind_cases_file_sha256": {
                "type": "string",
                "const": manifest["blind_cases_file_sha256"],
            },
            "label_schema_version": {"const": HUMAN_LABEL_SCHEMA_VERSION},
        },
    }


def build_llm_batch_label_schema(
    *,
    batch_cases_file_sha256: str,
) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "simid_open_llm_blind_rater_label.schema.json",
        "title": "SIMID Open LLM Blind Batch Rater Label",
        "type": "object",
        "additionalProperties": False,
        "required": list(LLM_BATCH_LABEL_REQUIRED_FIELDS),
        "properties": {
            "schema_version": {"const": LLM_BATCH_LABEL_SCHEMA_VERSION},
            "blind_case_id": {
                "type": "string",
                "pattern": "^simid_open_blind_[0-9a-f]+$",
            },
            "review_order": {"type": "integer", "minimum": 1},
            "label": {
                "type": "string",
                "enum": list(VALID_LABELS),
            },
            "confidence": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
            },
            "rule_gap": {"type": "boolean"},
            "flags": {
                "type": "array",
                "items": {"type": "string", "enum": list(VALID_FLAGS)},
                "uniqueItems": True,
            },
            "notes": {"type": "string"},
            "rater": {
                "type": "object",
                "additionalProperties": True,
                "required": ["type", "model", "prompt_version"],
                "properties": {
                    "type": {"type": "string"},
                    "model": {"type": "string"},
                    "prompt_version": {"type": "string"},
                },
            },
            "blind_cases_file_sha256": {"const": batch_cases_file_sha256},
        },
    }


def ui_manifest_binding(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest["schema_version"],
        "blind_cases_file_sha256": manifest["blind_cases_file_sha256"],
        "label_schema": manifest["label_schema"],
    }


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
                "schema_version": HUMAN_LABEL_SCHEMA_VERSION,
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
