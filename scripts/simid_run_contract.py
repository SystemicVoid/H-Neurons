"""Shared SIMID run-family manifest and output contract validation."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
from typing import Any, cast

from artifact_io import alpha_jsonl_path, file_sha256
from utils import normalize_answer


ROOT = Path(__file__).resolve().parent.parent
SIMID_MANIFEST_SCHEMA_VERSION = "simid_manifest/v1"
SIMID_LOCKED_MANIFEST_SCHEMA_VERSION = "simid_locked_manifest/v1"
SIMID_RUN_CONFIG_SCHEMA_VERSION = "simid_run_config/v1"
SIMID_RESUME_CONTRACT_SCHEMA_VERSION = "simid_resume_contract/v1"
DEFAULT_TRUTHFULQA_LEAKAGE_POLICY = "heldout_only"
REQUIRED_ROW_FIELDS = {
    "sample_id",
    "dataset",
    "question",
    "open_prompt",
    "mc_options",
    "gold_option_indices",
    "gold_aliases",
    "distractor_provenance",
    "option_order_seed",
    "option_order_replicate",
    "model_path",
    "tokenizer_path",
    "iti_artifact_path",
    "iti_artifact_sha256",
}
TRUTHFULQA_LEAKAGE_METADATA_FIELDS = (
    "truthfulqa_artifact_split",
    "truthfulqa_seen_in_iti_fit",
    "truthfulqa_leakage_policy",
)
LOCKED_MANIFEST_METADATA_FIELDS = (
    "base_sample_id",
    "dataset",
    "mc_endpoint",
    "option_order_replicate",
    *TRUTHFULQA_LEAKAGE_METADATA_FIELDS,
)


def optional_file_sha256(path: str | Path) -> str | None:
    resolved = Path(path)
    if not resolved.exists() or not resolved.is_file():
        return None
    return file_sha256(resolved)


def stable_payload_sha256(payload: Any, *, ensure_ascii: bool = False) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=ensure_ascii,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append(row)
    return rows


def validate_manifest_row(
    row: dict[str, Any], *, field: str = "manifest.rows[]"
) -> None:
    missing = sorted(REQUIRED_ROW_FIELDS - set(row))
    if missing:
        raise ValueError(f"{field}: SIMID manifest row missing fields: {missing}")
    sample_id = row["sample_id"]
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"{field}.sample_id is empty")
    base_sample_id = row.get("base_sample_id")
    if base_sample_id is not None and (
        not isinstance(base_sample_id, str) or not base_sample_id
    ):
        raise ValueError(f"{field}.base_sample_id must be a non-empty string")
    mc_options = row["mc_options"]
    if not isinstance(mc_options, list) or len(mc_options) < 2:
        raise ValueError(f"{field}.mc_options must contain >=2 options")
    if len({normalize_answer(option) for option in mc_options}) != len(mc_options):
        raise ValueError(f"{field}.mc_options contain normalized duplicates")
    gold_indices = row["gold_option_indices"]
    if not isinstance(gold_indices, list) or not gold_indices:
        raise ValueError(f"{field}.gold_option_indices cannot be empty")
    if any(type(idx) is not int for idx in gold_indices):
        raise ValueError(f"{field}.gold_option_indices must contain integers")
    if any(idx < 0 or idx >= len(mc_options) for idx in gold_indices):
        raise ValueError(f"{field}.gold_option_indices contains out-of-range index")
    provenance = row["distractor_provenance"]
    if not isinstance(provenance, list) or len(provenance) != len(mc_options):
        raise ValueError(f"{field}.distractor_provenance must align with mc_options")
    replicate = row["option_order_replicate"]
    if not isinstance(replicate, int) or isinstance(replicate, bool):
        raise ValueError(f"{field}.option_order_replicate must be an integer")
    if replicate < 0:
        raise ValueError(f"{field}.option_order_replicate must be non-negative")
    if not isinstance(row["gold_aliases"], list) or not row["gold_aliases"]:
        raise ValueError(f"{field}.gold_aliases cannot be empty")


def validate_manifest(
    payload: dict[str, Any],
    *,
    field: str = "manifest",
) -> None:
    if payload.get("schema_version") != SIMID_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"{field}.schema_version expected {SIMID_MANIFEST_SCHEMA_VERSION!r}, "
            f"got {payload.get('schema_version')!r}"
        )
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{field}.rows must be a non-empty list")
    seen_sample_ids: set[str] = set()
    seen_base_replicates: set[tuple[str, int]] = set()
    for idx, row in enumerate(rows):
        row_field = f"{field}.rows[{idx}]"
        if not isinstance(row, dict):
            raise ValueError(f"{row_field} must be an object")
        row_dict = cast(dict[str, Any], row)
        validate_manifest_row(row_dict, field=row_field)
        sample_id = str(row_dict["sample_id"])
        if sample_id in seen_sample_ids:
            raise ValueError(f"{row_field}.sample_id duplicate: {sample_id}")
        seen_sample_ids.add(sample_id)
        base_sample_id = row_dict.get("base_sample_id")
        if not isinstance(base_sample_id, str) or not base_sample_id:
            continue
        replicate = int(row_dict["option_order_replicate"])
        base_key = (base_sample_id, replicate)
        if base_key in seen_base_replicates:
            raise ValueError(
                f"{row_field}.base_sample_id/option_order_replicate duplicate: "
                f"{base_sample_id!r}/{replicate}"
            )
        seen_base_replicates.add(base_key)


def load_simid_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
        payload = {"schema_version": SIMID_MANIFEST_SCHEMA_VERSION, "rows": rows}
    if not isinstance(payload, dict):
        raise ValueError(
            f"{manifest_path}: SIMID manifest must be an object or row list"
        )
    validate_manifest(payload, field=str(manifest_path))
    rows = payload["rows"]
    if not isinstance(rows, list):
        raise ValueError(f"{manifest_path}.rows must be a list")
    return [dict(row) for row in rows]


def _manifest_path(path_or_run_dir: Path) -> Path:
    if path_or_run_dir.name == "manifest.locked.json":
        return path_or_run_dir
    return path_or_run_dir / "manifest.locked.json"


def load_locked_manifest_payload(path_or_run_dir: Path) -> dict[str, Any]:
    path = _manifest_path(path_or_run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing SIMID locked manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected SIMID locked manifest object")
    schema_version = payload.get("schema_version")
    if schema_version != SIMID_LOCKED_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"{path}.schema_version expected {SIMID_LOCKED_MANIFEST_SCHEMA_VERSION!r}, "
            f"got {schema_version!r}"
        )
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}.rows must be a non-empty list")
    validate_manifest(
        {"schema_version": SIMID_MANIFEST_SCHEMA_VERSION, "rows": rows},
        field=str(path),
    )
    return payload


def load_locked_manifest(path_or_run_dir: Path) -> list[dict[str, Any]]:
    payload = load_locked_manifest_payload(path_or_run_dir)
    rows = payload["rows"]
    if not isinstance(rows, list):
        raise ValueError(f"{_manifest_path(path_or_run_dir)}.rows must be a list")
    return [dict(row) for row in rows]


def load_locked_manifest_metadata(
    run_dir: Path,
    *,
    fields: tuple[str, ...] = LOCKED_MANIFEST_METADATA_FIELDS,
) -> dict[str, dict[str, Any]]:
    path = run_dir / "manifest.locked.json"
    if not path.exists():
        return {}
    payload = load_locked_manifest_payload(path)
    rows = payload.get("rows", [])
    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or "sample_id" not in row:
            continue
        metadata[str(row["sample_id"])] = {field: row.get(field) for field in fields}
    return metadata


def load_or_create_locked_manifest(
    *,
    source_manifest: str,
    output_dir: Path,
    max_items: int | None,
) -> list[dict[str, Any]]:
    locked_manifest_path = output_dir / "manifest.locked.json"
    if locked_manifest_path.exists():
        return load_locked_manifest(locked_manifest_path)

    rows = load_simid_manifest(source_manifest)
    if max_items is not None:
        rows = rows[:max_items]
    if not rows:
        raise ValueError("No SIMID rows selected")
    locked_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": SIMID_LOCKED_MANIFEST_SCHEMA_VERSION,
                "source_manifest": source_manifest,
                "source_manifest_sha256": file_sha256(source_manifest),
                "max_items": max_items,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows


def load_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing SIMID run_config.json: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected SIMID run_config.json object")
    if payload.get("schema_version") != SIMID_RUN_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"{path}.schema_version expected {SIMID_RUN_CONFIG_SCHEMA_VERSION!r}, "
            f"got {payload.get('schema_version')!r}"
        )
    return payload


def run_config_condition_names(run_config: dict[str, Any]) -> list[str]:
    conditions = run_config.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("run_config.json.conditions must be a non-empty list")
    names: list[str] = []
    for idx, condition in enumerate(conditions):
        if isinstance(condition, dict):
            condition_dict = cast(dict[str, Any], condition)
            name = condition_dict.get("name")
        else:
            name = condition
        if not isinstance(name, str) or not name:
            raise ValueError(f"run_config.json.conditions[{idx}].name is invalid")
        names.append(name)
    return names


def run_config_alphas(run_config: dict[str, Any]) -> list[float]:
    alphas = run_config.get("alphas")
    if not isinstance(alphas, list) or not alphas:
        raise ValueError("run_config.json.alphas must be a non-empty list")
    parsed: list[float] = []
    for idx, alpha in enumerate(alphas):
        if not isinstance(alpha, str | int | float):
            raise ValueError(f"run_config.json.alphas[{idx}] must be numeric")
        value = float(alpha)
        if not math.isfinite(value):
            raise ValueError(f"run_config.json.alphas[{idx}] must be finite")
        parsed.append(value)
    return parsed


def validate_complete_run_outputs(run_dir: Path) -> None:
    """Refuse to analyze partial SIMID runs by default."""
    run_config = load_run_config(run_dir)
    manifest_rows = load_locked_manifest(run_dir / "manifest.locked.json")
    expected_ids: set[str] = set()
    for idx, row in enumerate(manifest_rows):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"manifest.locked.json.rows[{idx}].sample_id is invalid")
        expected_ids.add(sample_id)
    expected_n = int(run_config.get("n_rows") or len(expected_ids))
    if len(expected_ids) != expected_n:
        raise ValueError(
            "manifest.locked.json.rows sample_id count does not match "
            f"run_config.json.n_rows: {len(expected_ids)} vs {expected_n}"
        )

    problems: list[str] = []
    for condition in run_config_condition_names(run_config):
        for alpha in run_config_alphas(run_config):
            path = alpha_jsonl_path(run_dir / condition, alpha)
            if not path.exists():
                problems.append(f"missing {path}")
                continue
            rows = load_jsonl(path)
            ids: list[str] = []
            for row_idx, row in enumerate(rows):
                sample_id = row.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id:
                    problems.append(f"{path}:{row_idx + 1} invalid sample_id")
                    continue
                ids.append(sample_id)
            id_set = set(ids)
            if len(rows) != expected_n:
                problems.append(f"{path} has {len(rows)} rows, expected {expected_n}")
            if len(ids) != len(id_set):
                problems.append(f"{path} contains duplicate sample_id values")
            if id_set != expected_ids:
                missing = sorted(expected_ids - id_set)[:5]
                extra = sorted(id_set - expected_ids)[:5]
                problems.append(
                    f"{path} sample_id set does not match manifest.locked.json.rows "
                    f"(missing={missing}, extra={extra})"
                )
    if problems:
        preview = "; ".join(problems[:8])
        extra = "" if len(problems) <= 8 else f"; ... {len(problems) - 8} more"
        raise ValueError(
            "SIMID run outputs are incomplete or inconsistent; refusing analysis. "
            f"{preview}{extra}. Pass --allow-incomplete-run only for a clearly "
            "diagnostic partial analysis."
        )


def resolve_repo_metadata_path(path_value: Any) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("metadata path is missing")
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def validate_content_bound_path(entry: dict[str, Any], *, row_name: str) -> Path:
    path = resolve_repo_metadata_path(entry.get("path"))
    expected_hash = entry.get("content_sha256")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError(f"{row_name}: missing content_sha256")
    observed_hash = file_sha256(path)
    if observed_hash != expected_hash:
        raise ValueError(
            f"{row_name}: content_sha256 mismatch for {path} "
            f"(expected {expected_hash}, observed {observed_hash})"
        )
    return path


def load_exclusion_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    for row in load_jsonl(path):
        for key in ("sample_id", "base_sample_id"):
            value = row.get(key)
            if isinstance(value, str) and value:
                ids.add(value)
    return ids


def validate_run_manifest_excludes_ids(*, run_dir: Path, exclusion_file: Path) -> None:
    excluded = load_exclusion_ids(exclusion_file)
    if not excluded:
        raise ValueError("prospective open authority exclusion file is empty")
    overlapping: set[str] = set()
    for row in load_locked_manifest(run_dir / "manifest.locked.json"):
        for key in ("sample_id", "base_sample_id"):
            value = row.get(key)
            if isinstance(value, str) and value in excluded:
                overlapping.add(value)
    if overlapping:
        preview = sorted(overlapping)[:10]
        raise ValueError(
            "Prospective SIMID effect manifest reuses excluded historical MVP "
            f"or calibration sample IDs: {preview}"
        )


def resolved_paths_match(left: str, right: str) -> bool:
    return (
        resolve_repo_metadata_path(left).resolve()
        == resolve_repo_metadata_path(right).resolve()
    )


def _manifest_sample_ids(rows: list[Any], *, row_source: str) -> list[str]:
    sample_ids: list[str] = []
    seen: set[str] = set()
    for idx, row in enumerate(rows):
        sample_id = row.get("sample_id") if isinstance(row, dict) else None
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{row_source}.rows[{idx}].sample_id is invalid")
        if sample_id in seen:
            raise ValueError(
                f"{row_source}.rows[{idx}].sample_id duplicate: {sample_id}"
            )
        sample_ids.append(sample_id)
        seen.add(sample_id)
    return sample_ids


def validate_locked_manifest_source_binding(
    *,
    run_dir: Path,
    authority_manifest: dict[str, Any],
    authority_path: Path,
) -> None:
    planned = authority_manifest.get("planned_run")
    if not isinstance(planned, dict):
        raise ValueError(f"{authority_path}: missing planned_run")
    sample_design = planned.get("sample_design")
    if not isinstance(sample_design, dict):
        raise ValueError(f"{authority_path}: missing planned_run.sample_design")
    planned_manifest = sample_design.get("manifest_path")
    if not isinstance(planned_manifest, str) or not planned_manifest:
        raise ValueError(
            f"{authority_path}: missing planned_run.sample_design.manifest_path"
        )
    expected_manifest_hash = sample_design.get("manifest_sha256")
    if not isinstance(expected_manifest_hash, str) or not expected_manifest_hash:
        raise ValueError(
            f"{authority_path}: missing planned manifest hash "
            "(planned_run.sample_design.manifest_sha256)"
        )
    locked_payload = load_locked_manifest_payload(run_dir / "manifest.locked.json")
    source_manifest = locked_payload.get("source_manifest")
    source_manifest_sha256 = locked_payload.get("source_manifest_sha256")
    if not isinstance(source_manifest, str) or not source_manifest:
        raise ValueError("manifest.locked.json.source_manifest is not recorded")
    if not isinstance(source_manifest_sha256, str) or not source_manifest_sha256:
        raise ValueError("manifest.locked.json.source_manifest_sha256 is not recorded")
    if "max_items" not in locked_payload:
        raise ValueError("manifest.locked.json.max_items is not recorded")
    if locked_payload["max_items"] is not None:
        raise ValueError(
            "manifest.locked.json.max_items is set; prospective open authority "
            "requires the full planned manifest"
        )
    if not resolved_paths_match(source_manifest, planned_manifest):
        raise ValueError(
            "manifest.locked.json.source_manifest was not built from "
            f"planned_run.sample_design.manifest_path {planned_manifest}"
        )
    source_manifest_path = resolve_repo_metadata_path(source_manifest)
    observed_hash = file_sha256(source_manifest_path)
    if observed_hash != source_manifest_sha256:
        raise ValueError(
            "manifest.locked.json.source_manifest_sha256 does not match "
            f"manifest file {source_manifest_path}"
        )
    if expected_manifest_hash != source_manifest_sha256:
        raise ValueError(
            "manifest.locked.json.source_manifest_sha256 does not match "
            "planned_run.sample_design.manifest_sha256"
        )
    planned_rows = load_simid_manifest(source_manifest_path)
    locked_rows = locked_payload.get("rows")
    if not isinstance(locked_rows, list) or not locked_rows:
        raise ValueError("manifest.locked.json.rows are not recorded")
    planned_sample_ids = _manifest_sample_ids(
        planned_rows,
        row_source=f"planned manifest {source_manifest_path}",
    )
    locked_sample_ids = _manifest_sample_ids(
        locked_rows,
        row_source="manifest.locked.json",
    )
    if locked_sample_ids != planned_sample_ids:
        missing = sorted(set(planned_sample_ids) - set(locked_sample_ids))[:5]
        extra = sorted(set(locked_sample_ids) - set(planned_sample_ids))[:5]
        raise ValueError(
            "manifest.locked.json.rows sample_ids do not match the full planned "
            f"manifest (locked={len(locked_sample_ids)}, "
            f"planned={len(planned_sample_ids)}, missing={missing}, extra={extra})"
        )
    planned_by_id = {
        sample_id: row
        for sample_id, row in zip(planned_sample_ids, planned_rows, strict=True)
    }
    locked_by_id = {
        sample_id: row
        for sample_id, row in zip(locked_sample_ids, locked_rows, strict=True)
    }
    mismatched = [
        sample_id
        for sample_id in planned_sample_ids
        if locked_by_id[sample_id] != planned_by_id[sample_id]
    ]
    if mismatched:
        raise ValueError(
            "manifest.locked.json.rows content does not match the full planned "
            f"manifest for sample_ids {mismatched[:5]}"
        )


def validate_locked_manifest_heldout_only(*, run_dir: Path) -> None:
    for idx, row in enumerate(load_locked_manifest(run_dir / "manifest.locked.json")):
        if row.get("dataset") != "truthfulqa":
            continue
        if row.get("truthfulqa_leakage_policy") != DEFAULT_TRUTHFULQA_LEAKAGE_POLICY:
            raise ValueError(
                "manifest.locked.json.rows"
                f"[{idx}].truthfulqa_leakage_policy is not heldout_only"
            )
        if row.get("truthfulqa_seen_in_iti_fit") is not False:
            raise ValueError(
                "manifest.locked.json.rows"
                f"[{idx}].truthfulqa_seen_in_iti_fit must be false"
            )


def validate_exclusion_provenance(
    *,
    authority_manifest: dict[str, Any],
    authority_path: Path,
    exclusion_path: Path,
) -> None:
    planned = authority_manifest.get("planned_run")
    sample_design = planned.get("sample_design") if isinstance(planned, dict) else None
    if not isinstance(sample_design, dict):
        raise ValueError(f"{authority_path}: missing planned_run.sample_design")
    exclusion_file = sample_design.get("exclusion_file")
    if not isinstance(exclusion_file, str) or not exclusion_file:
        raise ValueError(f"{authority_path}: missing sample_design.exclusion_file")
    if not resolved_paths_match(exclusion_file, str(exclusion_path)):
        raise ValueError(
            f"{authority_path}: sample_design.exclusion_file does not match "
            "exclusions.path"
        )
    rows = load_jsonl(exclusion_path)
    sources: set[str] = set()
    for row in rows:
        source = row.get("source")
        if isinstance(source, str) and source:
            sources.add(source)
        if row.get("also_in_prospective_open_calibration_sample") is True:
            sources.add("prospective_open_calibration_private_case_map")
    required_sources = {
        "historical_mvp_locked_manifest",
        "prospective_open_calibration_private_case_map",
    }
    missing = required_sources - sources
    if missing:
        raise ValueError(
            f"{authority_path}: exclusions missing expected provenance sources "
            f"{sorted(missing)}"
        )


def build_planned_sample_design(
    *,
    manifest_path: str,
    manifest_sha256: str,
    manifest_builder: str,
    seed: int,
    truthfulqa_n: str | int | None,
    min_truthfulqa_rows: int,
    bridge_n: int,
    min_bridge_rows: int,
    option_order_replicates: int,
    exclusion_file: str,
    freshness_policy: str,
) -> dict[str, Any]:
    if not manifest_path:
        raise ValueError("planned_run.sample_design.manifest_path is required")
    if not manifest_sha256:
        raise ValueError("planned_run.sample_design.manifest_sha256 is required")
    if option_order_replicates < 1:
        raise ValueError(
            "planned_run.sample_design.option_order_replicates must be >= 1"
        )
    if min_truthfulqa_rows < 0 or min_bridge_rows < 0:
        raise ValueError("planned_run.sample_design minimum rows must be non-negative")
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "manifest_builder": manifest_builder,
        "seed": int(seed),
        "truthfulqa": {
            "n": truthfulqa_n,
            "leakage_policy": DEFAULT_TRUTHFULQA_LEAKAGE_POLICY,
            "minimum_rows": int(min_truthfulqa_rows),
        },
        "triviaqa_bridge": {
            "n": int(bridge_n),
            "minimum_rows": int(min_bridge_rows),
        },
        "option_order_replicates": int(option_order_replicates),
        "exclusion_file": exclusion_file,
        "freshness_policy": freshness_policy,
    }
