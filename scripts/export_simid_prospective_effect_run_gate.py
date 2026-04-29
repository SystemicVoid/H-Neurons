#!/usr/bin/env python3
"""Export a prospective SIMID effect-run protocol gate."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
from typing import Any

from utils import (
    finish_run_provenance,
    format_path_for_metadata,
    get_git_dirty,
    get_git_sha,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)


ROOT = Path(__file__).resolve().parent.parent
BASE_SIMID_RUN_ROOT = (
    ROOT / "data/gemma3_4b/intervention/"
    "simid_iti_truthfulqa-paperfaithful_k12_first-3-tokens"
)
DEFAULT_SOURCE_MVP_RUN_DIR = BASE_SIMID_RUN_ROOT / "mvp_20260427_calibration"
DEFAULT_SOURCE_HUMAN_REVIEW_DIR = DEFAULT_SOURCE_MVP_RUN_DIR / "human_review_package"
DEFAULT_OPEN_GATE_DIR = (
    DEFAULT_SOURCE_HUMAN_REVIEW_DIR / "prospective_open_calibration_gate_20260429"
)
DEFAULT_PASSING_OPEN_ANALYSIS = (
    DEFAULT_OPEN_GATE_DIR
    / "prospective_open_calibration_analysis_opus_4_7_max_run_001.json"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_SOURCE_HUMAN_REVIEW_DIR
    / "prospective_effect_run_gate_20260429_r2_external_labels"
)
DEFAULT_PLANNED_MANIFEST = (
    ROOT / "data/manifests/simid_truthfulqa_bridge_prospective_effect_20260429.json"
)
DEFAULT_PLANNED_NOOP_RUN_DIR = (
    BASE_SIMID_RUN_ROOT / "prospective_effect_noop_preflight_20260429"
)
DEFAULT_PLANNED_EFFECT_RUN_DIR = (
    BASE_SIMID_RUN_ROOT / "prospective_effect_calibrated_open_20260429"
)

PACKAGE_SCHEMA_VERSION = "simid_prospective_effect_run_gate/v2"
PROTOCOL_VERSION = "simid_prospective_effect_run_protocol_20260429_r2/v1"
EXTERNAL_LABEL_PACKAGE_SCHEMA_VERSION = "simid_prospective_effect_open_label_package/v1"
EXTERNAL_LABEL_SCHEMA_VERSION = "simid_prospective_effect_open_label/v1"
OPEN_GATE_ANALYSIS_SCHEMA_VERSION = "simid_prospective_open_calibration_analysis/v1"
OPEN_GATE_TARGET = "simid_open_correctness_future_grading"
PRIMARY_ALPHA = 8.0
BASELINE_ALPHA = 0.0
ALPHA_GRID = [-8.0, 0.0, 4.0, 8.0]
OPTION_ORDER_REPLICATES = 2
MIN_TRUTHFULQA_ROWS = 450
MIN_BRIDGE_ROWS = 400
PLANNED_BRIDGE_N = 200
RANDOM_CONTROL_SEEDS = (1, 2, 3, 4, 5)
RANDOM_HEAD_CONTROLS = tuple(f"random_head_seed{seed}" for seed in RANDOM_CONTROL_SEEDS)
RANDOM_DIRECTION_CONTROLS = tuple(
    f"random_direction_seed{seed}" for seed in RANDOM_CONTROL_SEEDS
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-mvp-run-dir", type=Path, default=DEFAULT_SOURCE_MVP_RUN_DIR
    )
    parser.add_argument("--open-gate-dir", type=Path, default=DEFAULT_OPEN_GATE_DIR)
    parser.add_argument(
        "--passing-open-analysis",
        type=Path,
        default=DEFAULT_PASSING_OPEN_ANALYSIS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--planned-manifest", type=Path, default=DEFAULT_PLANNED_MANIFEST
    )
    parser.add_argument(
        "--planned-noop-run-dir",
        type=Path,
        default=DEFAULT_PLANNED_NOOP_RUN_DIR,
    )
    parser.add_argument(
        "--planned-effect-run-dir",
        type=Path,
        default=DEFAULT_PLANNED_EFFECT_RUN_DIR,
    )
    parser.add_argument(
        "--planned-truthfulqa-n",
        default="all_heldout",
        help=(
            "Use all held-out TruthfulQA rows by default. Pass an integer string "
            "only for an explicitly resized prospective run."
        ),
    )
    parser.add_argument("--planned-bridge-n", type=int, default=PLANNED_BRIDGE_N)
    parser.add_argument(
        "--option-order-replicates",
        type=int,
        default=OPTION_ORDER_REPLICATES,
    )
    parser.add_argument("--min-truthfulqa-rows", type=int, default=MIN_TRUTHFULQA_ROWS)
    parser.add_argument("--min-bridge-rows", type=int, default=MIN_BRIDGE_ROWS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files in an existing prospective effect gate directory.",
    )
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def relpath(path: Path) -> str:
    return format_path_for_metadata(path, root=ROOT)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def file_metadata(path: Path, *, schema_version: str | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "path": relpath(path),
        "content_sha256": file_sha256(path),
    }
    if schema_version is not None:
        metadata["schema_version"] = schema_version
    return metadata


def manifest_bound_file(manifest: dict[str, Any], key: str) -> dict[str, Any]:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("open gate review_manifest.json has no files object")
    entry = files.get(key)
    if not isinstance(entry, dict):
        raise ValueError(f"open gate review_manifest.json has no files.{key}")
    path_value = entry.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"open gate files.{key} has no path")
    path = Path(path_value)
    resolved = path if path.is_absolute() else ROOT / path
    expected_hash = entry.get("content_sha256")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError(f"open gate files.{key} has no content_sha256")
    observed_hash = file_sha256(resolved)
    if observed_hash != expected_hash:
        raise ValueError(
            f"open gate files.{key} hash mismatch: expected {expected_hash}, "
            f"observed {observed_hash}"
        )
    return {
        "path": relpath(resolved),
        "content_sha256": observed_hash,
        "schema_version": entry.get("schema_version"),
    }


def validate_truthfulqa_n(value: str) -> str:
    if value == "all_heldout":
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(
            "--planned-truthfulqa-n must be all_heldout or an integer string"
        ) from exc
    if parsed <= 0:
        raise ValueError("--planned-truthfulqa-n must be positive")
    return str(parsed)


def validate_export_args(args: argparse.Namespace) -> None:
    validate_truthfulqa_n(str(args.planned_truthfulqa_n))
    if args.planned_bridge_n <= 0:
        raise ValueError("--planned-bridge-n must be positive")
    if args.option_order_replicates < 1:
        raise ValueError("--option-order-replicates must be >= 1")
    if args.min_truthfulqa_rows < 1 or args.min_bridge_rows < 1:
        raise ValueError("minimum row counts must be positive")


def validate_open_gate_analysis(analysis: dict[str, Any], *, path: Path) -> None:
    if analysis.get("schema_version") != OPEN_GATE_ANALYSIS_SCHEMA_VERSION:
        raise ValueError(f"{path}: unexpected schema_version")
    outcome = analysis.get("outcome")
    if not isinstance(outcome, dict) or outcome.get("passed") is not True:
        raise ValueError(f"{path}: prospective open calibration gate did not pass")
    blockers = outcome.get("blockers")
    if blockers not in ([], None):
        raise ValueError(f"{path}: prospective open calibration has blockers")
    policy = analysis.get("policy")
    if not isinstance(policy, dict) or policy.get("target") != OPEN_GATE_TARGET:
        raise ValueError(f"{path}: prospective open calibration target mismatch")
    metrics = analysis.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{path}: missing metrics")
    if int(metrics.get("rule_gap_count", -1)) != 0:
        raise ValueError(f"{path}: rule gaps are not allowed for this authority")
    raw_agreement = metrics.get("raw_agreement")
    if not isinstance(raw_agreement, dict):
        raise ValueError(f"{path}: missing raw agreement")
    checks = {
        "raw_agreement": (
            float(raw_agreement.get("estimate", -1.0)),
            float(policy.get("min_raw_agreement", 1.0)),
        ),
        "cohen_kappa": (
            float(metrics.get("cohen_kappa", -1.0)),
            float(policy.get("min_cohen_kappa", 1.0)),
        ),
        "gwet_ac1": (
            float(metrics.get("gwet_ac1", -1.0)),
            float(policy.get("min_gwet_ac1", 1.0)),
        ),
    }
    for name, (observed, threshold) in checks.items():
        if observed < threshold:
            raise ValueError(
                f"{path}: {name}={observed} is below frozen threshold {threshold}"
            )
    if int(analysis.get("n_cases", 0)) < int(policy.get("min_cases", 0)):
        raise ValueError(f"{path}: n_cases below frozen minimum")

    label_file_value = analysis.get("label_file")
    if not isinstance(label_file_value, str) or not label_file_value:
        raise ValueError(f"{path}: missing label_file")
    label_path = Path(label_file_value)
    resolved_label_path = label_path if label_path.is_absolute() else ROOT / label_path
    observed_label_hash = file_sha256(resolved_label_path)
    if observed_label_hash != analysis.get("label_file_sha256"):
        raise ValueError(f"{path}: label_file_sha256 mismatch")


def validate_analysis_package_binding(
    analysis: dict[str, Any],
    *,
    open_gate_dir: Path,
    path: Path,
) -> None:
    package_dir_value = analysis.get("package_dir")
    if not isinstance(package_dir_value, str) or not package_dir_value:
        raise ValueError(f"{path}: missing package_dir")
    package_dir = Path(package_dir_value)
    if not package_dir.is_absolute():
        package_dir = ROOT / package_dir
    if package_dir.resolve() != open_gate_dir.resolve():
        raise ValueError(
            f"{path}: package_dir {package_dir} does not match open gate "
            f"directory {open_gate_dir}"
        )


def build_open_grading_authority(
    *,
    open_gate_dir: Path,
    passing_open_analysis: Path,
) -> dict[str, Any]:
    analysis = read_json(passing_open_analysis)
    validate_open_gate_analysis(analysis, path=passing_open_analysis)
    validate_analysis_package_binding(
        analysis,
        open_gate_dir=open_gate_dir,
        path=passing_open_analysis,
    )
    review_manifest_path = open_gate_dir / "review_manifest.json"
    review_manifest = read_json(review_manifest_path)
    if (
        review_manifest.get("schema_version")
        != "simid_prospective_open_calibration_gate/v1"
    ):
        raise ValueError(f"{review_manifest_path}: unexpected schema_version")
    rubric = review_manifest.get("rubric")
    if not isinstance(rubric, dict):
        raise ValueError(f"{review_manifest_path}: missing rubric object")
    rubric_path_value = rubric.get("path")
    if not isinstance(rubric_path_value, str) or not rubric_path_value:
        raise ValueError(f"{review_manifest_path}: missing rubric path")
    rubric_path = Path(rubric_path_value)
    resolved_rubric_path = (
        rubric_path if rubric_path.is_absolute() else ROOT / rubric_path
    )
    observed_rubric_hash = file_sha256(resolved_rubric_path)
    if observed_rubric_hash != rubric.get("content_sha256"):
        raise ValueError(f"{review_manifest_path}: rubric hash mismatch")

    label_file_path = Path(str(analysis["label_file"]))
    if not label_file_path.is_absolute():
        label_file_path = ROOT / label_file_path

    return {
        "authority_id": "prospective_open_calibration_gate_20260429_opus_4_7_max_run_001",
        "authority_scope": (
            "future_simid_open_grading_under_frozen_rubric_with_external_blind_"
            "labels_only"
        ),
        "requires_external_returned_labels": True,
        "diagnostic_only_judges": [
            {
                "judge_model": "gpt-4o",
                "reason": (
                    "Automated adjudication can be recorded for diagnostics and "
                    "queue construction, but it cannot satisfy the prospective "
                    "open-correctness claim contract."
                ),
            }
        ],
        "not_authority_for": [
            "historical_mvp_open_correctness_claimability",
            "retrospective_endpoint_tuning",
            "rubric_revision_without_a_fresh_blinded_sample",
            "gpt_4o_or_other_automated_adjudication_as_claim_bearing_labels",
        ],
        "gate_dir": {
            "path": relpath(open_gate_dir),
        },
        "review_manifest": file_metadata(
            review_manifest_path,
            schema_version=str(review_manifest.get("schema_version")),
        ),
        "rubric": {
            "path": relpath(resolved_rubric_path),
            "content_sha256": observed_rubric_hash,
            "version": rubric.get("version"),
            "hard_case_groups": rubric.get("hard_case_groups"),
        },
        "review_cases_blind": manifest_bound_file(
            review_manifest, "review_cases_blind"
        ),
        "label_schema": manifest_bound_file(review_manifest, "label_schema"),
        "passing_analysis": file_metadata(
            passing_open_analysis,
            schema_version=str(analysis.get("schema_version")),
        ),
        "passing_label_file": {
            "path": relpath(label_file_path),
            "content_sha256": file_sha256(label_file_path),
        },
        "metrics": analysis["metrics"],
        "policy": analysis["policy"],
        "claimability_guardrail": (
            "This authority can support a future SIMID open-grading endpoint only "
            "when a fresh prospective effect run records this exact path/hash "
            "binding and attaches complete external blind labels returned under "
            "the frozen rubric. It does not upgrade the historical MVP run, and "
            "it does not make gpt-4o automated adjudication claim-bearing."
        ),
    }


def shell_command(tokens: list[str | os.PathLike[str] | float | int]) -> str:
    return shlex.join(str(token) for token in tokens)


def command_record(
    tokens: list[str | os.PathLike[str] | float | int],
) -> dict[str, Any]:
    return {
        "argv": [str(token) for token in tokens],
        "rendered": shell_command(tokens),
    }


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


def build_exclusion_rows(
    *,
    source_mvp_run_dir: Path,
    open_gate_dir: Path,
) -> list[dict[str, Any]]:
    locked_manifest = read_json(source_mvp_run_dir / "manifest.locked.json")
    rows = locked_manifest.get("rows")
    if not isinstance(rows, list):
        raise ValueError("source MVP locked manifest has no rows list")

    exclusion_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sample_id = row.get("sample_id")
        base_sample_id = row.get("base_sample_id") or sample_id
        if not isinstance(sample_id, str) or not isinstance(base_sample_id, str):
            continue
        exclusion_rows[(base_sample_id, sample_id)] = {
            "schema_version": "simid_prospective_effect_exclusion/v1",
            "source": "historical_mvp_locked_manifest",
            "sample_id": sample_id,
            "base_sample_id": base_sample_id,
            "dataset": row.get("dataset"),
        }

    for row in load_jsonl(open_gate_dir / "private_case_map.jsonl"):
        sample_id = row.get("sample_id")
        base_sample_id = row.get("base_sample_id") or sample_id
        if not isinstance(sample_id, str) or not isinstance(base_sample_id, str):
            continue
        key = (base_sample_id, sample_id)
        if key in exclusion_rows:
            exclusion_rows[key]["also_in_prospective_open_calibration_sample"] = True
            continue
        exclusion_rows[key] = {
            "schema_version": "simid_prospective_effect_exclusion/v1",
            "source": "prospective_open_calibration_private_case_map",
            "sample_id": sample_id,
            "base_sample_id": base_sample_id,
            "dataset": row.get("dataset"),
        }
    return [exclusion_rows[key] for key in sorted(exclusion_rows)]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def jsonl_text(rows: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )


def generated_file_metadata(
    path: Path,
    *,
    content: str,
    schema_version: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "path": relpath(path),
        "content_sha256": text_sha256(content),
    }
    if schema_version is not None:
        metadata["schema_version"] = schema_version
    return metadata


def planned_manifest_command(
    *,
    args: argparse.Namespace,
    source_run_config: dict[str, Any],
) -> dict[str, Any]:
    iti_config = source_run_config["iti_config"]
    tokens: list[str | os.PathLike[str] | float | int] = [
        "uv",
        "run",
        "python",
        "scripts/build_simid_manifest.py",
        "--seed",
        args.seed,
        "--truthfulqa-leakage-policy",
        "heldout_only",
        "--option-order-replicates",
        args.option_order_replicates,
        "--min-truthfulqa-rows",
        args.min_truthfulqa_rows,
        "--min-bridge-rows",
        args.min_bridge_rows,
        "--exclude-sample-ids-jsonl",
        relpath(args.output_dir / "excluded_effect_sample_ids.jsonl"),
        "--model-path",
        str(iti_config["model_path"]),
        "--iti-artifact-path",
        str(iti_config["iti_artifact_path"]),
        "--bridge-n",
        args.planned_bridge_n,
        "--output",
        relpath(args.planned_manifest),
    ]
    truthfulqa_n = validate_truthfulqa_n(str(args.planned_truthfulqa_n))
    if truthfulqa_n != "all_heldout":
        tokens.extend(["--truthfulqa-n", truthfulqa_n])
    return command_record(tokens)


def planned_noop_command(
    *,
    args: argparse.Namespace,
    source_run_config: dict[str, Any],
) -> dict[str, Any]:
    iti_config = source_run_config["iti_config"]
    tokens: list[str | os.PathLike[str] | float | int] = [
        "uv",
        "run",
        "python",
        "scripts/run_simid.py",
        "--manifest",
        relpath(args.planned_manifest),
        "--output-dir",
        relpath(args.planned_noop_run_dir),
        "--model-path",
        str(iti_config["model_path"]),
        "--iti-artifact-path",
        str(iti_config["iti_artifact_path"]),
        "--iti-family",
        str(iti_config["family"]),
        "--iti-k",
        int(iti_config["k"]),
        "--decode-scope",
        str(iti_config["decode_scope"]),
        "--device-map",
        "cuda:0",
        "--alphas",
        BASELINE_ALPHA,
        "--conditions",
        "selected",
        "--include-unhooked",
        "--noop-check",
        "--top-k-first-token",
        10,
        "--mc-max-new-tokens",
        4,
        "--open-max-new-tokens",
        64,
    ]
    return command_record(tokens)


def planned_effect_command(
    *,
    args: argparse.Namespace,
    source_run_config: dict[str, Any],
) -> dict[str, Any]:
    iti_config = source_run_config["iti_config"]
    tokens: list[str | os.PathLike[str] | float | int] = [
        "uv",
        "run",
        "python",
        "scripts/run_simid.py",
        "--manifest",
        relpath(args.planned_manifest),
        "--output-dir",
        relpath(args.planned_effect_run_dir),
        "--model-path",
        str(iti_config["model_path"]),
        "--iti-artifact-path",
        str(iti_config["iti_artifact_path"]),
        "--iti-family",
        str(iti_config["family"]),
        "--iti-k",
        int(iti_config["k"]),
        "--decode-scope",
        str(iti_config["decode_scope"]),
        "--device-map",
        "cuda:0",
        "--alphas",
        *ALPHA_GRID,
        "--conditions",
        "selected",
        *RANDOM_HEAD_CONTROLS,
        *RANDOM_DIRECTION_CONTROLS,
        "--top-k-first-token",
        10,
        "--mc-max-new-tokens",
        4,
        "--open-max-new-tokens",
        64,
    ]
    return command_record(tokens)


def planned_analysis_commands(
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    noop_tokens: list[str | os.PathLike[str] | float | int] = [
        "uv",
        "run",
        "python",
        "scripts/analyze_simid.py",
        "--run-dir",
        relpath(args.planned_noop_run_dir),
        "--conditions",
        "unhooked",
        "selected",
        "--alphas",
        BASELINE_ALPHA,
        "--phase0-gates",
        "--n-resamples",
        2000,
        "--output-json",
        relpath(args.planned_noop_run_dir / "results_noop_preflight.json"),
        "--report-md",
        relpath(args.planned_noop_run_dir / "report_noop_preflight.md"),
    ]
    diagnostic_tokens: list[str | os.PathLike[str] | float | int] = [
        "uv",
        "run",
        "python",
        "scripts/analyze_simid.py",
        "--run-dir",
        relpath(args.planned_effect_run_dir),
        "--adjudicate-open",
        "--adjudication-mode",
        "batch",
        "--phase0-gates",
        "--judge-model",
        "gpt-4o",
        "--adjudication-output",
        "open_adjudication_diagnostic_gpt4o.jsonl",
        "--output-json",
        relpath(args.planned_effect_run_dir / "results_gpt4o_diagnostic.json"),
        "--report-md",
        relpath(args.planned_effect_run_dir / "report_gpt4o_diagnostic.md"),
        "--alias-audit-output",
        relpath(
            args.planned_effect_run_dir / "alias_audit_queue_gpt4o_diagnostic.jsonl"
        ),
    ]
    label_package_dir = args.planned_effect_run_dir / "external_open_label_package"
    claim_tokens: list[str | os.PathLike[str] | float | int] = [
        "uv",
        "run",
        "python",
        "scripts/analyze_simid.py",
        "--run-dir",
        relpath(args.planned_effect_run_dir),
        "--phase0-gates",
        "--judge-model",
        "gpt-4o",
        "--adjudication-output",
        "open_adjudication_diagnostic_gpt4o.jsonl",
        "--output-json",
        relpath(args.planned_effect_run_dir / "results_external_open_labels.json"),
        "--report-md",
        relpath(args.planned_effect_run_dir / "report_external_open_labels.md"),
        "--alias-audit-output",
        relpath(
            args.planned_effect_run_dir / "alias_audit_queue_external_labels.jsonl"
        ),
        "--prospective-open-authority-manifest",
        relpath(args.output_dir / "effect_run_manifest.json"),
        "--prospective-open-label-package",
        relpath(label_package_dir),
        "--prospective-open-labels",
        relpath(label_package_dir / "prospective_open_labels_external.jsonl"),
        "--require-prospective-open-authority",
    ]
    return {
        "noop_preflight_analysis": command_record(noop_tokens),
        "effect_diagnostic_gpt4o_analysis": command_record(diagnostic_tokens),
        "effect_external_label_analysis": command_record(claim_tokens),
        "claimability_note": (
            "Open correctness is claim-bearing only if analyze_simid.py records "
            "the prospective_open_authority binding from this r2 package, "
            "validates complete external blind labels from the bound label "
            "package, and all other open-grading blockers are absent. gpt-4o "
            "adjudication remains diagnostic-only."
        ),
    }


def build_planned_run(
    *,
    args: argparse.Namespace,
    source_run_config: dict[str, Any],
) -> dict[str, Any]:
    iti_config = source_run_config["iti_config"]
    return {
        "selected_iti_condition": {
            "name": "selected",
            "family": iti_config["family"],
            "effective_family": iti_config.get("effective_family"),
            "k": int(iti_config["k"]),
            "decode_scope": iti_config["decode_scope"],
            "iti_artifact_path": iti_config["iti_artifact_path"],
            "iti_artifact_sha256": iti_config["iti_artifact_sha256"],
            "selection_strategy": "ranked",
            "direction_mode": "artifact",
        },
        "controls": [
            {
                "name": "unhooked_noop_preflight",
                "control_type": "unhooked",
                "purpose": "prove selected alpha=0 is equivalent to no installed hook",
                "run_dir": relpath(args.planned_noop_run_dir),
            },
            *[
                {
                    "name": control,
                    "control_type": "selected_heads_random_direction",
                    "purpose": (
                        "control for injecting a same-norm random direction "
                        "on selected heads"
                    ),
                    "seed": seed,
                    "run_dir": relpath(args.planned_effect_run_dir),
                }
                for seed, control in zip(
                    RANDOM_CONTROL_SEEDS,
                    RANDOM_DIRECTION_CONTROLS,
                    strict=True,
                )
            ],
            *[
                {
                    "name": control,
                    "control_type": "layer_matched_random_head",
                    "purpose": "control for choosing layer-matched nonselected heads",
                    "seed": seed,
                    "run_dir": relpath(args.planned_effect_run_dir),
                }
                for seed, control in zip(
                    RANDOM_CONTROL_SEEDS,
                    RANDOM_HEAD_CONTROLS,
                    strict=True,
                )
            ],
        ],
        "sample_design": {
            "manifest_path": relpath(args.planned_manifest),
            "manifest_sha256": file_sha256(args.planned_manifest),
            "manifest_builder": "scripts/build_simid_manifest.py",
            "seed": int(args.seed),
            "truthfulqa": {
                "n": args.planned_truthfulqa_n,
                "leakage_policy": "heldout_only",
                "minimum_rows": int(args.min_truthfulqa_rows),
            },
            "triviaqa_bridge": {
                "n": int(args.planned_bridge_n),
                "minimum_rows": int(args.min_bridge_rows),
            },
            "option_order_replicates": int(args.option_order_replicates),
            "exclusion_file": relpath(
                args.output_dir / "excluded_effect_sample_ids.jsonl"
            ),
            "freshness_policy": (
                "Exclude historical MVP locked-manifest sample/base IDs and "
                "all private prospective-open-calibration sample/base IDs before "
                "applying dataset caps."
            ),
        },
        "alpha_grid": {
            "values": ALPHA_GRID,
            "baseline_alpha": BASELINE_ALPHA,
            "primary_alpha": PRIMARY_ALPHA,
            "curve_summary": (
                "Pre-specify selected-vs-control curve summaries across the full "
                "grid; do not choose alpha post hoc after seeing future outcomes."
            ),
        },
        "primary_estimand": {
            "name": "truthfulqa_paired_open_correctness_delta_alpha8_vs_alpha0",
            "dataset": "truthfulqa",
            "condition": "selected",
            "metric": ("external_blind_open_correct_under_frozen_prospective_rubric"),
            "comparison": {
                "treatment_alpha": PRIMARY_ALPHA,
                "baseline_alpha": BASELINE_ALPHA,
            },
            "unit": "base_sample_id_with_option_order_replicates_averaged",
            "uncertainty": "paired bootstrap 95% CI at base-item level",
            "minimum_practical_effect_pp": 5.0,
        },
        "external_open_label_package": {
            "schema_version": EXTERNAL_LABEL_PACKAGE_SCHEMA_VERSION,
            "label_schema_version": EXTERNAL_LABEL_SCHEMA_VERSION,
            "package_dir": relpath(
                args.planned_effect_run_dir / "external_open_label_package"
            ),
            "labels_file": relpath(
                args.planned_effect_run_dir
                / "external_open_label_package"
                / "prospective_open_labels_external.jsonl"
            ),
            "blind_review_rows_public_fields": [
                "blind_case_id",
                "review_order",
                "question",
                "gold_aliases",
                "predicted_answer",
            ],
            "private_map_required_bindings": [
                "blind_case_id",
                "sample_id",
                "base_sample_id",
                "condition",
                "alpha",
                "dataset",
                "run_row_sha256",
                "authority_manifest_sha256",
            ],
            "required_label_bindings": [
                "blind_cases_file_sha256",
                "rubric_sha256",
                "external_rater",
            ],
            "claimability_requirement": (
                "Complete external returned labels are required for all open "
                "eligible prospective effect rows. Automated gpt-4o adjudication "
                "cannot satisfy this requirement."
            ),
        },
        "secondary_endpoints": [
            {
                "metric": "mc_letter_likelihood_correct",
                "role": "target-compatible MC accuracy check",
                "claim_scope": "secondary",
            },
            {
                "metric": "adjudicated_open_attempted",
                "role": "attempt/refusal or non-answer behavior proxy",
                "claim_scope": "secondary",
            },
            {
                "metric": "open_first3_margin",
                "role": "mechanistic-direction margin diagnostic",
                "claim_scope": "diagnostic",
            },
            {
                "metric": "triviaqa_bridge_adjudicated_open_correct",
                "role": "cross-dataset externality/specificity check",
                "claim_scope": "secondary",
            },
        ],
        "commands": {
            "build_manifest": planned_manifest_command(
                args=args,
                source_run_config=source_run_config,
            ),
            "noop_preflight_run": planned_noop_command(
                args=args,
                source_run_config=source_run_config,
            ),
            "effect_run": planned_effect_command(
                args=args,
                source_run_config=source_run_config,
            ),
            **planned_analysis_commands(args=args),
        },
    }


def build_claimability_gates() -> dict[str, Any]:
    return {
        "pre_run_blockers": [
            "passing_prospective_open_grading_authority_not_bound_by_path_and_hash",
            "truthfulqa_heldout_only_manifest_not_confirmed",
            "planned_output_paths_already_contain_committed_or_analyzed_outputs",
            "noop_preflight_not_completed",
        ],
        "analysis_blockers": [
            "external_open_labels_not_loaded_for_all_effect_rows",
            "external_open_label_package_not_bound_to_authority_hash",
            "external_open_label_package_leaks_private_keys_in_blind_rows",
            "gpt4o_adjudication_used_as_claim_bearing_open_labels",
            "mixed_adjudication_and_deterministic_open_grade_sources",
            "unknown_or_error_open_judge_verdicts",
            "prospective_open_authority_not_recorded_in_results",
            "prospective_open_external_labels_not_recorded_in_results",
            "locked_manifest_source_manifest_sha256_not_recorded",
            "locked_manifest_not_built_from_planned_manifest_path_and_hash",
            "truthfulqa_locked_manifest_not_heldout_only",
            "prospective_effect_manifest_reuses_historical_mvp_or_calibration_rows",
            "control_pairing_or_replicate_pairing_failure",
        ],
        "effect_pass_gates": [
            {
                "gate": "primary_truthfulqa_open_delta_positive",
                "rule": (
                    "selected alpha=8 minus selected alpha=0 paired "
                    "external-label open-correctness delta has lower 95% CI > 0 "
                    "and point estimate >= +5 pp"
                ),
            },
            {
                "gate": "selected_exceeds_random_direction",
                "rule": (
                    "selected primary open delta lower 95% CI exceeds every "
                    "random-direction seed's primary delta upper 95% CI, and "
                    "the selected point estimate exceeds the random-direction "
                    "seed-family mean delta by at least the minimum practical "
                    "effect"
                ),
            },
            {
                "gate": "selected_exceeds_random_head",
                "rule": (
                    "selected primary open delta lower 95% CI exceeds every "
                    "random-head seed's primary delta upper 95% CI, and the "
                    "selected point estimate exceeds the random-head seed-family "
                    "mean delta by at least the minimum practical effect"
                ),
            },
            {
                "gate": "mc_degradation_blocker",
                "rule": (
                    "selected alpha=8 versus alpha=0 MC letter-likelihood "
                    "accuracy delta must have lower 95% CI >= -2 pp; stronger "
                    "MC degradation blocks a truthfulness-improvement claim"
                ),
            },
            {
                "gate": "attempt_shift_blocker",
                "rule": (
                    "selected alpha=8 versus alpha=0 open-attempted delta must "
                    "be reported with paired uncertainty; an open-correctness "
                    "gain whose 95% CI overlaps an equal-or-larger attempt-rate "
                    "increase is treated as an attempt-rate artifact and blocks "
                    "the claim"
                ),
            },
        ],
        "claim_scope_if_passed": (
            "At most, a fresh prospective same-item SIMID effect claim for the "
            "specified Gemma 3 4B ITI condition under the frozen grading policy."
        ),
        "non_claimable_even_if_passed": [
            "historical MVP open-correctness improvement",
            "general truthfulness improvement beyond the sampled SIMID battery",
            "mechanism-level causal specificity without further ablations",
        ],
    }


def build_output_paths(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "package_dir": relpath(args.output_dir),
        "planned_manifest": relpath(args.planned_manifest),
        "noop_preflight_run_dir": relpath(args.planned_noop_run_dir),
        "effect_run_dir": relpath(args.planned_effect_run_dir),
        "expected_noop_outputs": {
            "run_config": relpath(args.planned_noop_run_dir / "run_config.json"),
            "manifest_locked": relpath(
                args.planned_noop_run_dir / "manifest.locked.json"
            ),
            "results": relpath(
                args.planned_noop_run_dir / "results_noop_preflight.json"
            ),
            "report": relpath(args.planned_noop_run_dir / "report_noop_preflight.md"),
            "provenance": "timestamped *.provenance.YYYYMMDD_HHMMSS.json sidecars",
        },
        "expected_effect_outputs": {
            "run_config": relpath(args.planned_effect_run_dir / "run_config.json"),
            "manifest_locked": relpath(
                args.planned_effect_run_dir / "manifest.locked.json"
            ),
            "diagnostic_gpt4o_adjudication": relpath(
                args.planned_effect_run_dir / "open_adjudication_diagnostic_gpt4o.jsonl"
            ),
            "external_open_label_package": relpath(
                args.planned_effect_run_dir / "external_open_label_package"
            ),
            "external_open_labels": relpath(
                args.planned_effect_run_dir
                / "external_open_label_package"
                / "prospective_open_labels_external.jsonl"
            ),
            "claim_bearing_results": relpath(
                args.planned_effect_run_dir / "results_external_open_labels.json"
            ),
            "claim_bearing_report": relpath(
                args.planned_effect_run_dir / "report_external_open_labels.md"
            ),
            "diagnostic_results": relpath(
                args.planned_effect_run_dir / "results_gpt4o_diagnostic.json"
            ),
            "diagnostic_report": relpath(
                args.planned_effect_run_dir / "report_gpt4o_diagnostic.md"
            ),
            "alias_audit_queue": relpath(
                args.planned_effect_run_dir / "alias_audit_queue_external_labels.jsonl"
            ),
            "provenance": "timestamped *.provenance.YYYYMMDD_HHMMSS.json sidecars",
            "authority_binding": relpath(args.output_dir / "effect_run_manifest.json"),
        },
    }


def build_protocol_text(manifest: dict[str, Any]) -> str:
    authority = manifest["frozen_open_grading_authority"]
    planned = manifest["planned_run"]
    output_paths = manifest["output_paths"]
    gates = manifest["claimability_gates"]
    return f"""# Prospective SIMID Effect-Run Gate

Status: frozen prospective protocol package.
Protocol version: `{PROTOCOL_VERSION}`.

This r2 package pre-specifies the next SIMID effect run. It supersedes
`prospective_effect_run_gate_20260429/`, which remains append-only and
diagnostic. This package is not an execution result and it is not a
retrospective cleanup pass. Historical MVP open metrics remain diagnostic-only.

## Frozen Open-Grading Authority

Future claim-bearing open grading is bound to the passing Opus prospective
calibration run and complete external blind labels returned under the frozen
rubric:

- Analysis: `{authority["passing_analysis"]["path"]}`
- Analysis SHA-256: `{authority["passing_analysis"]["content_sha256"]}`
- Rubric: `{authority["rubric"]["path"]}`
- Rubric SHA-256: `{authority["rubric"]["content_sha256"]}`
- Labels: `{authority["passing_label_file"]["path"]}`
- Label SHA-256: `{authority["passing_label_file"]["content_sha256"]}`

The authority scope is `{authority["authority_scope"]}`. It does not upgrade the
historical MVP run, justify endpoint tuning on old rows, or make `gpt-4o`
automated adjudication claim-bearing.

## Planned Effect Run

- Selected ITI condition: `{planned["selected_iti_condition"]["family"]}`,
  k={planned["selected_iti_condition"]["k"]},
  decode scope `{planned["selected_iti_condition"]["decode_scope"]}`.
- Manifest: `{planned["sample_design"]["manifest_path"]}`
- Effect run directory: `{output_paths["effect_run_dir"]}`
- No-op preflight directory: `{output_paths["noop_preflight_run_dir"]}`
- Alpha grid: `{planned["alpha_grid"]["values"]}`
- Primary alpha: `{planned["alpha_grid"]["primary_alpha"]}`
- Baseline alpha: `{planned["alpha_grid"]["baseline_alpha"]}`

Controls are frozen as unhooked/no-op preflight, five random-direction seeds,
and five layer-matched random-head seeds. A future stronger mechanism claim can
add more controls, but it must not remove these controls after seeing outcomes.

## Primary Estimand

`{planned["primary_estimand"]["name"]}`:
paired TruthfulQA open-correctness delta for selected alpha=8 versus alpha=0,
graded by external blind labels under the frozen prospective rubric and
summarized with a paired base-item bootstrap 95% interval. The minimum
practical effect is +5 pp.

## External Label Requirement

The future claim-bearing analysis must pass
`--prospective-open-label-package`, `--prospective-open-labels`, and
`--require-prospective-open-authority`. Reviewer-facing rows may contain only
`blind_case_id`, `review_order`, `question`, `gold_aliases`, and
`predicted_answer`. The private map must bind each blind ID to the run row,
the locked authority hash, and the run-row hash. Returned labels must be from
external raters and must bind to the blind-case and rubric hashes.

## Secondary Endpoints

- MC letter-likelihood accuracy.
- Open attempted/non-attempted behavior.
- Open first-3-token margin diagnostics.
- TriviaQA Bridge open-correctness externality/specificity check.

## Claimability Gates

The effect run remains non-claimable if any blocker is present:

{chr(10).join(f"- `{item}`" for item in gates["analysis_blockers"])}

To pass the prospective effect gate, the future analysis must satisfy:

{chr(10).join(f"- `{item['gate']}`: {item['rule']}" for item in gates["effect_pass_gates"])}

Even if all gates pass, the claim scope is limited to the specified fresh
prospective same-item SIMID run. It does not imply historical MVP claimability
or a broad truthfulness-improvement claim.

## Commands

Build the locked manifest:

```bash
{planned["commands"]["build_manifest"]["rendered"]}
```

Run the no-op preflight:

```bash
{planned["commands"]["noop_preflight_run"]["rendered"]}
```

Analyze the no-op preflight:

```bash
{planned["commands"]["noop_preflight_analysis"]["rendered"]}
```

Run the effect grid:

```bash
{planned["commands"]["effect_run"]["rendered"]}
```

Analyze future open responses with diagnostic-only `gpt-4o` adjudication:

```bash
{planned["commands"]["effect_diagnostic_gpt4o_analysis"]["rendered"]}
```

Analyze future open responses with returned external blind labels:

```bash
{planned["commands"]["effect_external_label_analysis"]["rendered"]}
```

Before treating open correctness as claim-bearing, the analysis output must
record this r2 package's prospective open-grading authority by path and hash,
validate complete external blind labels, and confirm that the locked manifest
was built from the planned manifest path/hash with held-out TruthfulQA rows
disjoint from the excluded historical MVP and prospective-calibration sample
IDs.
"""


def build_readme_text() -> str:
    return """# Prospective SIMID Effect-Run Gate r2

This append-only r2 package supersedes
`prospective_effect_run_gate_20260429/`. The original package remains
diagnostic-only and is not rewritten.

This package freezes the next SIMID effect-run protocol after the positive
prospective open-grading agreement result, with an additional requirement that
claim-bearing open correctness must use complete external blind labels.

Files:

- `protocol.md`: reviewer-facing frozen protocol and commands.
- `effect_run_manifest.json`: machine-readable protocol, path/hash bindings,
  metrics, controls, and claimability gates.
- `excluded_effect_sample_ids.jsonl`: historical MVP and prospective
  calibration sample/base IDs that the future manifest must exclude.

`gpt-4o` adjudication is diagnostic-only in this protocol. This package is a
planning gate only. It does not run SIMID and does not make historical MVP
open-correctness metrics claim-bearing.
"""


def build_manifest(
    args: argparse.Namespace,
    *,
    exclusions_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_export_args(args)
    source_run_config_path = args.source_mvp_run_dir / "run_config.json"
    source_run_config = read_json(source_run_config_path)
    authority = build_open_grading_authority(
        open_gate_dir=args.open_gate_dir,
        passing_open_analysis=args.passing_open_analysis,
    )
    if exclusions_metadata is None:
        exclusion_path = args.output_dir / "excluded_effect_sample_ids.jsonl"
        if exclusion_path.exists():
            exclusions_metadata = file_metadata(
                exclusion_path,
                schema_version="simid_prospective_effect_exclusion/v1",
            )
        else:
            exclusion_rows = build_exclusion_rows(
                source_mvp_run_dir=args.source_mvp_run_dir,
                open_gate_dir=args.open_gate_dir,
            )
            exclusions_metadata = generated_file_metadata(
                exclusion_path,
                content=jsonl_text(exclusion_rows),
                schema_version="simid_prospective_effect_exclusion/v1",
            )
    return {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git": {
            "sha": get_git_sha(),
            "scripts_dirty": get_git_dirty(),
        },
        "generator": file_metadata(
            Path(__file__).resolve(),
            schema_version=PACKAGE_SCHEMA_VERSION,
        ),
        "source_mvp_run": {
            "path": relpath(args.source_mvp_run_dir),
            "run_config": file_metadata(
                source_run_config_path,
                schema_version=str(source_run_config.get("schema_version")),
            ),
            "status": "diagnostic_only_not_retrospectively_upgraded",
            "role": (
                "Design inheritance for model, artifact, and prior diagnostic "
                "context; not evidence for this prospective effect gate."
            ),
        },
        "frozen_open_grading_authority": authority,
        "planned_run": build_planned_run(
            args=args,
            source_run_config=source_run_config,
        ),
        "exclusions": exclusions_metadata,
        "claimability_gates": build_claimability_gates(),
        "output_paths": build_output_paths(args),
        "append_only_policy": {
            "do_not_rewrite": [
                "historical SIMID MVP outputs",
                "old calibration files",
                "adjudication files",
                "exact-propagation artifacts",
                "returned-label artifacts",
            ],
            "new_outputs_must_use": "fresh semantic run directories listed in output_paths",
        },
    }


def refuse_overwrite(output_dir: Path, *, overwrite: bool) -> None:
    if overwrite:
        return
    planned_files = [
        output_dir / "README.md",
        output_dir / "protocol.md",
        output_dir / "effect_run_manifest.json",
        output_dir / "excluded_effect_sample_ids.jsonl",
    ]
    existing = [path for path in planned_files if path.exists()]
    if existing:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing prospective effect gate files: {rendered}"
        )


def _command_arg_after(command: dict[str, Any], flag: str) -> str:
    argv = command.get("argv")
    if not isinstance(argv, list) or flag not in argv:
        raise ValueError(f"missing {flag} in command")
    index = argv.index(flag)
    try:
        value = argv[index + 1]
    except IndexError as exc:
        raise ValueError(f"missing value after {flag}") from exc
    if not isinstance(value, str):
        raise ValueError(f"non-string value after {flag}")
    return value


def validate_export_payloads(
    *,
    args: argparse.Namespace,
    exclusion_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    protocol_text: str,
    readme_text: str,
) -> None:
    if not exclusion_rows:
        raise ValueError("refusing to export an empty exclusion file")
    if manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION:
        raise ValueError("manifest schema_version mismatch")
    package_dir = manifest.get("output_paths", {}).get("package_dir")
    if package_dir != relpath(args.output_dir):
        raise ValueError("manifest package_dir does not match output_dir")
    commands = manifest.get("planned_run", {}).get("commands")
    if not isinstance(commands, dict):
        raise ValueError("manifest is missing planned commands")
    diagnostic_command = commands.get("effect_diagnostic_gpt4o_analysis")
    if not isinstance(diagnostic_command, dict):
        raise ValueError("manifest is missing diagnostic gpt-4o command")
    adjudication_output = _command_arg_after(
        diagnostic_command,
        "--adjudication-output",
    )
    if Path(adjudication_output).is_absolute() or "/" in adjudication_output:
        raise ValueError("--adjudication-output must be run-dir-relative")
    claim_command = commands.get("effect_external_label_analysis")
    if not isinstance(claim_command, dict):
        raise ValueError("manifest is missing external-label analysis command")
    for required_flag in (
        "--prospective-open-label-package",
        "--prospective-open-labels",
        "--require-prospective-open-authority",
    ):
        argv = claim_command.get("argv")
        if not isinstance(argv, list) or required_flag not in argv:
            raise ValueError(f"external-label command missing {required_flag}")
    combined_text = protocol_text + "\n" + readme_text
    for required_phrase in (
        "external blind labels",
        "gpt-4o` adjudication is diagnostic-only",
        "supersedes",
    ):
        if required_phrase not in combined_text:
            raise ValueError(f"export text missing required phrase: {required_phrase}")


def export_package(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    args.source_mvp_run_dir = args.source_mvp_run_dir.resolve()
    args.open_gate_dir = args.open_gate_dir.resolve()
    args.passing_open_analysis = args.passing_open_analysis.resolve()
    args.output_dir = output_dir
    args.planned_manifest = args.planned_manifest.resolve()
    args.planned_noop_run_dir = args.planned_noop_run_dir.resolve()
    args.planned_effect_run_dir = args.planned_effect_run_dir.resolve()

    refuse_overwrite(output_dir, overwrite=bool(args.overwrite))
    exclusion_rows = build_exclusion_rows(
        source_mvp_run_dir=args.source_mvp_run_dir,
        open_gate_dir=args.open_gate_dir,
    )
    exclusion_text = jsonl_text(exclusion_rows)
    exclusions_metadata = generated_file_metadata(
        output_dir / "excluded_effect_sample_ids.jsonl",
        content=exclusion_text,
        schema_version="simid_prospective_effect_exclusion/v1",
    )
    manifest = build_manifest(args, exclusions_metadata=exclusions_metadata)
    protocol_text = build_protocol_text(manifest)
    readme_text = build_readme_text()
    validate_export_payloads(
        args=args,
        exclusion_rows=exclusion_rows,
        manifest=manifest,
        protocol_text=protocol_text,
        readme_text=readme_text,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_text(output_dir / "README.md", readme_text)
    write_text(output_dir / "protocol.md", protocol_text)
    write_json(output_dir / "effect_run_manifest.json", manifest)
    (output_dir / "excluded_effect_sample_ids.jsonl").write_text(
        exclusion_text,
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    provenance = start_run_provenance(
        args,
        primary_target=output_dir,
        output_targets=[
            output_dir / "README.md",
            output_dir / "protocol.md",
            output_dir / "effect_run_manifest.json",
            output_dir / "excluded_effect_sample_ids.jsonl",
        ],
        primary_target_is_dir=True,
        extra={"simid_schema": PACKAGE_SCHEMA_VERSION},
    )
    status = "completed"
    extra: dict[str, Any] = {}
    try:
        manifest = export_package(args)
        extra = {
            "schema_version": manifest["schema_version"],
            "authority_id": manifest["frozen_open_grading_authority"]["authority_id"],
            "planned_effect_run_dir": manifest["output_paths"]["effect_run_dir"],
        }
        print(
            "Wrote prospective SIMID effect-run gate to "
            f"{relpath(output_dir / 'effect_run_manifest.json')}"
        )
    except BaseException as exc:
        status = provenance_status_for_exception(exc)
        extra["error"] = provenance_error_message(exc)
        raise
    finally:
        finish_run_provenance(provenance, status, extra)


if __name__ == "__main__":
    main()
