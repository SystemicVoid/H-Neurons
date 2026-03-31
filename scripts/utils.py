"""Shared lightweight utilities used across pipeline scripts.

Keep this module free of heavy imports (torch, transformers, etc.)
so it can be used from any script without pulling in GPU dependencies.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import statistics
import string
import subprocess
import sys
from typing import Any
from collections.abc import Sequence


_SENSITIVE_ARG_SUFFIXES = ("token", "secret", "password", "auth_key", "_api_key")
_SENSITIVE_ARG_NAMES = {"api_key", "sampling_api_key"}
_REDACTED_ARG_VALUE = "[REDACTED]"


def format_alpha_label(alpha: float) -> str:
    """Format alpha values without collapsing nearby micro-betas."""
    rounded_one_decimal = round(alpha, 1)
    if math.isclose(alpha, rounded_one_decimal, abs_tol=1e-9):
        return f"{rounded_one_decimal:.1f}"
    label = f"{alpha:.6f}".rstrip("0").rstrip(".")
    return "0" if label in {"-0", "-0.0"} else label


def normalize_answer(s: str | None) -> str:
    """Standardize answer strings for comparison."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def handle_punc(text: str) -> str:
        exclude = set(string.punctuation + "\u2018\u2019\u00b4`")
        return "".join(ch if ch not in exclude else " " for ch in text)

    if not s:
        return ""
    return white_space_fix(
        remove_articles(handle_punc(str(s).lower().replace("_", " ")))
    ).strip()


def extract_mc_answer(response: str, valid_letters: list[str]) -> str | None:
    """Extract a multiple-choice letter from model response."""
    text = response.strip()
    if text and text[0].upper() in valid_letters:
        return text[0].upper()
    for pattern in [
        r"\b(?:answer|correct)\s+(?:is|:)\s*\(?([A-Z])\)?",
        r"^\(?([A-Z])\)",
        r"^([A-Z])\.",
        r"\*\*([A-Z])\*\*",
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in valid_letters:
                return letter
    for letter in valid_letters:
        if re.search(rf"\b{letter}\b", text):
            return letter
    return None


def get_git_sha() -> str | None:
    """Return the current git SHA when available, otherwise None."""
    result = _run_git_command(["git", "rev-parse", "HEAD"])
    if result is None:
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def get_git_dirty() -> bool | None:
    """Return whether scripts/ has uncommitted changes, otherwise None.

    Only checks the scripts/ directory — data files, notes, and other
    non-code changes do not affect reproducibility of a run.
    """
    result = _run_git_command(["git", "status", "--short", "--", "scripts/"])
    if result is None:
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def _run_git_command(
    command: list[str],
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, ValueError):
        return None


def _is_sensitive_arg_name(name: str) -> bool:
    lowered = name.lower()
    if lowered in _SENSITIVE_ARG_NAMES:
        return True
    return lowered in _SENSITIVE_ARG_SUFFIXES or lowered.endswith(
        _SENSITIVE_ARG_SUFFIXES
    )


def sanitize_run_config(
    args_dict: dict[str, Any], extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Drop secret-like CLI args before uploading run metadata."""
    config = {
        key: value
        for key, value in args_dict.items()
        if key != "wandb" and not _is_sensitive_arg_name(key)
    }
    if extra:
        config.update({key: value for key, value in extra.items() if value is not None})
    normalized = _normalize_json_value(config)
    return normalized if isinstance(normalized, dict) else config


def summarize_numeric_values(values: Sequence[float | int]) -> dict[str, Any]:
    normalized = [float(value) for value in values]
    if not normalized:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p95": None,
        }

    sorted_values = sorted(normalized)
    p95_index = min(len(sorted_values) - 1, round((len(sorted_values) - 1) * 0.95))
    return {
        "count": len(sorted_values),
        "min": min(sorted_values),
        "max": max(sorted_values),
        "mean": statistics.fmean(sorted_values),
        "median": statistics.median(sorted_values),
        "p95": sorted_values[p95_index],
    }


def resolve_provenance_path(
    primary_target: str | os.PathLike[str],
    script_stem: str,
    is_dir: bool,
    run_ts: str | None = None,
) -> Path:
    """Return the canonical sidecar path for a run provenance file.

    Each run gets a unique timestamped filename so reruns do not overwrite
    the previous record.  ``run_ts`` should be a compact UTC timestamp string
    (e.g. ``"20260324_012100"``); if omitted one is generated from the current
    time.
    """
    ts = run_ts or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target_path = Path(primary_target)
    if is_dir:
        return target_path / f"{script_stem}.provenance.{ts}.json"
    return Path(f"{target_path}.provenance.{ts}.json")


def _write_provenance_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(payload), encoding="utf-8")


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, dict):
        return {str(key): _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, set):
        return [_normalize_json_value(item) for item in value]
    return value


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_output_targets(
    output_targets: list[str | os.PathLike[str]],
) -> list[str]:
    return [str(Path(target).resolve()) for target in output_targets]


def _cli_flag_name(token: str) -> str | None:
    if not token.startswith("-"):
        return None
    stripped = token.lstrip("-")
    if not stripped:
        return None
    name, _, _ = stripped.partition("=")
    return name.replace("-", "_").lower()


def _redact_argv(argv: list[str]) -> list[str]:
    redacted: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        flag_name = _cli_flag_name(token)
        if flag_name and _is_sensitive_arg_name(flag_name):
            if "=" in token:
                flag, _, _ = token.partition("=")
                redacted.append(f"{flag}={_REDACTED_ARG_VALUE}")
            else:
                redacted.append(token)
                if index + 1 < len(argv):
                    redacted.append(_REDACTED_ARG_VALUE)
                    index += 1
            index += 1
            continue
        redacted.append(token)
        index += 1
    return redacted


def _build_command() -> tuple[list[str], str]:
    raw_argv = getattr(sys, "orig_argv", None) or [sys.executable, *sys.argv]
    argv = _redact_argv([str(arg) for arg in raw_argv])
    return argv, shlex.join(argv)


def _safe_env() -> dict[str, str]:
    keys = ("CUDA_VISIBLE_DEVICES", "WANDB_MODE")
    return {key: value for key in keys if (value := os.environ.get(key)) is not None}


def init_wandb_run(
    wandb_module: Any,
    args: Any,
    *,
    job_type: str,
    tags: list[str],
    group: str,
    config_extra: dict[str, Any] | None = None,
    project: str = "h-neurons",
    name: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    config = sanitize_run_config(
        vars(args) if hasattr(args, "__dict__") else dict(args),
        extra=config_extra,
    )
    git_sha = get_git_sha()
    if git_sha is None:
        print(
            "Warning: git metadata unavailable; omitting git_sha from W&B config.",
            file=sys.stderr,
        )
    else:
        config["git_sha"] = git_sha

    run = wandb_module.init(
        project=project,
        config=config,
        tags=tags,
        group=group,
        job_type=job_type,
        name=name,
    )
    return run, {
        "project": project,
        "mode": os.environ.get("WANDB_MODE", "online"),
        "tags": tags,
        "group": group,
        "job_type": job_type,
    }


def define_wandb_metrics(
    wandb_module: Any,
    *,
    step_metric: str,
    metrics: list[str],
) -> None:
    wandb_module.define_metric(step_metric)
    for metric in metrics:
        wandb_module.define_metric(metric, step_metric=step_metric)


def log_wandb_files_as_artifact(
    wandb_run: Any,
    wandb_module: Any,
    *,
    name: str,
    artifact_type: str,
    paths: Sequence[str | os.PathLike[str]],
) -> None:
    artifact = wandb_module.Artifact(name=name, type=artifact_type)
    added = False
    for path_like in paths:
        path = Path(path_like)
        if not path.exists() or path.is_dir():
            continue
        artifact.add_file(str(path), name=path.name)
        added = True
    if added:
        wandb_run.log_artifact(artifact)


def provenance_status_for_exception(exc: BaseException) -> str:
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return "interrupted"
    return "failed"


def provenance_error_message(exc: BaseException) -> str:
    detail = str(exc).strip()
    if detail:
        return f"{type(exc).__name__}: {detail}"
    return type(exc).__name__


def start_run_provenance(
    args: Any,
    primary_target: str | os.PathLike[str],
    output_targets: list[str | os.PathLike[str]],
    extra: dict[str, Any] | None = None,
    *,
    primary_target_is_dir: bool = False,
    run_ts: str | None = None,
) -> dict[str, Any] | None:
    """Write initial run provenance sidecar and return a mutable handle."""
    run_ts = run_ts or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    script_stem = Path(sys.argv[0]).stem or "run"
    sidecar_path = resolve_provenance_path(
        primary_target,
        script_stem=script_stem,
        is_dir=primary_target_is_dir,
        run_ts=run_ts,
    )
    if primary_target_is_dir:
        Path(primary_target).mkdir(parents=True, exist_ok=True)
    else:
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    argv, command = _build_command()
    args_dict = vars(args) if hasattr(args, "__dict__") else dict(args)
    payload = {
        "schema_version": "run_provenance/v1",
        "script": os.path.basename(sys.argv[0]),
        "argv": argv,
        "command": command,
        "args": sanitize_run_config(args_dict),
        "cwd": str(Path.cwd()),
        "hostname": socket.gethostname(),
        "python_version": sys.version,
        "git_sha": get_git_sha(),
        "git_dirty": get_git_dirty(),
        "started_at_utc": _utc_now_iso(),
        "completed_at_utc": None,
        "status": "running",
        "output_targets": _resolve_output_targets(output_targets),
        "safe_env": _safe_env(),
    }
    if extra:
        payload.update(
            {key: value for key, value in extra.items() if value is not None}
        )

    handle = {
        "path": sidecar_path,
        "payload": payload,
        "finalized": False,
        "run_ts": run_ts,
    }
    try:
        _write_provenance_file(sidecar_path, payload)
    except Exception as exc:
        print(
            f"Warning: failed to write run provenance to {sidecar_path}: {exc}",
            file=sys.stderr,
        )
        return None
    return handle


def finish_run_provenance(
    handle: dict[str, Any] | None,
    status: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Finalize a run provenance sidecar if one was created successfully."""
    if handle is None or handle.get("finalized", False):
        return
    payload = handle["payload"]
    payload["status"] = status
    payload["completed_at_utc"] = _utc_now_iso()
    if extra:
        payload.update(
            {key: value for key, value in extra.items() if value is not None}
        )
    try:
        _write_provenance_file(handle["path"], payload)
        handle["finalized"] = True
    except Exception as exc:
        print(
            f"Warning: failed to finalize run provenance at {handle['path']}: {exc}",
            file=sys.stderr,
        )


def json_dumps(payload: dict[str, Any]) -> str:
    normalized_payload = _normalize_json_value(payload)
    return f"{json.dumps(normalized_payload, indent=2, sort_keys=True)}\n"


def fingerprint_ids(values: list[str]) -> str:
    """SHA-256 fingerprint (16 hex chars) of a sorted list of IDs."""
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()[:16]


def parse_semicolon_answers(cell: object) -> list[str]:
    """Split a semicolon-delimited TruthfulQA answer cell into individual answers."""
    return [a.strip() for a in str(cell).split(";") if a.strip()]


def audit_split_leakage(metadata: dict[str, Any]) -> dict[str, Any]:
    """Audit extraction metadata for data leakage between splits.

    Checks that train/val/dev/test ID sets are properly disjoint and that
    direction fitting used only dev data. Prints a human-readable report
    and returns the audit dict for embedding in provenance.

    Raises ``RuntimeError`` if any test ID appears in the fitting set.
    """
    train_ids = set(metadata.get("question_ids_train", []))
    val_ids = set(metadata.get("question_ids_val", []))
    dev_ids = set(metadata.get("question_ids_dev", []))
    test_ids = set(metadata.get("question_ids_test", []))

    # Overlap matrix
    overlap_train_val = sorted(train_ids & val_ids)
    overlap_train_test = sorted(train_ids & test_ids)
    overlap_val_test = sorted(val_ids & test_ids)
    overlap_dev_test = sorted(dev_ids & test_ids)

    # Direction fit check: dev should equal train ∪ val
    expected_dev = train_ids | val_ids
    direction_fit_matches_dev = (dev_ids == expected_dev) if dev_ids else None

    # Sample IDs (up to 5) from each split for human inspection
    def _sample(ids: set[str], n: int = 5) -> list[str]:
        return sorted(ids)[:n]

    audit = {
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "dev": len(dev_ids),
            "test": len(test_ids),
        },
        "overlap": {
            "train_val": len(overlap_train_val),
            "train_test": len(overlap_train_test),
            "val_test": len(overlap_val_test),
            "dev_test": len(overlap_dev_test),
        },
        "sample_ids": {
            "train": _sample(train_ids),
            "val": _sample(val_ids),
            "dev": _sample(dev_ids),
            "test": _sample(test_ids),
        },
        "direction_fit_ids_equal_dev_ids": direction_fit_matches_dev,
        "direction_fit_source": metadata.get("direction_fit_source"),
        "sigma_fit_source": metadata.get("sigma_fit_source"),
        "leakage_detected": False,
    }

    # Print report
    print("\n--- Split Leakage Audit ---")
    for split, count in audit["counts"].items():
        print(f"  {split:>5s}: {count} questions")
    print("  Overlap matrix:")
    for pair, count in audit["overlap"].items():
        status = "CLEAN" if count == 0 else f"LEAKED ({count})"
        print(f"    {pair}: {status}")
    print("  Sample IDs:")
    for split, ids in audit["sample_ids"].items():
        print(f"    {split}: {ids}")
    print(f"  direction_fit_ids == dev_ids: {direction_fit_matches_dev}")
    print("--- End Audit ---\n")

    # Hard abort on test leakage into fitting set
    if overlap_dev_test:
        audit["leakage_detected"] = True
        raise RuntimeError(
            f"FATAL: {len(overlap_dev_test)} test question(s) found in "
            f"dev/fitting set: {overlap_dev_test[:5]}"
        )
    if overlap_train_test:
        audit["leakage_detected"] = True
        raise RuntimeError(
            f"FATAL: {len(overlap_train_test)} test question(s) found in "
            f"train set: {overlap_train_test[:5]}"
        )

    return audit
