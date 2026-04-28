#!/usr/bin/env python3
"""Label SIMID open calibration queues with an independent LLM rater."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from analyze_simid import (
    OPEN_CORRECTNESS_CALIBRATION_MIN_CASES,
    SIMID_OPEN_FINAL_GRADES,
)
from bridge_irr import build_adjudication_rule_metadata
from evaluate_intervention import parse_simpleqa_verdict
from finalize_simid_open_calibration import grade_from_row, index_by_case, load_jsonl
from utils import (
    finish_run_provenance,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)

SECONDARY_SCHEMA_VERSION = "simid_open_calibration_secondary_label/v1"
ADJUDICATION_SCHEMA_VERSION = "simid_open_calibration_adjudication/v1"
PROMPT_VERSION = "simid_open_calibration_rule/v1"
RAW_RESPONSE_EXCERPT_CHARS = 1000
ROOT = Path(__file__).resolve().parent.parent


def load_rule(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Calibration rule not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_empty_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


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


def bounded_excerpt(text: str | None) -> str | None:
    if text is None:
        return None
    if len(text) <= RAW_RESPONSE_EXCERPT_CHARS:
        return text
    return text[:RAW_RESPONSE_EXCERPT_CHARS] + "...[truncated]"


def model_uses_max_completion_tokens(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith("gpt-5") or normalized.startswith("o")


def model_supports_reasoning_effort_none(model: str) -> bool:
    normalized = model.strip().lower()
    if "pro" in normalized:
        return False
    if normalized.startswith("gpt-5."):
        minor_match = re.match(r"gpt-5\.(\d+)", normalized)
        return minor_match is not None and int(minor_match.group(1)) >= 1
    return False


def resolve_reasoning_effort(model: str, reasoning_effort: str | None) -> str | None:
    normalized = model.strip().lower()
    if "pro" in normalized and reasoning_effort in {None, "", "auto"}:
        return "high"
    if "pro" in normalized and reasoning_effort != "high":
        raise ValueError(
            f"{model} only supports --reasoning-effort high in this labeler; "
            f"got {reasoning_effort!r}"
        )
    if reasoning_effort in {None, "", "auto"}:
        return "none" if model_supports_reasoning_effort_none(model) else None
    if reasoning_effort == "none" and not model_supports_reasoning_effort_none(model):
        raise ValueError(
            f"{model} does not support --reasoning-effort none; use auto or a "
            "supported value"
        )
    return reasoning_effort


def chat_completion_kwargs_for_model(
    model: str,
    *,
    max_output_tokens: int,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    if model_uses_max_completion_tokens(model):
        kwargs: dict[str, Any] = {"max_completion_tokens": max_output_tokens}
        resolved_reasoning_effort = resolve_reasoning_effort(model, reasoning_effort)
        if resolved_reasoning_effort is not None:
            kwargs["reasoning_effort"] = resolved_reasoning_effort
        return kwargs
    return {"temperature": 0.0, "max_tokens": max_output_tokens}


def case_custom_id(
    case_id: str,
    *,
    mode: str,
    model: str,
    rule_metadata: dict[str, str],
    queue_row_fingerprint: str,
    prompt_inputs: dict[str, Any] | None = None,
) -> str:
    if not queue_row_fingerprint:
        raise ValueError("queue_row_fingerprint is required for calibration batches")
    payload = json.dumps(
        {
            "case_id": case_id,
            "mode": mode,
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "rule_content_sha256": rule_metadata["content_sha256"],
            "queue_row_sha256": queue_row_fingerprint,
            "prompt_inputs": prompt_inputs or {},
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"simid_cal_{mode}_{digest}"


def case_payload(row: dict[str, Any]) -> str:
    payload = {
        "calibration_case_id": row.get("calibration_case_id"),
        "question": row.get("question"),
        "gold_aliases": row.get("gold_aliases"),
        "predicted_answer": row.get("response"),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def secondary_label_messages(
    row: dict[str, Any],
    *,
    rule_text: str,
) -> list[dict[str, str]]:
    content = f"""\
You are the independent secondary rater for SIMID open correctness calibration.
Apply the frozen rule exactly. Do not infer or use the primary judge label,
deterministic alias grade, sample source, or any other hidden metadata.

Frozen rule:
{rule_text}

Case:
{case_payload(row)}

Return exactly one label: CORRECT, INCORRECT, or NOT_ATTEMPTED.
"""
    return [{"role": "user", "content": content}]


def adjudication_messages(
    row: dict[str, Any],
    *,
    secondary_label: str,
    rule_text: str,
) -> list[dict[str, str]]:
    primary_label = grade_from_row(row, row_kind="primary queue")
    payload = {
        "case": json.loads(case_payload(row)),
        "primary_label": primary_label,
        "secondary_label": secondary_label,
    }
    content = f"""\
You are adjudicating one SIMID open correctness calibration disagreement.
Apply the frozen rule exactly and resolve the final label from the case content.
Use the primary and secondary labels only as disagreement context.

Frozen rule:
{rule_text}

Disagreement:
{json.dumps(payload, indent=2, ensure_ascii=False)}

Return a JSON object with exactly these keys:
- "label": one of "CORRECT", "INCORRECT", "NOT_ATTEMPTED"
- "rule_gap": boolean
- "notes": short explanation citing the decisive rule
"""
    return [{"role": "user", "content": content}]


def parse_label(raw_content: str | None) -> str:
    if raw_content is None:
        raise ValueError("Missing judge response content")
    label = parse_simpleqa_verdict(raw_content)
    if label not in SIMID_OPEN_FINAL_GRADES:
        raise ValueError(
            f"Could not parse SIMID open calibration label: {raw_content!r}"
        )
    return label


def parse_adjudication_payload(raw_content: str | None) -> dict[str, Any]:
    if raw_content is None:
        raise ValueError("Missing adjudication response content")
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Adjudication response was not JSON: {raw_content!r}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Adjudication response was not a JSON object: {payload!r}")
    raw_label = (
        payload.get("label") or payload.get("judge_grade") or payload.get("verdict")
    )
    label = str(raw_label).strip().upper() if raw_label is not None else ""
    if label not in SIMID_OPEN_FINAL_GRADES:
        label = parse_simpleqa_verdict(str(raw_label or raw_content))
    if label not in SIMID_OPEN_FINAL_GRADES:
        raise ValueError(f"Invalid adjudication label: {raw_label!r}")
    rule_gap = payload.get("rule_gap")
    if not isinstance(rule_gap, bool):
        raise ValueError(f"Adjudication rule_gap must be boolean: {rule_gap!r}")
    notes = payload.get("notes", "")
    return {"label": label, "rule_gap": rule_gap, "notes": str(notes)}


def load_existing_rows(path: Path) -> dict[str, dict[str, Any]]:
    return index_by_case(load_jsonl(path, required=False), row_kind="existing output")


def validate_existing_cases(
    existing: dict[str, dict[str, Any]],
    *,
    allowed_case_ids: set[str],
    output_kind: str,
) -> None:
    extra = set(existing) - allowed_case_ids
    if extra:
        example = sorted(extra)[:5]
        raise ValueError(
            f"{output_kind} output contains cases not in the current target set: {example}"
        )


def nested_str(row: dict[str, Any], *path: str) -> str | None:
    current: Any = row
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if current is None:
        return None
    return str(current)


def validate_existing_output_rows(
    existing: dict[str, dict[str, Any]],
    *,
    output_kind: str,
    actor_key: str,
    model: str,
    rule_metadata: dict[str, str],
) -> None:
    expected_rule_hash = rule_metadata["content_sha256"]
    for case_id, row in existing.items():
        grade_from_row(row, row_kind=output_kind)
        if nested_str(row, actor_key, "model") != model:
            raise ValueError(
                f"Existing {output_kind} row for {case_id} was produced by "
                f"{nested_str(row, actor_key, 'model')!r}, expected {model!r}"
            )
        if nested_str(row, actor_key, "prompt_version") != PROMPT_VERSION:
            raise ValueError(
                f"Existing {output_kind} row for {case_id} has prompt version "
                f"{nested_str(row, actor_key, 'prompt_version')!r}, expected "
                f"{PROMPT_VERSION!r}"
            )
        if nested_str(row, "rule", "content_sha256") != expected_rule_hash:
            raise ValueError(
                f"Existing {output_kind} row for {case_id} does not match the "
                "current frozen rule hash"
            )


def validate_existing_adjudication_context(
    existing: dict[str, dict[str, Any]],
    *,
    queue_by_case: dict[str, dict[str, Any]],
    secondary_labels: dict[str, str],
) -> None:
    for case_id, row in existing.items():
        queue_row = queue_by_case[case_id]
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
        expected_hash = row.get("queue_row_sha256")
        if not isinstance(expected_hash, str) or not expected_hash:
            raise ValueError(
                f"Existing secondary-rater row for {case_id} is missing "
                "queue_row_sha256; cannot verify the current calibration queue row"
            )
        observed_hash = queue_row_sha256(queue_by_case[case_id])
        if expected_hash != observed_hash:
            raise ValueError(
                f"Existing secondary-rater row for {case_id} was produced for a "
                "different calibration queue row"
            )


def build_secondary_requests(
    queue_rows: list[dict[str, Any]],
    *,
    existing_rows: dict[str, dict[str, Any]],
    rule_text: str,
    rule_metadata: dict[str, str],
    model: str,
    max_output_tokens: int,
    reasoning_effort: str | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    from openai_batch import build_chat_request

    requests: list[dict[str, Any]] = []
    request_map: dict[str, dict[str, Any]] = {}
    kwargs = chat_completion_kwargs_for_model(
        model,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
    )
    for row in queue_rows:
        case_id = str(row["calibration_case_id"])
        if case_id in existing_rows:
            continue
        custom_id = case_custom_id(
            case_id,
            mode="secondary",
            model=model,
            rule_metadata=rule_metadata,
            queue_row_fingerprint=queue_row_sha256(row),
        )
        request = build_chat_request(
            custom_id=custom_id,
            model=model,
            messages=secondary_label_messages(row, rule_text=rule_text),
            **kwargs,
        )
        requests.append(request)
        request_map[custom_id] = row
    return requests, request_map


def make_secondary_row(
    row: dict[str, Any],
    *,
    model: str,
    rule_metadata: dict[str, str],
    raw_content: str | None,
) -> dict[str, Any]:
    label = parse_label(raw_content)
    return {
        "schema_version": SECONDARY_SCHEMA_VERSION,
        "calibration_case_id": row["calibration_case_id"],
        "queue_row_sha256": queue_row_sha256(row),
        "label": label,
        "judge_grade": label,
        "rater": {
            "type": "llm",
            "model": model,
            "prompt_version": PROMPT_VERSION,
        },
        "rule": rule_metadata,
        "labeled_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_judge_response_excerpt": bounded_excerpt(raw_content),
    }


def secondary_label_by_case(secondary_rows: list[dict[str, Any]]) -> dict[str, str]:
    by_case = index_by_case(secondary_rows, row_kind="secondary-rater")
    return {
        case_id: grade_from_row(row, row_kind="secondary-rater")
        for case_id, row in by_case.items()
    }


def disagreement_rows(
    queue_rows: list[dict[str, Any]],
    *,
    secondary_labels: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in queue_rows:
        case_id = str(row["calibration_case_id"])
        secondary = secondary_labels.get(case_id)
        if secondary is None:
            raise ValueError(f"Missing secondary label for {case_id}")
        primary = grade_from_row(row, row_kind="primary queue")
        if primary != secondary:
            rows.append(row)
    return rows


def build_adjudication_requests(
    rows: list[dict[str, Any]],
    *,
    secondary_labels: dict[str, str],
    existing_rows: dict[str, dict[str, Any]],
    rule_text: str,
    rule_metadata: dict[str, str],
    model: str,
    max_output_tokens: int,
    reasoning_effort: str | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    from openai_batch import build_chat_request

    kwargs = chat_completion_kwargs_for_model(
        model,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
    )
    kwargs["response_format"] = {"type": "json_object"}
    requests: list[dict[str, Any]] = []
    request_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row["calibration_case_id"])
        if case_id in existing_rows:
            continue
        custom_id = case_custom_id(
            case_id,
            mode="adjudicate",
            model=model,
            rule_metadata=rule_metadata,
            queue_row_fingerprint=queue_row_sha256(row),
            prompt_inputs={
                "primary_label": grade_from_row(row, row_kind="primary queue"),
                "secondary_label": secondary_labels[case_id],
            },
        )
        request = build_chat_request(
            custom_id=custom_id,
            model=model,
            messages=adjudication_messages(
                row,
                secondary_label=secondary_labels[case_id],
                rule_text=rule_text,
            ),
            **kwargs,
        )
        requests.append(request)
        request_map[custom_id] = row
    return requests, request_map


def make_adjudication_row(
    row: dict[str, Any],
    *,
    secondary_label: str,
    model: str,
    rule_metadata: dict[str, str],
    raw_content: str | None,
) -> dict[str, Any]:
    payload = parse_adjudication_payload(raw_content)
    label = payload["label"]
    return {
        "schema_version": ADJUDICATION_SCHEMA_VERSION,
        "calibration_case_id": row["calibration_case_id"],
        "label": label,
        "judge_grade": label,
        "rule_gap": payload["rule_gap"],
        "notes": payload["notes"],
        "primary_grade": grade_from_row(row, row_kind="primary queue"),
        "secondary_grade": secondary_label,
        "adjudicator": {
            "type": "llm",
            "model": model,
            "prompt_version": PROMPT_VERSION,
        },
        "rule": rule_metadata,
        "adjudicated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_judge_response_excerpt": bounded_excerpt(raw_content),
    }


def run_batch(
    requests: list[dict[str, Any]],
    *,
    state_path: Path,
    model: str,
    task: str,
    api_key: str | None,
    max_enqueued_tokens: int | None,
) -> dict[str, dict[str, Any]]:
    from dotenv import load_dotenv
    from openai import OpenAI
    from openai_batch import resume_or_submit

    load_dotenv(dotenv_path=ROOT / ".env")
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or --api-key")
    client = OpenAI(api_key=resolved_api_key)
    return resume_or_submit(
        client,
        requests,
        state_path,
        metadata={
            "script": "label_simid_open_calibration",
            "task": task,
            "judge_model": model,
            "prompt_version": PROMPT_VERSION,
        },
        max_enqueued_tokens=max_enqueued_tokens,
    )


def result_error_summary(entry: dict[str, Any] | None) -> str:
    if entry is None:
        return "missing batch result"
    response = entry.get("response") if isinstance(entry, dict) else None
    status = response.get("status_code") if isinstance(response, dict) else None
    error = entry.get("error") if isinstance(entry, dict) else None
    return f"status={status!r} error={error!r}"


def raise_after_partial_write(errors: list[str], *, output_path: Path) -> None:
    if not errors:
        return
    preview = "; ".join(errors[:5])
    extra = "" if len(errors) <= 5 else f"; ... {len(errors) - 5} more"
    raise RuntimeError(
        f"{len(errors)} batch result(s) could not be materialized after writing "
        f"successful rows to {output_path}: {preview}{extra}"
    )


class CalibrationQueueLaunchError(ValueError):
    """Raised when the calibration queue fails pre-launch sanity checks."""


def validate_calibration_queue_for_launch(
    queue_rows: list[dict[str, Any]],
    *,
    min_cases: int = OPEN_CORRECTNESS_CALIBRATION_MIN_CASES,
    allow_undersized: bool = False,
) -> None:
    """Refuse to launch secondary labeling on a non-claimable calibration queue.

    Mirrors `analyze_simid.open_calibration_policy()`: a queue that cannot meet
    the policy preconditions (size, both sample sources present, ≥2 distinct
    primary grades, no duplicate case ids) cannot yield a valid kappa/AC1 and
    must not consume secondary/adjudication budget. All failures are reported
    in a single error message rather than short-circuiting on the first.
    """
    failures: list[str] = []

    # 1. n_cases (the only check that may be bypassed for canary smoke runs).
    if not allow_undersized and len(queue_rows) < min_cases:
        failures.append(
            f"n_cases below policy threshold: {len(queue_rows)} < {min_cases}"
        )

    # 2. Missing/duplicate calibration_case_id values.
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

    # 3 & 4. Both sample sources must be represented.
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

    # 5. Primary grades must parse, and ≥2 distinct classes are required;
    # otherwise kappa is undefined.
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


def run_secondary(
    args: argparse.Namespace,
    queue_rows: list[dict[str, Any]],
    *,
    rule_metadata: dict[str, str],
) -> dict[str, Any]:
    # Sanity-check the queue BEFORE building any batch requests so we never
    # spend secondary budget on a queue that cannot satisfy the calibration
    # policy. Adjudication is intentionally NOT gated by this check: by then
    # the secondary cost is already sunk, and refusing to adjudicate would
    # waste rather than save budget.
    validate_calibration_queue_for_launch(
        queue_rows,
        allow_undersized=getattr(args, "allow_undersized_queue", False),
    )
    rule_text = load_rule(args.rule_path)
    existing = load_existing_rows(args.output)
    queue_by_case = {str(row["calibration_case_id"]): row for row in queue_rows}
    validate_existing_cases(
        existing,
        allowed_case_ids=set(queue_by_case),
        output_kind="secondary",
    )
    validate_secondary_queue_context(existing, queue_by_case=queue_by_case)
    validate_existing_output_rows(
        existing,
        output_kind="secondary",
        actor_key="rater",
        model=args.model,
        rule_metadata=rule_metadata,
    )
    requests, request_map = build_secondary_requests(
        queue_rows,
        existing_rows=existing,
        rule_text=rule_text,
        rule_metadata=rule_metadata,
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
    )
    if not requests:
        print(f"Secondary labels already complete: {len(existing)}/{len(queue_rows)}")
        return {
            "mode": "secondary",
            "n_queue_rows": len(queue_rows),
            "n_existing_rows": len(existing),
            "n_requested": 0,
            "n_written": 0,
        }
    results = run_batch(
        requests,
        state_path=args.output.with_suffix(".batch_state.json"),
        model=args.model,
        task="simid_open_calibration_secondary",
        api_key=args.api_key,
        max_enqueued_tokens=args.batch_max_enqueued_tokens,
    )
    rows = []
    errors: list[str] = []
    from openai_batch import parse_chat_content

    for custom_id, row in request_map.items():
        entry = results.get(custom_id)
        try:
            rows.append(
                make_secondary_row(
                    row,
                    model=args.model,
                    rule_metadata=rule_metadata,
                    raw_content=parse_chat_content(entry or {}),
                )
            )
        except Exception as exc:
            errors.append(
                f"{row['calibration_case_id']} ({custom_id}): "
                f"{type(exc).__name__}: {exc}; {result_error_summary(entry)}"
            )
    append_jsonl(args.output, rows)
    print(
        "Wrote SIMID secondary calibration labels: "
        f"{len(rows)} new, {len(existing) + len(rows)}/{len(queue_rows)} total"
    )
    raise_after_partial_write(errors, output_path=args.output)
    return {
        "mode": "secondary",
        "n_queue_rows": len(queue_rows),
        "n_existing_rows": len(existing),
        "n_requested": len(requests),
        "n_written": len(rows),
    }


def run_adjudicate(
    args: argparse.Namespace,
    queue_rows: list[dict[str, Any]],
    *,
    rule_metadata: dict[str, str],
) -> dict[str, Any]:
    if args.secondary_rater_path is None:
        raise ValueError("--secondary-rater-path is required in adjudicate mode")
    rule_text = load_rule(args.rule_path)
    secondary_rows = load_jsonl(args.secondary_rater_path)
    secondary_labels = secondary_label_by_case(secondary_rows)
    expected_case_ids = {str(row["calibration_case_id"]) for row in queue_rows}
    secondary_case_ids = set(secondary_labels)
    if secondary_case_ids != expected_case_ids:
        missing_secondary = expected_case_ids - secondary_case_ids
        extra_secondary = secondary_case_ids - expected_case_ids
        raise ValueError(
            "Secondary labels must cover exactly every calibration case before "
            f"adjudication; missing {len(missing_secondary)}, "
            f"extra {len(extra_secondary)}"
        )
    secondary_by_case = index_by_case(secondary_rows, row_kind="secondary-rater")
    queue_by_case = {str(row["calibration_case_id"]): row for row in queue_rows}
    validate_secondary_queue_context(
        secondary_by_case,
        queue_by_case=queue_by_case,
    )
    validate_existing_output_rows(
        secondary_by_case,
        output_kind="secondary-rater",
        actor_key="rater",
        model=args.secondary_model,
        rule_metadata=rule_metadata,
    )
    disagreements = disagreement_rows(queue_rows, secondary_labels=secondary_labels)
    disagreement_ids = {str(row["calibration_case_id"]) for row in disagreements}
    existing = load_existing_rows(args.output)
    validate_existing_cases(
        existing,
        allowed_case_ids=disagreement_ids,
        output_kind="adjudication",
    )
    validate_existing_output_rows(
        existing,
        output_kind="adjudication",
        actor_key="adjudicator",
        model=args.model,
        rule_metadata=rule_metadata,
    )
    validate_existing_adjudication_context(
        existing,
        queue_by_case=queue_by_case,
        secondary_labels=secondary_labels,
    )
    if not disagreements:
        write_empty_jsonl(args.output)
        print("No primary/secondary disagreements; wrote empty adjudication file")
        return {
            "mode": "adjudicate",
            "n_queue_rows": len(queue_rows),
            "n_disagreements": 0,
            "n_existing_rows": len(existing),
            "n_requested": 0,
            "n_written": 0,
        }
    requests, request_map = build_adjudication_requests(
        disagreements,
        secondary_labels=secondary_labels,
        existing_rows=existing,
        rule_text=rule_text,
        rule_metadata=rule_metadata,
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
    )
    if not requests:
        print(
            "Adjudication already complete: "
            f"{len(existing)}/{len(disagreements)} disagreements"
        )
        return {
            "mode": "adjudicate",
            "n_queue_rows": len(queue_rows),
            "n_disagreements": len(disagreements),
            "n_existing_rows": len(existing),
            "n_requested": 0,
            "n_written": 0,
        }
    results = run_batch(
        requests,
        state_path=args.output.with_suffix(".batch_state.json"),
        model=args.model,
        task="simid_open_calibration_adjudication",
        api_key=args.api_key,
        max_enqueued_tokens=args.batch_max_enqueued_tokens,
    )
    rows = []
    errors: list[str] = []
    from openai_batch import parse_chat_content

    for custom_id, row in request_map.items():
        case_id = str(row["calibration_case_id"])
        entry = results.get(custom_id)
        try:
            rows.append(
                make_adjudication_row(
                    row,
                    secondary_label=secondary_labels[case_id],
                    model=args.model,
                    rule_metadata=rule_metadata,
                    raw_content=parse_chat_content(entry or {}),
                )
            )
        except Exception as exc:
            errors.append(
                f"{case_id} ({custom_id}): {type(exc).__name__}: {exc}; "
                f"{result_error_summary(entry)}"
            )
    append_jsonl(args.output, rows)
    print(
        "Wrote SIMID calibration adjudications: "
        f"{len(rows)} new, {len(existing) + len(rows)}/{len(disagreements)} total"
    )
    raise_after_partial_write(errors, output_path=args.output)
    return {
        "mode": "adjudicate",
        "n_queue_rows": len(queue_rows),
        "n_disagreements": len(disagreements),
        "n_existing_rows": len(existing),
        "n_requested": len(requests),
        "n_written": len(rows),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("secondary", "adjudicate"), required=True)
    parser.add_argument("--queue-path", type=Path, required=True)
    parser.add_argument("--rule-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--secondary-rater-path", type=Path)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--secondary-model", default="gpt-5.5")
    parser.add_argument("--api-key")
    parser.add_argument("--batch-max-enqueued-tokens", type=int)
    parser.add_argument("--reasoning-effort", default="auto")
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument(
        "--allow-undersized-queue",
        action="store_true",
        default=False,
        help=(
            "Bypass ONLY the n_cases >= policy-min check (for canary smoke "
            "runs). The other queue sanity checks (duplicates, both sample "
            "sources, distinct grades) are never bypassable."
        ),
    )
    return parser.parse_args(argv)


def provenance_extra_for_args(
    args: argparse.Namespace,
    *,
    queue_rows: list[dict[str, Any]],
    rule_text: str,
    rule_metadata: dict[str, str],
) -> dict[str, Any]:
    return {
        "mode": args.mode,
        "queue_path": str(args.queue_path),
        "rule_path": str(args.rule_path),
        "rule_content_sha256": rule_metadata["content_sha256"],
        "rule": rule_metadata,
        "judge_model": args.model,
        "secondary_model": args.secondary_model,
        "prompt_version": PROMPT_VERSION,
        "n_queue_rows": len(queue_rows),
        "queue_rows_sha256": queue_rows_sha256(queue_rows),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.max_output_tokens is None:
        args.max_output_tokens = 200 if args.mode == "adjudicate" else 32
    queue_rows = load_jsonl(args.queue_path)
    rule_text = load_rule(args.rule_path)
    rule_metadata = build_adjudication_rule_metadata(args.rule_path)
    output_targets = [
        args.output,
        args.output.with_suffix(".batch_state.json"),
        args.output.with_suffix(".batch_state.results.jsonl"),
    ]
    provenance = start_run_provenance(
        args,
        primary_target=args.output,
        output_targets=output_targets,
        extra=provenance_extra_for_args(
            args,
            queue_rows=queue_rows,
            rule_text=rule_text,
            rule_metadata=rule_metadata,
        ),
    )
    status = "completed"
    extra: dict[str, Any] = {}
    try:
        if args.mode == "secondary":
            extra = run_secondary(args, queue_rows, rule_metadata=rule_metadata)
        else:
            extra = run_adjudicate(args, queue_rows, rule_metadata=rule_metadata)
    except BaseException as exc:
        status = provenance_status_for_exception(exc)
        extra["error"] = provenance_error_message(exc)
        raise
    finally:
        finish_run_provenance(provenance, status, extra)


if __name__ == "__main__":
    main()
