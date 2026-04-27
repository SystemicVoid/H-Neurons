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
from typing import Any

from analyze_simid import SIMID_OPEN_FINAL_GRADES
from evaluate_intervention import parse_simpleqa_verdict
from finalize_simid_open_calibration import grade_from_row, index_by_case, load_jsonl

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


def bounded_excerpt(text: str | None) -> str | None:
    if text is None:
        return None
    if len(text) <= RAW_RESPONSE_EXCERPT_CHARS:
        return text
    return text[:RAW_RESPONSE_EXCERPT_CHARS] + "...[truncated]"


def model_uses_max_completion_tokens(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith("gpt-5") or normalized.startswith("o")


def chat_completion_kwargs_for_model(
    model: str,
    *,
    max_output_tokens: int,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    if model_uses_max_completion_tokens(model):
        kwargs: dict[str, Any] = {"max_completion_tokens": max_output_tokens}
        if reasoning_effort is not None and model.strip().lower().startswith("gpt-5"):
            kwargs["reasoning_effort"] = reasoning_effort
        return kwargs
    return {"temperature": 0.0, "max_tokens": max_output_tokens}


def case_custom_id(case_id: str, *, mode: str, model: str) -> str:
    payload = json.dumps(
        {"case_id": case_id, "mode": mode, "model": model},
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


def build_secondary_requests(
    queue_rows: list[dict[str, Any]],
    *,
    existing_rows: dict[str, dict[str, Any]],
    rule_text: str,
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
        custom_id = case_custom_id(case_id, mode="secondary", model=model)
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
    raw_content: str | None,
) -> dict[str, Any]:
    label = parse_label(raw_content)
    return {
        "schema_version": SECONDARY_SCHEMA_VERSION,
        "calibration_case_id": row["calibration_case_id"],
        "label": label,
        "judge_grade": label,
        "rater": {
            "type": "llm",
            "model": model,
            "prompt_version": PROMPT_VERSION,
        },
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
        custom_id = case_custom_id(case_id, mode="adjudicate", model=model)
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


def run_secondary(args: argparse.Namespace, queue_rows: list[dict[str, Any]]) -> None:
    rule_text = load_rule(args.rule_path)
    existing = load_existing_rows(args.output)
    validate_existing_cases(
        existing,
        allowed_case_ids={str(row["calibration_case_id"]) for row in queue_rows},
        output_kind="secondary",
    )
    requests, request_map = build_secondary_requests(
        queue_rows,
        existing_rows=existing,
        rule_text=rule_text,
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
    )
    if not requests:
        print(f"Secondary labels already complete: {len(existing)}/{len(queue_rows)}")
        return
    results = run_batch(
        requests,
        state_path=args.output.with_suffix(".batch_state.json"),
        model=args.model,
        task="simid_open_calibration_secondary",
        api_key=args.api_key,
        max_enqueued_tokens=args.batch_max_enqueued_tokens,
    )
    rows = []
    from openai_batch import parse_chat_content

    for custom_id, row in request_map.items():
        rows.append(
            make_secondary_row(
                row,
                model=args.model,
                raw_content=parse_chat_content(results.get(custom_id, {})),
            )
        )
    append_jsonl(args.output, rows)
    print(
        "Wrote SIMID secondary calibration labels: "
        f"{len(rows)} new, {len(existing) + len(rows)}/{len(queue_rows)} total"
    )


def run_adjudicate(args: argparse.Namespace, queue_rows: list[dict[str, Any]]) -> None:
    if args.secondary_rater_path is None:
        raise ValueError("--secondary-rater-path is required in adjudicate mode")
    rule_text = load_rule(args.rule_path)
    secondary_rows = load_jsonl(args.secondary_rater_path)
    secondary_labels = secondary_label_by_case(secondary_rows)
    expected_case_ids = {str(row["calibration_case_id"]) for row in queue_rows}
    missing_secondary = expected_case_ids - set(secondary_labels)
    if missing_secondary:
        raise ValueError(
            "Secondary labels must cover every calibration case before adjudication; "
            f"missing {len(missing_secondary)}"
        )
    disagreements = disagreement_rows(queue_rows, secondary_labels=secondary_labels)
    disagreement_ids = {str(row["calibration_case_id"]) for row in disagreements}
    existing = load_existing_rows(args.output)
    validate_existing_cases(
        existing,
        allowed_case_ids=disagreement_ids,
        output_kind="adjudication",
    )
    if not disagreements:
        write_empty_jsonl(args.output)
        print("No primary/secondary disagreements; wrote empty adjudication file")
        return
    requests, request_map = build_adjudication_requests(
        disagreements,
        secondary_labels=secondary_labels,
        existing_rows=existing,
        rule_text=rule_text,
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
    )
    if not requests:
        print(
            "Adjudication already complete: "
            f"{len(existing)}/{len(disagreements)} disagreements"
        )
        return
    results = run_batch(
        requests,
        state_path=args.output.with_suffix(".batch_state.json"),
        model=args.model,
        task="simid_open_calibration_adjudication",
        api_key=args.api_key,
        max_enqueued_tokens=args.batch_max_enqueued_tokens,
    )
    rows = []
    from openai_batch import parse_chat_content

    for custom_id, row in request_map.items():
        case_id = str(row["calibration_case_id"])
        rows.append(
            make_adjudication_row(
                row,
                secondary_label=secondary_labels[case_id],
                model=args.model,
                raw_content=parse_chat_content(results.get(custom_id, {})),
            )
        )
    append_jsonl(args.output, rows)
    print(
        "Wrote SIMID calibration adjudications: "
        f"{len(rows)} new, {len(existing) + len(rows)}/{len(disagreements)} total"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("secondary", "adjudicate"), required=True)
    parser.add_argument("--queue-path", type=Path, required=True)
    parser.add_argument("--rule-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--secondary-rater-path", type=Path)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--api-key")
    parser.add_argument("--batch-max-enqueued-tokens", type=int)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--max-output-tokens", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.max_output_tokens is None:
        args.max_output_tokens = 200 if args.mode == "adjudicate" else 32
    queue_rows = load_jsonl(args.queue_path)
    if args.mode == "secondary":
        run_secondary(args, queue_rows)
    else:
        run_adjudicate(args, queue_rows)


if __name__ == "__main__":
    main()
