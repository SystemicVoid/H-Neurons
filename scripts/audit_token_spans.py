"""Audit tokenizer/chat-template answer spans before activation extraction."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Any

from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from extract_activations import get_region_indices, unwrap_chat_template_output
from model_registry import (
    assert_causal_lm_supported,
    model_metadata,
    model_path_for,
    tokenizer_kwargs_for,
)
from utils import (
    finish_run_provenance,
    fingerprint_ids,
    json_dumps,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)

SUMMARY_SCHEMA_VERSION = "token_span_audit_summary/v1"
SUMMARY_VALIDATION_SCHEMA_VERSION = "token_span_audit_summary_validation/v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Emit tokenizer/template/span audit rows from answer-token JSONL "
            "without loading the model."
        )
    )
    parser.add_argument(
        "--model_key",
        type=str,
        default=os.environ.get("HNEURONS_MODEL_KEY"),
        help="Registered model key. Used for defaults and tokenizer quirks.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.environ.get("HNEURONS_MODEL_PATH"),
        help="Path or HF id for the tokenizer/model.",
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        default=None,
        help="Path to answer_tokens JSONL.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="JSONL path for per-sample token span audit rows.",
    )
    parser.add_argument(
        "--validate_summary",
        type=Path,
        default=None,
        help="Validate an existing token_span_audit_summary/v1 summary and exit.",
    )
    parser.add_argument(
        "--summary_path",
        type=Path,
        default=None,
        help="Optional JSON summary path. Defaults to <output_path>.summary.json.",
    )
    parser.add_argument(
        "--rendered_chat_path",
        type=Path,
        default=None,
        help="Optional JSON path for rendered chat-template examples.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50,
        help="Maximum records to audit after optional ID filtering.",
    )
    parser.add_argument(
        "--sample_ids_path",
        type=Path,
        default=None,
        help=(
            "Optional manifest of IDs to audit. Accepts a JSON list, an object "
            "with ids/sample_ids, or a qid split object with t/f."
        ),
    )
    parser.add_argument(
        "--rendered_examples",
        type=int,
        default=5,
        help="Number of audited rows to include in rendered chat-template examples.",
    )
    parser.add_argument(
        "--window_tokens",
        type=int,
        default=6,
        help="Decoded token window to store around the resolved answer span.",
    )
    args = parser.parse_args(argv)
    if args.validate_summary is None and (
        args.input_path is None or args.output_path is None
    ):
        parser.error(
            "--input_path and --output_path are required unless --validate_summary is used"
        )
    if args.validate_summary is not None:
        return args
    args.model_path = model_path_for(args.model_key, args.model_path)
    assert_causal_lm_supported(args.model_key, args.model_path)
    if args.summary_path is None and args.output_path is not None:
        args.summary_path = args.output_path.with_suffix(
            f"{args.output_path.suffix}.summary.json"
        )
    if args.max_samples <= 0:
        raise ValueError("--max_samples must be positive")
    if args.rendered_examples < 0:
        raise ValueError("--rendered_examples must be non-negative")
    if args.window_tokens < 0:
        raise ValueError("--window_tokens must be non-negative")
    return args


def load_sample_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return {str(item) for item in payload}
    if isinstance(payload, dict):
        for key in ("ids", "sample_ids"):
            value = payload.get(key)
            if isinstance(value, list):
                return {str(item) for item in value}
        if isinstance(payload.get("t"), list) and isinstance(payload.get("f"), list):
            return {str(item) for item in payload["t"] + payload["f"]}
    raise ValueError(
        "sample ID manifest must be a JSON list, an object with ids/sample_ids, "
        "or a qid split object with t/f"
    )


def iter_answer_token_records(
    path: Path,
    *,
    max_samples: int,
    sample_ids: set[str] | None,
) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    scanned = 0
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            scanned += 1
            payload = json.loads(stripped)
            if not isinstance(payload, dict) or len(payload) != 1:
                raise ValueError(
                    f"{path}:{line_no} must be a one-key object keyed by qid"
                )
            qid, data = next(iter(payload.items()))
            qid = str(qid)
            if sample_ids is not None and qid not in sample_ids:
                continue
            if not isinstance(data, dict):
                raise ValueError(f"{path}:{line_no} row for {qid!r} is not an object")
            data_row: dict[str, Any] = {str(key): value for key, value in data.items()}
            records.append({"qid": qid, "line_no": line_no, **data_row})
            if len(records) >= max_samples:
                break
    return records, scanned


def _tensor_ids_to_list(input_ids: Any) -> list[int]:
    if hasattr(input_ids, "reshape"):
        return [int(token_id) for token_id in input_ids.reshape(-1).tolist()]
    if input_ids and isinstance(input_ids[0], list):
        return [int(token_id) for token_id in input_ids[0]]
    return [int(token_id) for token_id in input_ids]


def _region_payload(region: tuple[int, int] | None) -> dict[str, int] | None:
    if region is None:
        return None
    start, end = region
    return {"start": int(start), "end": int(end), "length": int(end - start)}


def _decoded_window(
    tokenizer: Any,
    token_ids: list[int],
    region: tuple[int, int] | None,
    *,
    window_tokens: int,
) -> dict[str, Any]:
    decoded_tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
    if region is None:
        return {
            "window_start": None,
            "window_end": None,
            "tokens": [],
            "decoded_text": None,
        }
    start, end = region
    window_start = max(0, start - window_tokens)
    window_end = min(len(token_ids), end + window_tokens)
    return {
        "window_start": window_start,
        "window_end": window_end,
        "tokens": decoded_tokens[window_start:window_end],
        "decoded_text": tokenizer.decode(token_ids[start:end]),
    }


def _chat_template_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_tensors="pt",
    )
    return _tensor_ids_to_list(unwrap_chat_template_output(rendered))


def _render_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    return str(rendered)


def audit_answer_token_record(
    tokenizer: Any,
    record: dict[str, Any],
    *,
    window_tokens: int,
) -> dict[str, Any]:
    qid = str(record["qid"])
    question = str(record.get("question", ""))
    response = str(record.get("response", ""))
    raw_answer_tokens = record.get("answer_tokens", [])
    answer_tokens = [str(token) for token in raw_answer_tokens]
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]
    full_ids_raw = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=False,
    )
    full_ids = unwrap_chat_template_output(full_ids_raw)
    token_ids = _tensor_ids_to_list(full_ids)
    regions = get_region_indices(
        full_ids,
        tokenizer,
        question,
        response,
        answer_tokens,
    )
    answer_region = regions["answer_tokens"]
    status = "ok" if answer_region is not None else "missing_answer_region"
    answer_window = _decoded_window(
        tokenizer,
        token_ids,
        answer_region,
        window_tokens=window_tokens,
    )
    user_messages = [{"role": "user", "content": question}]
    user_no_generation_ids = _chat_template_ids(
        tokenizer,
        user_messages,
        add_generation_prompt=False,
    )
    user_generation_ids = _chat_template_ids(
        tokenizer,
        user_messages,
        add_generation_prompt=True,
    )
    normalized_answer_tokens = [
        token.replace("▁", " ").replace("Ġ", " ") for token in answer_tokens
    ]
    return {
        "schema_version": "token_span_audit/v1",
        "qid": qid,
        "line_no": int(record["line_no"]),
        "status": status,
        "judge": record.get("judge"),
        "question_preview": question[:240],
        "response_preview": response[:240],
        "answer_tokens": answer_tokens,
        "normalized_answer_tokens": normalized_answer_tokens,
        "token_counts": {
            "full_chat": len(token_ids),
            "user_no_generation_prompt": len(user_no_generation_ids),
            "user_generation_prompt": len(user_generation_ids),
            "answer_tokens": len(answer_tokens),
        },
        "regions": {
            "input": _region_payload(regions["input"]),
            "output": _region_payload(regions["output"]),
            "answer_tokens": _region_payload(answer_region),
        },
        "answer_window": answer_window,
    }


def render_example(
    tokenizer: Any,
    record: dict[str, Any],
    audit_row: dict[str, Any],
) -> dict[str, Any]:
    question = str(record.get("question", ""))
    response = str(record.get("response", ""))
    user_messages = [{"role": "user", "content": question}]
    full_messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]
    return {
        "qid": audit_row["qid"],
        "status": audit_row["status"],
        "regions": audit_row["regions"],
        "user_no_generation_prompt": _render_chat_template(
            tokenizer,
            user_messages,
            add_generation_prompt=False,
        ),
        "user_generation_prompt": _render_chat_template(
            tokenizer,
            user_messages,
            add_generation_prompt=True,
        ),
        "full_user_assistant_chat": _render_chat_template(
            tokenizer,
            full_messages,
            add_generation_prompt=False,
        ),
    }


def build_summary(
    args: argparse.Namespace,
    *,
    rows: list[dict[str, Any]],
    total_rows_scanned: int,
    sample_ids: set[str] | None,
) -> dict[str, Any]:
    status_counts = Counter(str(row["status"]) for row in rows)
    audited_ids = [str(row["qid"]) for row in rows]
    status = (
        "needs_review"
        if not rows or status_counts.get("missing_answer_region")
        else "ok"
    )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": status,
        "model_key": args.model_key,
        "model_path": args.model_path,
        "registered_model": model_metadata(args.model_key, args.model_path),
        "tokenizer_kwargs": tokenizer_kwargs_for(args.model_key, args.model_path),
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "rendered_chat_path": (
            str(args.rendered_chat_path) if args.rendered_chat_path else None
        ),
        "total_rows_scanned": total_rows_scanned,
        "audited_rows": len(rows),
        "audited_id_fingerprint": fingerprint_ids(audited_ids),
        "sample_ids_path": str(args.sample_ids_path) if args.sample_ids_path else None,
        "sample_id_filter_count": len(sample_ids) if sample_ids is not None else None,
        "status_counts": dict(status_counts),
        "missing_answer_region_qids": [
            str(row["qid"]) for row in rows if row["status"] == "missing_answer_region"
        ],
        "model_loaded": False,
        "gpu_required": False,
        "api_required": False,
    }


def validate_token_span_summary(summary: dict[str, Any]) -> dict[str, Any]:
    status_counts = summary.get("status_counts")
    status_counts = status_counts if isinstance(status_counts, dict) else {}
    audited_rows = summary.get("audited_rows")
    audited_rows = audited_rows if isinstance(audited_rows, int) else None
    missing_qids = summary.get("missing_answer_region_qids")
    missing_qids = missing_qids if isinstance(missing_qids, list) else None
    checks = {
        "schema_version": summary.get("schema_version") == SUMMARY_SCHEMA_VERSION,
        "status_ok": summary.get("status") == "ok",
        "audited_rows_positive": isinstance(audited_rows, int) and audited_rows > 0,
        "all_rows_ok": audited_rows is not None
        and status_counts.get("ok") == audited_rows,
        "no_missing_answer_regions": missing_qids == [],
        "model_not_loaded": summary.get("model_loaded") is False,
        "no_gpu_required": summary.get("gpu_required") is False,
        "no_api_required": summary.get("api_required") is False,
    }
    messages = {
        "schema_version": f"schema_version must be {SUMMARY_SCHEMA_VERSION}",
        "status_ok": "summary status must be ok",
        "audited_rows_positive": "audited_rows must be positive",
        "all_rows_ok": "all audited rows must have status ok",
        "no_missing_answer_regions": "missing_answer_region_qids must be empty",
        "model_not_loaded": "token-span preflight must not load the model",
        "no_gpu_required": "token-span preflight must not require GPU",
        "no_api_required": "token-span preflight must not require API access",
    }
    reasons = [messages[key] for key, passed in checks.items() if not passed]
    accepted = not reasons
    return {
        "schema_version": SUMMARY_VALIDATION_SCHEMA_VERSION,
        "status": "ok" if accepted else "needs_review",
        "accepted": accepted,
        "checks": checks,
        "reasons": reasons,
    }


def validate_token_span_summary_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {
            "schema_version": SUMMARY_VALIDATION_SCHEMA_VERSION,
            "status": "needs_review",
            "accepted": False,
            "checks": {},
            "reasons": [f"{path} must contain a JSON object"],
        }
    return validate_token_span_summary(payload)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.validate_summary is not None:
        validation = validate_token_span_summary_file(args.validate_summary)
        print(json_dumps(validation))
        return 0 if validation["accepted"] else 1

    if args.output_path is None or args.input_path is None:
        raise RuntimeError(
            "--input_path and --output_path are required outside validation mode"
        )
    if args.summary_path is None:
        raise RuntimeError("--summary_path was not resolved")
    output_targets = [args.output_path, args.summary_path]
    if args.rendered_chat_path is not None:
        output_targets.append(args.rendered_chat_path)
    provenance_handle = start_run_provenance(
        args,
        primary_target=args.output_path,
        output_targets=output_targets,
    )
    provenance_status = "completed"
    provenance_extra: dict[str, Any] = {}
    try:
        sample_ids = load_sample_ids(args.sample_ids_path)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            **tokenizer_kwargs_for(args.model_key, args.model_path),
        )
        records, total_rows_scanned = iter_answer_token_records(
            args.input_path,
            max_samples=args.max_samples,
            sample_ids=sample_ids,
        )
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.summary_path is not None:
            args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        if args.rendered_chat_path is not None:
            args.rendered_chat_path.parent.mkdir(parents=True, exist_ok=True)

        audit_rows: list[dict[str, Any]] = []
        rendered_examples: list[dict[str, Any]] = []
        with args.output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                audit_row = audit_answer_token_record(
                    tokenizer,
                    record,
                    window_tokens=args.window_tokens,
                )
                audit_rows.append(audit_row)
                handle.write(json.dumps(audit_row, sort_keys=True) + "\n")
                if len(rendered_examples) < args.rendered_examples:
                    rendered_examples.append(
                        render_example(tokenizer, record, audit_row)
                    )

        summary = build_summary(
            args,
            rows=audit_rows,
            total_rows_scanned=total_rows_scanned,
            sample_ids=sample_ids,
        )
        args.summary_path.write_text(json_dumps(summary), encoding="utf-8")
        if args.rendered_chat_path is not None:
            args.rendered_chat_path.write_text(
                json_dumps(
                    {
                        "schema_version": "rendered_chat_examples/v1",
                        "model_key": args.model_key,
                        "model_path": args.model_path,
                        "examples": rendered_examples,
                    }
                ),
                encoding="utf-8",
            )
        provenance_extra["summary"] = {
            "audited_rows": len(audit_rows),
            "status_counts": summary["status_counts"],
            "gpu_required": False,
            "api_required": False,
        }
    except BaseException as exc:
        provenance_status = provenance_status_for_exception(exc)
        provenance_extra["error"] = provenance_error_message(exc)
        raise
    finally:
        finish_run_provenance(provenance_handle, provenance_status, provenance_extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
