"""Materialize truncated jailbreak evaluation inputs from full responses.

The output JSONL keeps the original row identity and metadata, replaces
``response`` with a tokenizer-truncated prefix, and removes evaluator fields
from the full-response source so binary/CSV judges score the truncated view.
On rerun, compatible evaluator fields already present on the truncated view
are preserved so completed judgments are not resubmitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from model_registry import model_path_for, tokenizer_kwargs_for
from utils import (
    finish_run_provenance,
    format_alpha_label,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)

EVALUATOR_FIELDS = {
    "judge",
    "judge_raw",
    "judge_label",
    "judge_label_at_creation",
    "compliance",
    "csv2",
    "strongreject",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def truncate_response(
    tokenizer: Any, response: str, token_limit: int
) -> tuple[str, int]:
    token_ids = tokenizer.encode(response, add_special_tokens=False)
    truncated_ids = token_ids[:token_limit]
    truncated = tokenizer.decode(truncated_ids, skip_special_tokens=False)
    return truncated, len(token_ids)


def truncate_record(
    record: dict[str, Any],
    *,
    tokenizer: Any,
    token_limit: int,
    model_key: str | None,
    model_path: str,
) -> dict[str, Any]:
    response = str(record.get("response", ""))
    truncated, n_tokens = truncate_response(tokenizer, response, token_limit)
    out = {key: value for key, value in record.items() if key not in EVALUATOR_FIELDS}
    out["response"] = truncated
    out["response_truncation"] = {
        "schema_version": "jailbreak_truncation_view/v1",
        "source_response_sha256": sha256_text(response),
        "source_response_chars": len(response),
        "source_response_model_tokens": n_tokens,
        "truncated_response_chars": len(truncated),
        "truncated_response_model_tokens": min(n_tokens, token_limit),
        "token_limit": token_limit,
        "tokenizer_model_key": model_key,
        "tokenizer_model_path": model_path,
        "truncated": n_tokens > token_limit,
    }
    return out


def unique_records_by_id(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]] | None:
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id or record_id in by_id:
            return None
        by_id[record_id] = record
    return by_id


def find_existing_record(
    input_record: dict[str, Any],
    index: int,
    *,
    existing_records: list[dict[str, Any]],
    existing_by_id: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    record_id = input_record.get("id")
    if isinstance(record_id, str) and record_id:
        if existing_by_id is not None:
            return existing_by_id.get(record_id)
        if index < len(existing_records):
            candidate = existing_records[index]
            if candidate.get("id") == record_id:
                return candidate
        return None

    if index >= len(existing_records):
        return None
    candidate = existing_records[index]
    if "id" in candidate:
        return None
    return candidate


def truncation_view_matches(
    existing_record: dict[str, Any],
    truncated_record: dict[str, Any],
) -> bool:
    return existing_record.get("response") == truncated_record.get(
        "response"
    ) and existing_record.get("response_truncation") == truncated_record.get(
        "response_truncation"
    )


def preserve_existing_evaluator_fields(
    truncated_record: dict[str, Any],
    existing_record: dict[str, Any] | None,
) -> None:
    if existing_record is None:
        return
    if not truncation_view_matches(existing_record, truncated_record):
        return
    for field in EVALUATOR_FIELDS:
        if field in existing_record:
            truncated_record[field] = existing_record[field]


def materialize_alpha_file(
    input_path: Path,
    output_path: Path,
    *,
    tokenizer: Any,
    token_limit: int,
    model_key: str | None,
    model_path: str,
) -> int:
    records = load_jsonl(input_path)
    existing_records = load_jsonl(output_path) if output_path.exists() else []
    existing_by_id = unique_records_by_id(existing_records)
    truncated = []
    for index, record in enumerate(records):
        truncated_record = truncate_record(
            record,
            tokenizer=tokenizer,
            token_limit=token_limit,
            model_key=model_key,
            model_path=model_path,
        )
        existing_record = find_existing_record(
            record,
            index,
            existing_records=existing_records,
            existing_by_id=existing_by_id,
        )
        preserve_existing_evaluator_fields(truncated_record, existing_record)
        truncated.append(truncated_record)
    write_jsonl(output_path, truncated)
    return len(truncated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy jailbreak alpha JSONL files with response truncated for evaluation."
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", required=True)
    parser.add_argument("--model_key", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--response-token-limit", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.response_token_limit < 1:
        raise ValueError("--response-token-limit must be positive")

    resolved_model_path = model_path_for(args.model_key, args.model_path)
    manifest_path = args.output_dir / "truncation_view_manifest.json"
    alpha_output_paths = [
        args.output_dir / f"alpha_{format_alpha_label(alpha)}.jsonl"
        for alpha in args.alphas
    ]

    manifest: dict[str, Any] = {
        "schema_version": "jailbreak_truncation_view_manifest/v1",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "alphas": [float(alpha) for alpha in args.alphas],
        "response_token_limit": args.response_token_limit,
        "tokenizer_model_key": args.model_key,
        "tokenizer_model_path": resolved_model_path,
        "files": {},
    }

    provenance_handle = start_run_provenance(
        args,
        primary_target=args.output_dir,
        output_targets=[args.output_dir, *alpha_output_paths, manifest_path],
        extra={
            "input_dir": str(args.input_dir),
            "manifest_path": str(manifest_path),
            "response_token_limit": args.response_token_limit,
            "tokenizer_model_key": args.model_key,
            "tokenizer_model_path": resolved_model_path,
        },
        primary_target_is_dir=True,
    )
    provenance_status = "completed"
    total_records = 0
    provenance_extra: dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "n_records": total_records,
        "files": manifest["files"],
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_model_path,
            **tokenizer_kwargs_for(args.model_key, resolved_model_path),
        )

        for alpha in args.alphas:
            alpha_label = format_alpha_label(alpha)
            input_path = args.input_dir / f"alpha_{alpha_label}.jsonl"
            output_path = args.output_dir / f"alpha_{alpha_label}.jsonl"
            if not input_path.exists():
                raise FileNotFoundError(input_path)
            n_records = materialize_alpha_file(
                input_path,
                output_path,
                tokenizer=tokenizer,
                token_limit=args.response_token_limit,
                model_key=args.model_key,
                model_path=resolved_model_path,
            )
            total_records += n_records
            manifest["files"][alpha_label] = {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "n_records": n_records,
            }
            provenance_extra["n_records"] = total_records
            provenance_extra["files"] = manifest["files"]
            print(f"alpha={alpha_label}: wrote {n_records} rows to {output_path}")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"manifest: {manifest_path}")
    except BaseException as exc:
        provenance_status = provenance_status_for_exception(exc)
        provenance_extra["error"] = provenance_error_message(exc)
        raise
    finally:
        finish_run_provenance(
            provenance_handle,
            provenance_status,
            provenance_extra,
        )


if __name__ == "__main__":
    main()
