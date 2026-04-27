#!/usr/bin/env python3
"""Analyze SIMID run outputs with paired item-level estimates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Callable

import numpy as np

from evaluate_intervention import (
    SIMPLEQA_JUDGE_KWARGS,
    build_triviaqa_bridge_judge_messages,
    parse_simpleqa_verdict,
)
from run_simid import OPEN_GRADING_METHOD, assert_noop_equivalence, load_jsonl
from uncertainty import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    build_rate_summary,
    paired_bootstrap_continuous_mean_difference_raw,
    percentile_interval,
)
from utils import (
    finish_run_provenance,
    format_alpha_label,
    provenance_error_message,
    provenance_status_for_exception,
    start_run_provenance,
)


MetricFn = Callable[[dict[str, Any]], float | bool]
RowGroup = list[dict[str, Any]]
Panel = dict[float, dict[str, RowGroup]]
StratumKey = tuple[str, str, tuple[tuple[str, str], ...]]
PRIMARY_MC_RATE_METRIC = "mc_letter_likelihood_correct"
PRIMARY_MC_MARGIN_METRIC = "mc_full_margin"
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
OPTION_ORDER_STABILITY_THRESHOLD = 0.25
DEFAULT_SIMID_OPEN_JUDGE_MODEL = "gpt-4o"
SIMID_OPEN_ADJUDICATION_SCHEMA_VERSION = "simid_open_adjudication/v1"
SIMID_OPEN_JUDGE_PROMPT_VERSION = "simpleqa_verified_aliases/v1"
SIMID_OPEN_ADJUDICATION_PROMPT_BUILDER = (
    "evaluate_intervention.build_triviaqa_bridge_judge_messages"
)
SIMID_OPEN_ADJUDICATION_PARSER = "evaluate_intervention.parse_simpleqa_verdict"
SIMID_OPEN_ADJUDICATION_ELIGIBLE_DATASETS = frozenset({"triviaqa_bridge", "truthfulqa"})
SIMID_OPEN_FINAL_GRADE_LABELS = ("CORRECT", "INCORRECT", "NOT_ATTEMPTED")
SIMID_OPEN_FINAL_GRADES = frozenset(SIMID_OPEN_FINAL_GRADE_LABELS)
SIMID_OPEN_CALIBRATION_QUEUE_SCHEMA_VERSION = "simid_open_calibration_queue/v1"
RAW_JUDGE_RESPONSE_EXCERPT_CHARS = 500
OPEN_CORRECTNESS_CLAIM_CONTRACT = (
    "diagnostic_only_until_judge_calibration_evidence_recorded"
)
OPEN_CORRECTNESS_CALIBRATED_CLAIM_CONTRACT = (
    "claimable_with_recorded_judge_calibration_evidence"
)
OPEN_CORRECTNESS_JUDGE_BLOCKER = "calibration_evidence_not_recorded"
OPEN_CORRECTNESS_NO_JUDGE_BLOCKER = "judge_adjudication_not_loaded"
OPEN_CORRECTNESS_MIXED_SOURCE_BLOCKER = "mixed_adjudication_and_deterministic_sources"
OPEN_CORRECTNESS_INVALID_JUDGE_BLOCKER = "judge_verdict_unknown_or_error"
OPEN_CORRECTNESS_CALIBRATION_SCHEMA_VERSION = "simid_open_calibration/v1"
OPEN_CORRECTNESS_CALIBRATION_TARGET = "simid_open_correctness"
OPEN_CORRECTNESS_CALIBRATION_INVALID_BLOCKER = "calibration_evidence_invalid"
OPEN_CORRECTNESS_CALIBRATION_FAILED_BLOCKER = "calibration_evidence_failed_thresholds"
OPEN_CORRECTNESS_CALIBRATION_RULE_GAP_BLOCKER = "calibration_rule_gap_recorded"
OPEN_CORRECTNESS_CALIBRATION_MIN_KAPPA = 0.80
OPEN_CORRECTNESS_CALIBRATION_MIN_AC1 = 0.80


def deterministic_open_correct(row: dict[str, Any]) -> bool:
    return bool(row["open_grade"]["correct"])


def deterministic_open_attempted(row: dict[str, Any]) -> bool:
    return bool(row["open_grade"]["attempted"])


def adjudication_verdict(adjudication: dict[str, Any] | None) -> str | None:
    if not adjudication:
        return None
    raw = adjudication.get("judge_grade")
    if raw is None:
        parse_result = adjudication.get("parse_result")
        if isinstance(parse_result, dict):
            raw = parse_result.get("verdict")
    if raw is None:
        return None
    return str(raw).strip().upper()


def effective_open_grade(row: dict[str, Any]) -> dict[str, Any]:
    """Return the analysis-grade view of SIMID open correctness.

    Deterministic aliases are only a fallback when no adjudication exists. If
    an adjudication exists but is malformed or unparseable, it is intentionally
    not replaced by the deterministic grade.
    """
    adjudication = row.get("open_adjudication")
    verdict = adjudication_verdict(
        adjudication if isinstance(adjudication, dict) else None
    )
    if verdict is None:
        if isinstance(adjudication, dict):
            return {
                "correct": False,
                "attempted": False,
                "source": "adjudication_unknown",
                "judge_grade": None,
                "parse_valid": False,
                "claimable": False,
                "claimability_blocker": OPEN_CORRECTNESS_INVALID_JUDGE_BLOCKER,
            }
        return {
            "correct": deterministic_open_correct(row),
            "attempted": deterministic_open_attempted(row),
            "source": "deterministic_alias",
            "judge_grade": None,
            "parse_valid": False,
            "claimable": False,
            "claimability_blocker": OPEN_CORRECTNESS_NO_JUDGE_BLOCKER,
        }
    if verdict in SIMID_OPEN_FINAL_GRADES:
        return {
            "correct": verdict == "CORRECT",
            "attempted": verdict in {"CORRECT", "INCORRECT"},
            "source": "adjudication",
            "judge_grade": verdict,
            "parse_valid": True,
            "claimable": False,
            "claimability_blocker": OPEN_CORRECTNESS_JUDGE_BLOCKER,
        }
    return {
        "correct": False,
        "attempted": False,
        "source": "adjudication_unknown",
        "judge_grade": verdict,
        "parse_valid": False,
        "claimable": False,
        "claimability_blocker": OPEN_CORRECTNESS_INVALID_JUDGE_BLOCKER,
    }


def effective_open_correct(row: dict[str, Any]) -> bool:
    return bool(effective_open_grade(row)["correct"])


def effective_open_attempted(row: dict[str, Any]) -> bool:
    return bool(effective_open_grade(row)["attempted"])


BASE_BOOL_METRICS: dict[str, MetricFn] = {
    "mc_letter_likelihood_correct": lambda row: bool(
        row["mc_letter_likelihood"]["chosen_is_gold"]
    ),
    "mc_generated_letter_correct": lambda row: bool(
        row["mc_generated_letter"]["chosen_is_gold"]
    ),
    "mc_likelihood_full_correct": lambda row: bool(
        row["mc_likelihood"]["full"]["chosen_is_gold"]
    ),
    "mc_likelihood_avg_correct": lambda row: bool(
        row["mc_likelihood"]["avg"]["chosen_is_gold"]
    ),
}
DETERMINISTIC_BOOL_METRICS: dict[str, MetricFn] = {
    **BASE_BOOL_METRICS,
    "open_correct": deterministic_open_correct,
    "open_attempted": deterministic_open_attempted,
}
ADJUDICATED_BOOL_METRICS: dict[str, MetricFn] = {
    **BASE_BOOL_METRICS,
    "deterministic_open_correct": deterministic_open_correct,
    "deterministic_open_attempted": deterministic_open_attempted,
    "adjudicated_open_correct": effective_open_correct,
    "adjudicated_open_attempted": effective_open_attempted,
}
BOOL_METRICS: dict[str, MetricFn] = DETERMINISTIC_BOOL_METRICS
CONTINUOUS_METRICS: dict[str, MetricFn] = {
    "mc_full_margin": lambda row: float(row["mc_letter_likelihood"]["full"]["margin"]),
    "mc_avg_margin": lambda row: float(row["mc_letter_likelihood"]["avg"]["margin"]),
    "mc_option_text_full_margin": lambda row: float(
        row["mc_likelihood"]["full"]["margin"]
    ),
    "mc_option_text_avg_margin": lambda row: float(
        row["mc_likelihood"]["avg"]["margin"]
    ),
    "open_first_token_margin": lambda row: float(
        row["open_margins"]["first_token"]["margin"]
    ),
    "open_first3_margin": lambda row: float(row["open_margins"]["first3"]["margin"]),
    "open_full_margin": lambda row: float(row["open_margins"]["full"]["margin"]),
    "open_avg_margin": lambda row: float(row["open_margins"]["avg"]["margin"]),
}


def parse_alpha_label(path: Path) -> float:
    stem = path.stem
    if not stem.startswith("alpha_"):
        raise ValueError(f"Expected alpha_*.jsonl file, got {path}")
    return float(stem.removeprefix("alpha_"))


def load_locked_manifest_metadata(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "manifest.locked.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or "sample_id" not in row:
            continue
        metadata[str(row["sample_id"])] = {
            field: row.get(field) for field in LOCKED_MANIFEST_METADATA_FIELDS
        }
    return metadata


def load_run_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest_metadata = load_locked_manifest_metadata(run_dir)
    for path in sorted(run_dir.glob("*/alpha_*.jsonl")):
        alpha = parse_alpha_label(path)
        condition = path.parent.name
        for row in load_jsonl(path):
            row = dict(row)
            metadata = manifest_metadata.get(str(row.get("sample_id")))
            if metadata:
                for key, value in metadata.items():
                    if value is not None:
                        row.setdefault(key, value)
            row.setdefault("condition", condition)
            row.setdefault("alpha", alpha)
            rows.append(row)
    if not rows:
        raise ValueError(f"No SIMID alpha JSONL files found under {run_dir}")
    return rows


OpenAdjudicationKey = tuple[str, str, float]
OpenAdjudicationDedupKey = tuple[str, str, str, tuple[str, ...], str]


def open_adjudication_key(row: dict[str, Any]) -> OpenAdjudicationKey:
    return (str(row["sample_id"]), str(row["condition"]), float(row["alpha"]))


def open_adjudication_key_payload(row: dict[str, Any]) -> dict[str, Any]:
    sample_id, condition, alpha = open_adjudication_key(row)
    return {"sample_id": sample_id, "condition": condition, "alpha": alpha}


def open_adjudication_response_payload(row: dict[str, Any]) -> Any:
    open_generation = row.get("open_generation")
    if (
        isinstance(open_generation, dict)
        and open_generation.get("response") is not None
    ):
        return open_generation.get("response")
    return row.get("response")


def open_adjudication_dedup_key(row: dict[str, Any]) -> OpenAdjudicationDedupKey:
    """Return the conservative judge-prompt identity for open adjudication.

    Condition and MC option-order are intentionally absent: they do not appear
    in the open-response judge prompt. Alpha is retained so a full intervention
    sweep does not silently merge otherwise identical answers from different
    treatment strengths.
    """
    aliases = tuple(str(alias) for alias in row.get("gold_aliases", []))
    response = open_adjudication_response_payload(row)
    return (
        str(row.get("base_sample_id") or row["sample_id"]),
        format_alpha_label(float(row["alpha"])),
        str(row["question"]),
        aliases,
        str(response),
    )


def load_open_adjudications(path: Path) -> dict[OpenAdjudicationKey, dict[str, Any]]:
    if not path.exists():
        return {}
    adjudications: dict[OpenAdjudicationKey, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = (
                    str(row["sample_id"]),
                    str(row["condition"]),
                    float(row["alpha"]),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Malformed SIMID open adjudication row in {path}:{line_no}"
                ) from exc
            adjudications[key] = row
    return adjudications


def open_adjudication_has_final_grade(adjudication: dict[str, Any] | None) -> bool:
    if not isinstance(adjudication, dict):
        return False
    verdict = adjudication_verdict(adjudication)
    if verdict not in SIMID_OPEN_FINAL_GRADES:
        return False
    parse_result = adjudication.get("parse_result")
    if isinstance(parse_result, dict) and parse_result.get("valid") is False:
        return False
    return True


def final_open_adjudications_by_dedup_key(
    rows: list[dict[str, Any]],
    existing_adjudications: dict[OpenAdjudicationKey, dict[str, Any]],
) -> dict[OpenAdjudicationDedupKey, dict[str, Any]]:
    final_by_dedup_key: dict[OpenAdjudicationDedupKey, dict[str, Any]] = {}
    for adjudication in existing_adjudications.values():
        if not open_adjudication_has_final_grade(adjudication):
            continue
        try:
            dedup_key = open_adjudication_dedup_key(adjudication)
        except (KeyError, TypeError, ValueError):
            continue
        add_final_open_adjudication_by_dedup_key(
            final_by_dedup_key,
            dedup_key=dedup_key,
            adjudication=adjudication,
        )
    for row in rows:
        if not eligible_for_open_adjudication(row):
            continue
        adjudication = existing_adjudications.get(open_adjudication_key(row))
        if not isinstance(adjudication, dict) or not open_adjudication_has_final_grade(
            adjudication
        ):
            continue
        dedup_key = open_adjudication_dedup_key(row)
        add_final_open_adjudication_by_dedup_key(
            final_by_dedup_key,
            dedup_key=dedup_key,
            adjudication=adjudication,
        )
    return final_by_dedup_key


def add_final_open_adjudication_by_dedup_key(
    final_by_dedup_key: dict[OpenAdjudicationDedupKey, dict[str, Any]],
    *,
    dedup_key: OpenAdjudicationDedupKey,
    adjudication: dict[str, Any],
) -> None:
    previous = final_by_dedup_key.get(dedup_key)
    if previous is None:
        final_by_dedup_key[dedup_key] = adjudication
        return
    previous_verdict = adjudication_verdict(previous)
    verdict = adjudication_verdict(adjudication)
    if verdict != previous_verdict:
        raise ValueError(
            "Conflicting final SIMID open adjudications for deduplicated "
            f"judge prompt {dedup_key!r}: {previous_verdict!r} vs {verdict!r}"
        )


def make_open_adjudication_fanout_row(
    row: dict[str, Any],
    *,
    source_adjudication: dict[str, Any],
) -> dict[str, Any]:
    verdict = adjudication_verdict(source_adjudication)
    valid = verdict in SIMID_OPEN_FINAL_GRADES
    adjudicated_correct = verdict == "CORRECT" if valid else None
    deterministic_correct = deterministic_open_correct(row)
    fanout = deepcopy(source_adjudication)
    fanout.update(
        {
            **open_adjudication_key_payload(row),
            "base_sample_id": row.get("base_sample_id"),
            "dataset": row.get("dataset"),
            "mc_endpoint": row.get("mc_endpoint"),
            "question": row.get("question"),
            "gold_aliases": row.get("gold_aliases"),
            "response": row.get("open_generation", {}).get("response"),
            "deterministic_open_grade": row.get("open_grade"),
            "judge_grade": verdict,
            "adjudicated_correct": adjudicated_correct,
            "adjudicated_attempted": (
                verdict in {"CORRECT", "INCORRECT"} if valid else None
            ),
            "judge_disagrees_with_deterministic": (
                adjudicated_correct != deterministic_correct if valid else None
            ),
            "dedup_fanout_source": open_adjudication_key_payload(source_adjudication),
        }
    )
    return fanout


def fanout_existing_open_adjudications(
    rows: list[dict[str, Any]],
    existing_adjudications: dict[OpenAdjudicationKey, dict[str, Any]],
) -> list[dict[str, Any]]:
    final_by_dedup_key = final_open_adjudications_by_dedup_key(
        rows,
        existing_adjudications,
    )
    fanout_rows: list[dict[str, Any]] = []
    for row in rows:
        if not eligible_for_open_adjudication(row):
            continue
        exact_key = open_adjudication_key(row)
        if open_adjudication_has_final_grade(existing_adjudications.get(exact_key)):
            continue
        source_adjudication = final_by_dedup_key.get(open_adjudication_dedup_key(row))
        if source_adjudication is None:
            continue
        fanout_row = make_open_adjudication_fanout_row(
            row,
            source_adjudication=source_adjudication,
        )
        existing_adjudications[exact_key] = fanout_row
        fanout_rows.append(fanout_row)
    return fanout_rows


def attach_open_adjudications(
    rows: list[dict[str, Any]],
    adjudications: dict[OpenAdjudicationKey, dict[str, Any]],
) -> int:
    attached = 0
    for row in rows:
        adjudication = adjudications.get(open_adjudication_key(row))
        if adjudication is None:
            row.pop("open_adjudication", None)
        else:
            row["open_adjudication"] = adjudication
            attached += 1
        row["effective_open_grade"] = effective_open_grade(row)
    return attached


def has_open_adjudication(rows: list[dict[str, Any]]) -> bool:
    return any(isinstance(row.get("open_adjudication"), dict) for row in rows)


def bool_metrics_for_rows(rows: list[dict[str, Any]]) -> dict[str, MetricFn]:
    if has_open_adjudication(rows):
        return ADJUDICATED_BOOL_METRICS
    return DETERMINISTIC_BOOL_METRICS


def open_metric_names_for_rows(rows: list[dict[str, Any]]) -> tuple[str, str]:
    if has_open_adjudication(rows):
        return "adjudicated_open_correct", "adjudicated_open_attempted"
    return "open_correct", "open_attempted"


def eligible_for_open_adjudication(row: dict[str, Any]) -> bool:
    if str(row.get("dataset")) not in SIMID_OPEN_ADJUDICATION_ELIGIBLE_DATASETS:
        return False
    aliases = row.get("gold_aliases")
    response = open_adjudication_response_payload(row)
    return isinstance(aliases, list) and bool(aliases) and response is not None


def simid_open_adjudication_custom_id(row: dict[str, Any]) -> str:
    key = open_adjudication_dedup_key(row)
    payload = json.dumps(key, ensure_ascii=True, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"simid_open_{digest}"


def build_simid_open_adjudication_request(
    row: dict[str, Any],
    *,
    judge_model: str,
    prompt_cache_retention: str | None = None,
) -> dict[str, Any]:
    from openai_batch import build_chat_request

    aliases = [str(alias) for alias in row.get("gold_aliases", [])]
    messages = build_triviaqa_bridge_judge_messages(
        str(row["question"]),
        aliases,
        str(row["open_generation"]["response"]),
    )
    kwargs: dict[str, Any] = dict(SIMPLEQA_JUDGE_KWARGS)
    if prompt_cache_retention:
        kwargs["prompt_cache_retention"] = prompt_cache_retention
    return build_chat_request(
        custom_id=simid_open_adjudication_custom_id(row),
        model=judge_model,
        messages=messages,
        **kwargs,
    )


def build_simid_open_adjudication_requests(
    rows: list[dict[str, Any]],
    *,
    existing_adjudications: dict[OpenAdjudicationKey, dict[str, Any]],
    judge_model: str,
    prompt_cache_retention: str | None = None,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    final_by_dedup_key = final_open_adjudications_by_dedup_key(
        rows,
        existing_adjudications,
    )
    request_groups: dict[OpenAdjudicationDedupKey, list[dict[str, Any]]] = {}
    request_group_order: list[OpenAdjudicationDedupKey] = []
    for row in rows:
        if not eligible_for_open_adjudication(row):
            continue
        if open_adjudication_has_final_grade(
            existing_adjudications.get(open_adjudication_key(row))
        ):
            continue
        dedup_key = open_adjudication_dedup_key(row)
        if dedup_key in final_by_dedup_key:
            continue
        if dedup_key not in request_groups:
            if limit is not None and len(request_group_order) >= limit:
                continue
            request_groups[dedup_key] = []
            request_group_order.append(dedup_key)
        request_groups[dedup_key].append(row)

    requests: list[dict[str, Any]] = []
    request_map: dict[str, list[dict[str, Any]]] = {}
    for dedup_key in request_group_order:
        grouped_rows = request_groups[dedup_key]
        request = build_simid_open_adjudication_request(
            grouped_rows[0],
            judge_model=judge_model,
            prompt_cache_retention=prompt_cache_retention,
        )
        custom_id = str(request["custom_id"])
        requests.append(request)
        request_map[custom_id] = grouped_rows
    return requests, request_map


def bounded_excerpt(text: str | None, *, max_chars: int) -> str | None:
    if text is None:
        return None
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"


def batch_response_request_id(entry: dict[str, Any] | None) -> str | None:
    response = entry.get("response") if isinstance(entry, dict) else None
    if not isinstance(response, dict):
        return None
    request_id = response.get("request_id")
    return str(request_id) if request_id is not None else None


def batch_response_status_code(entry: dict[str, Any] | None) -> int | None:
    response = entry.get("response") if isinstance(entry, dict) else None
    if not isinstance(response, dict):
        return None
    status_code = response.get("status_code")
    return int(status_code) if isinstance(status_code, int) else None


def make_open_adjudication_row(
    row: dict[str, Any],
    *,
    judge_model: str,
    batch_custom_id: str,
    batch_state_path: Path,
    batch_result_entry: dict[str, Any] | None,
    raw_content: str | None,
) -> dict[str, Any]:
    verdict = (
        parse_simpleqa_verdict(raw_content) if raw_content is not None else "ERROR"
    )
    valid = verdict in SIMID_OPEN_FINAL_GRADES
    deterministic_correct = deterministic_open_correct(row)
    adjudicated_correct = verdict == "CORRECT" if valid else None
    return {
        "schema_version": SIMID_OPEN_ADJUDICATION_SCHEMA_VERSION,
        **open_adjudication_key_payload(row),
        "base_sample_id": row.get("base_sample_id"),
        "dataset": row.get("dataset"),
        "mc_endpoint": row.get("mc_endpoint"),
        "question": row.get("question"),
        "gold_aliases": row.get("gold_aliases"),
        "response": row.get("open_generation", {}).get("response"),
        "deterministic_open_grade": row.get("open_grade"),
        "judge": {
            "model": judge_model,
            "prompt_version": SIMID_OPEN_JUDGE_PROMPT_VERSION,
            "prompt_builder": SIMID_OPEN_ADJUDICATION_PROMPT_BUILDER,
            "parser": SIMID_OPEN_ADJUDICATION_PARSER,
            "kwargs": dict(SIMPLEQA_JUDGE_KWARGS),
        },
        "judge_model": judge_model,
        "judge_prompt_version": SIMID_OPEN_JUDGE_PROMPT_VERSION,
        "parse_result": {
            "parser": SIMID_OPEN_ADJUDICATION_PARSER,
            "verdict": verdict,
            "valid": valid,
        },
        "judge_grade": verdict,
        "adjudicated_correct": adjudicated_correct,
        "adjudicated_attempted": (
            verdict in {"CORRECT", "INCORRECT"} if valid else None
        ),
        "raw_judge_response_excerpt": bounded_excerpt(
            raw_content,
            max_chars=RAW_JUDGE_RESPONSE_EXCERPT_CHARS,
        ),
        "adjudicated_at_utc": datetime.now(timezone.utc).isoformat(),
        "batch_state_path": str(batch_state_path),
        "batch_custom_id": batch_custom_id,
        "openai_request_id": batch_response_request_id(batch_result_entry),
        "batch_result_status_code": batch_response_status_code(batch_result_entry),
        "batch_error": (
            batch_result_entry.get("error")
            if isinstance(batch_result_entry, dict)
            else None
        ),
        "judge_disagrees_with_deterministic": (
            adjudicated_correct != deterministic_correct if valid else None
        ),
    }


def append_open_adjudications(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def adjudicate_open_responses_batch(
    rows: list[dict[str, Any]],
    *,
    output_path: Path,
    judge_model: str,
    api_key: str | None,
    prompt_cache_retention: str | None,
    max_enqueued_tokens: int | None,
    limit: int | None,
) -> dict[str, Any]:
    from dotenv import load_dotenv
    from openai_batch import parse_chat_content, resume_or_submit

    load_dotenv()

    existing = load_open_adjudications(output_path)
    n_existing = len(existing)
    reused_adjudication_rows = fanout_existing_open_adjudications(rows, existing)
    batch_requests, request_map = build_simid_open_adjudication_requests(
        rows,
        existing_adjudications=existing,
        judge_model=judge_model,
        prompt_cache_retention=prompt_cache_retention,
        limit=limit,
    )
    state_path = output_path.with_suffix(".batch_state.json")
    if not batch_requests:
        append_open_adjudications(output_path, reused_adjudication_rows)
        return {
            "path": str(output_path),
            "state_path": str(state_path),
            "judge_model": judge_model,
            "n_requested": 0,
            "n_written": len(reused_adjudication_rows),
            "n_existing": n_existing,
            "n_reused_existing_dedup_rows": len(reused_adjudication_rows),
            "n_deduplicated_rows": 0,
        }

    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or --api-key")

    from openai import OpenAI

    client = OpenAI(api_key=resolved_api_key)
    results = resume_or_submit(
        client,
        batch_requests,
        state_path,
        metadata={
            "script": "analyze_simid",
            "task": "simid_open_adjudication",
            "prompt_version": SIMID_OPEN_JUDGE_PROMPT_VERSION,
            "judge_model": judge_model,
        },
        max_enqueued_tokens=max_enqueued_tokens,
    )
    adjudication_rows: list[dict[str, Any]] = []
    n_deduplicated_rows = 0
    for custom_id, grouped_rows in request_map.items():
        n_deduplicated_rows += max(0, len(grouped_rows) - 1)
        entry = results.get(custom_id)
        content = parse_chat_content(entry) if entry is not None else None
        for row in grouped_rows:
            adjudication_rows.append(
                make_open_adjudication_row(
                    row,
                    judge_model=judge_model,
                    batch_custom_id=custom_id,
                    batch_state_path=state_path,
                    batch_result_entry=entry,
                    raw_content=content,
                )
            )
    append_open_adjudications(output_path, reused_adjudication_rows + adjudication_rows)
    return {
        "path": str(output_path),
        "state_path": str(state_path),
        "judge_model": judge_model,
        "n_requested": len(batch_requests),
        "n_written": len(reused_adjudication_rows) + len(adjudication_rows),
        "n_existing": n_existing,
        "n_reused_existing_dedup_rows": len(reused_adjudication_rows),
        "n_deduplicated_rows": n_deduplicated_rows,
    }


def gwet_ac1_for_labels(
    labels_a: list[str],
    labels_b: list[str],
    *,
    labels: tuple[str, ...],
) -> float:
    if len(labels_a) != len(labels_b):
        raise ValueError("AC1 requires equal-length label sequences")
    n_items = len(labels_a)
    if n_items == 0:
        return 0.0

    if set(labels_a) == set(labels_b) and len(set(labels_a)) == 1:
        return 1.0

    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n_items
    counts = Counter(labels_a)
    counts.update(labels_b)
    expected = 0.0
    category_count = len(labels)
    for label in labels:
        p_label = counts.get(label, 0) / (2.0 * n_items)
        expected += p_label * (1.0 - p_label) / (category_count - 1.0)

    if abs(1.0 - expected) < 1e-12:
        return 1.0
    return float((observed - expected) / (1.0 - expected))


def _nested_value(payload: dict[str, Any], *path: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _maybe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def open_calibration_policy() -> dict[str, Any]:
    return {
        "schema_version": OPEN_CORRECTNESS_CALIBRATION_SCHEMA_VERSION,
        "target": OPEN_CORRECTNESS_CALIBRATION_TARGET,
        "requires_pre_frozen_rule": True,
        "requires_audit_queue_disagreement_sample": True,
        "requires_stratified_panel": True,
        "min_cohen_kappa": OPEN_CORRECTNESS_CALIBRATION_MIN_KAPPA,
        "min_gwet_ac1": OPEN_CORRECTNESS_CALIBRATION_MIN_AC1,
        "max_rule_gap_cases": 0,
    }


def _calibration_rule_is_pre_frozen(rule: Any) -> bool:
    if not isinstance(rule, dict):
        return False
    if rule.get("pre_frozen") is True:
        return True
    return all(
        isinstance(rule.get(field), str) and bool(str(rule.get(field)).strip())
        for field in ("path", "content_sha256", "git_commit")
    )


def normalize_open_calibration_summary(
    payload: dict[str, Any],
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    """Validate and normalize L4-style calibration evidence for open grades."""
    schema_version = str(payload.get("schema_version", "")).strip()
    status = str(payload.get("status", "")).strip()
    target = str(payload.get("target", "")).strip()
    sampling_raw = payload.get("sampling")
    sampling: dict[str, Any] = sampling_raw if isinstance(sampling_raw, dict) else {}
    rule = payload.get("rule")
    if not isinstance(rule, dict):
        rule = _nested_value(payload, "provenance", "adjudication_rule")

    n_cases = _maybe_int(
        payload.get("n_cases") or _nested_value(payload, "irr", "n_cases")
    )
    audit_n = _maybe_int(
        _nested_value(sampling, "audit_queue_disagreement_n")
        or _nested_value(sampling, "audit_queue_disagreements_n")
    )
    stratified_n = _maybe_int(_nested_value(sampling, "stratified_panel_n"))
    cohen_kappa = _maybe_float(_nested_value(payload, "irr", "cohen_kappa"))
    gwet_ac1 = _maybe_float(_nested_value(payload, "irr", "gwet_ac1"))
    raw_agreement_count = _maybe_int(
        _nested_value(payload, "irr", "raw_agreement", "count")
    )
    raw_agreement_n = _maybe_int(_nested_value(payload, "irr", "raw_agreement", "n"))
    rule_gap_count = _maybe_int(
        _nested_value(payload, "adjudication", "rule_gap_cases", "count")
    )
    rule_gap_n = _maybe_int(
        _nested_value(payload, "adjudication", "rule_gap_cases", "n")
    )

    blockers: list[str] = []
    if schema_version != OPEN_CORRECTNESS_CALIBRATION_SCHEMA_VERSION:
        blockers.append("schema_version_mismatch")
    if status != "adjudicated":
        blockers.append("status_not_adjudicated")
    if target != OPEN_CORRECTNESS_CALIBRATION_TARGET:
        blockers.append("target_mismatch")
    if not _calibration_rule_is_pre_frozen(rule):
        blockers.append("pre_frozen_rule_not_recorded")
    if n_cases is None or n_cases <= 0:
        blockers.append("n_cases_not_recorded")
    if audit_n is None or audit_n <= 0:
        blockers.append("audit_queue_disagreement_sample_not_recorded")
    if stratified_n is None or stratified_n <= 0:
        blockers.append("stratified_panel_not_recorded")
    if cohen_kappa is None:
        blockers.append("cohen_kappa_not_recorded")
    elif cohen_kappa < OPEN_CORRECTNESS_CALIBRATION_MIN_KAPPA:
        blockers.append("cohen_kappa_below_threshold")
    if gwet_ac1 is None:
        blockers.append("gwet_ac1_not_recorded")
    elif gwet_ac1 < OPEN_CORRECTNESS_CALIBRATION_MIN_AC1:
        blockers.append("gwet_ac1_below_threshold")
    if rule_gap_count is None or rule_gap_n is None:
        blockers.append("rule_gap_not_recorded")
    elif rule_gap_count > 0:
        blockers.append(OPEN_CORRECTNESS_CALIBRATION_RULE_GAP_BLOCKER)

    if raw_agreement_count is None or raw_agreement_n is None:
        raw_agreement = _nested_value(payload, "irr", "raw_agreement")
    else:
        raw_agreement = build_rate_summary(raw_agreement_count, raw_agreement_n)

    failure_blocker: str | None
    if not blockers:
        failure_blocker = None
    elif OPEN_CORRECTNESS_CALIBRATION_RULE_GAP_BLOCKER in blockers:
        failure_blocker = OPEN_CORRECTNESS_CALIBRATION_RULE_GAP_BLOCKER
    elif any(blocker.endswith("_below_threshold") for blocker in blockers):
        failure_blocker = OPEN_CORRECTNESS_CALIBRATION_FAILED_BLOCKER
    else:
        failure_blocker = OPEN_CORRECTNESS_CALIBRATION_INVALID_BLOCKER

    normalized: dict[str, Any] = {
        "schema_version": schema_version,
        "status": status,
        "target": target,
        "path": str(path) if path is not None else None,
        "sampling": {
            "n_cases": n_cases,
            "audit_queue_disagreement_n": audit_n,
            "stratified_panel_n": stratified_n,
        },
        "rule": rule if isinstance(rule, dict) else None,
        "irr": {
            "n_cases": n_cases,
            "raw_agreement": raw_agreement,
            "cohen_kappa": cohen_kappa,
            "gwet_ac1": gwet_ac1,
        },
        "adjudication": {
            "rule_gap_cases": {
                "count": rule_gap_count,
                "n": rule_gap_n,
            },
        },
        "claimability": {
            "passed": not blockers,
            "blockers": blockers,
            "blocker": failure_blocker,
            "policy": open_calibration_policy(),
        },
    }
    return normalized


def load_open_calibration_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected SIMID open calibration JSON object in {path}")
    return normalize_open_calibration_summary(payload, path=path)


def open_calibration_claimability_blocker(
    calibration_summary: dict[str, Any] | None,
) -> str | None:
    if calibration_summary is None:
        return OPEN_CORRECTNESS_JUDGE_BLOCKER
    claimability = calibration_summary.get("claimability")
    if not isinstance(claimability, dict) or not bool(claimability.get("passed")):
        blocker = (
            claimability.get("blocker") if isinstance(claimability, dict) else None
        )
        return str(blocker or OPEN_CORRECTNESS_CALIBRATION_INVALID_BLOCKER)
    return None


def open_correctness_claimability_blocker(
    effective_source_counts: Counter[str],
    *,
    has_judge: bool,
    calibration_summary: dict[str, Any] | None = None,
) -> str | None:
    if (
        effective_source_counts.get("adjudication", 0) > 0
        and effective_source_counts.get("deterministic_alias", 0) > 0
    ):
        return OPEN_CORRECTNESS_MIXED_SOURCE_BLOCKER
    if effective_source_counts.get("adjudication_unknown", 0) > 0:
        return OPEN_CORRECTNESS_INVALID_JUDGE_BLOCKER
    if has_judge:
        blocker = open_calibration_claimability_blocker(calibration_summary)
        return blocker
    return OPEN_CORRECTNESS_NO_JUDGE_BLOCKER


def summarize_open_grading(
    rows: list[dict[str, Any]],
    *,
    adjudication_path: Path,
    adjudication_run: dict[str, Any] | None,
    calibration_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    correct_metric, attempted_metric = open_metric_names_for_rows(rows)
    effective_source_counts: Counter[str] = Counter()
    judge_grade_counts: Counter[str] = Counter()
    valid_adjudications = 0
    invalid_adjudications = 0
    disagreements = 0
    for row in rows:
        grade = effective_open_grade(row)
        effective_source_counts[str(grade["source"])] += 1
        verdict = grade.get("judge_grade")
        if verdict is not None:
            judge_grade_counts[str(verdict)] += 1
        if grade["source"] == "adjudication":
            valid_adjudications += 1
            if bool(grade["correct"]) != deterministic_open_correct(row):
                disagreements += 1
        elif grade["source"] == "adjudication_unknown":
            invalid_adjudications += 1

    has_judge = has_open_adjudication(rows)
    claimability_blocker = open_correctness_claimability_blocker(
        effective_source_counts,
        has_judge=has_judge,
        calibration_summary=calibration_summary,
    )
    claimable_open_correctness = claimability_blocker is None
    summary: dict[str, Any] = {
        "deterministic_alias_grader": OPEN_GRADING_METHOD,
        "metric_correct": correct_metric,
        "metric_attempted": attempted_metric,
        "effective_grade_source_counts": dict(sorted(effective_source_counts.items())),
        "adjudication": {
            "path": str(adjudication_path),
            "schema_version": SIMID_OPEN_ADJUDICATION_SCHEMA_VERSION,
            "judge_prompt_version": SIMID_OPEN_JUDGE_PROMPT_VERSION,
            "prompt_builder": SIMID_OPEN_ADJUDICATION_PROMPT_BUILDER,
            "parser": SIMID_OPEN_ADJUDICATION_PARSER,
            "judge_grade_counts": dict(sorted(judge_grade_counts.items())),
            "n_valid": valid_adjudications,
            "n_unknown_or_error": invalid_adjudications,
            "deterministic_vs_judge_disagreements": disagreements,
        },
        "open_correctness_claim_contract": (
            OPEN_CORRECTNESS_CALIBRATED_CLAIM_CONTRACT
            if claimable_open_correctness
            else OPEN_CORRECTNESS_CLAIM_CONTRACT
        ),
        "open_metrics_scope": (
            "claimable" if claimable_open_correctness else "diagnostic_only"
        ),
        "claimable_open_correctness": claimable_open_correctness,
        "claimability_blocker": claimability_blocker,
    }
    if adjudication_run is not None:
        summary["adjudication"]["last_run"] = adjudication_run
    if calibration_summary is not None:
        summary["calibration"] = calibration_summary
    return summary


def index_rows(rows: list[dict[str, Any]]) -> dict[str, Panel]:
    indexed: dict[str, Panel] = defaultdict(lambda: defaultdict(dict))
    exact_keys: set[tuple[str, float, str]] = set()
    for row in rows:
        condition = str(row["condition"])
        alpha = float(row["alpha"])
        sample_id = str(row["sample_id"])
        base_sample_id = str(row.get("base_sample_id") or sample_id)
        exact_key = (condition, alpha, sample_id)
        if exact_key in exact_keys:
            raise ValueError(
                f"Duplicate SIMID row for condition={condition}, "
                f"alpha={alpha}, sample_id={sample_id}"
            )
        exact_keys.add(exact_key)
        bucket = indexed[condition][alpha]
        bucket.setdefault(base_sample_id, []).append(row)
    return {condition: dict(panel) for condition, panel in indexed.items()}


def replicate_key(row: dict[str, Any]) -> str:
    if "option_order_replicate" in row:
        return f"ord:{int(row['option_order_replicate'])}"
    return f"sample:{row['sample_id']}"


def require_option_order_replicate(row: dict[str, Any]) -> int:
    if "option_order_replicate" not in row or row["option_order_replicate"] is None:
        raise ValueError(
            "missing option_order_replicate metadata for "
            f"sample_id={row.get('sample_id')!r}; rerun with updated SIMID rows "
            "or analyze a run directory that contains manifest.locked.json"
        )
    try:
        return int(row["option_order_replicate"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "invalid option_order_replicate metadata for "
            f"sample_id={row.get('sample_id')!r}: "
            f"{row['option_order_replicate']!r}"
        ) from exc


def replicate_key_set(rows: RowGroup) -> set[str]:
    return {replicate_key(row) for row in rows}


def row_mc_endpoint(row: dict[str, Any]) -> str:
    return str(row.get("mc_endpoint") or "unknown")


def dataset_endpoint_for_group(rows: RowGroup) -> tuple[str, str]:
    datasets = {str(row.get("dataset") or "unknown") for row in rows}
    endpoints = {row_mc_endpoint(row) for row in rows}
    if len(datasets) != 1 or len(endpoints) != 1:
        raise ValueError(
            "SIMID base-item replicate group mixes datasets or MC endpoints: "
            f"datasets={sorted(datasets)}, mc_endpoints={sorted(endpoints)}"
        )
    return next(iter(datasets)), next(iter(endpoints))


def dataset_endpoint_key(dataset: str, mc_endpoint: str) -> str:
    return f"{dataset}::{mc_endpoint}"


def render_stratum_metadata_value(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def truthfulqa_leakage_metadata_for_group(
    rows: RowGroup,
    *,
    dataset: str,
) -> tuple[tuple[str, str], ...]:
    if dataset != "truthfulqa":
        return ()
    has_leakage_metadata = any(
        any(field in row for field in TRUTHFULQA_LEAKAGE_METADATA_FIELDS)
        for row in rows
    )
    if not has_leakage_metadata:
        return ()
    metadata: list[tuple[str, str]] = []
    for field in TRUTHFULQA_LEAKAGE_METADATA_FIELDS:
        values = {render_stratum_metadata_value(row.get(field)) for row in rows}
        if len(values) != 1:
            raise ValueError(
                "SIMID TruthfulQA replicate group mixes leakage metadata for "
                f"{field}: {sorted(values)}"
            )
        metadata.append((field, next(iter(values))))
    return tuple(metadata)


def analysis_stratum_for_group(rows: RowGroup) -> StratumKey:
    dataset, mc_endpoint = dataset_endpoint_for_group(rows)
    return (
        dataset,
        mc_endpoint,
        truthfulqa_leakage_metadata_for_group(rows, dataset=dataset),
    )


def analysis_stratum_key(
    dataset: str,
    mc_endpoint: str,
    leakage_metadata: tuple[tuple[str, str], ...],
) -> str:
    key = dataset_endpoint_key(dataset, mc_endpoint)
    if not leakage_metadata:
        return key
    suffix = "::".join(f"{field}={value}" for field, value in leakage_metadata)
    return f"{key}::{suffix}"


def flatten_groups(groups_by_id: dict[str, RowGroup]) -> list[dict[str, Any]]:
    return [row for rows in groups_by_id.values() for row in rows]


def group_metric_value(rows: RowGroup, metric: MetricFn) -> float:
    if not rows:
        raise ValueError("SIMID row group cannot be empty")
    return float(np.mean([float(metric(row)) for row in rows]))


def bootstrap_mean_summary(
    values: list[float] | np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    scale: float = 1.0,
    estimate_key: str = "estimate",
    ci_key: str = "ci",
) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if len(arr) == 0:
        raise ValueError("values cannot be empty")
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)
    for sample_idx in range(n_resamples):
        indices = rng.choice(len(arr), size=len(arr), replace=True)
        samples[sample_idx] = float(arr[indices].mean() * scale)
    interval = percentile_interval(
        samples,
        method="bootstrap_percentile_item_grouped",
    )
    return {
        estimate_key: float(arr.mean() * scale),
        "sum": float(arr.sum()),
        "n": int(len(arr)),
        ci_key: interval.to_dict(),
        "bootstrap": {
            "n_resamples": int(n_resamples),
            "seed": int(seed),
            "confidence": float(interval.level),
            "resampling": "base_sample_id",
            "interval": "percentile",
        },
    }


def paired_rate_delta(
    sample_ids: list[str],
    baseline_rows: dict[str, RowGroup],
    comparison_rows: dict[str, RowGroup],
    metric: MetricFn,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    baseline = np.array(
        [
            group_metric_value(baseline_rows[sample_id], metric)
            for sample_id in sample_ids
        ],
        dtype=float,
    )
    comparison = np.array(
        [
            group_metric_value(comparison_rows[sample_id], metric)
            for sample_id in sample_ids
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)
    for sample_idx in range(n_resamples):
        indices = rng.choice(len(sample_ids), size=len(sample_ids), replace=True)
        samples[sample_idx] = (
            float(comparison[indices].mean()) - float(baseline[indices].mean())
        ) * 100.0
    interval = percentile_interval(
        samples,
        method="bootstrap_percentile_paired_base_sample_id",
    )
    return {
        "estimate_pp": float((comparison.mean() - baseline.mean()) * 100.0),
        "ci_pp": interval.to_dict(),
        "bootstrap": {
            "n_resamples": int(n_resamples),
            "seed": int(seed),
            "confidence": float(interval.level),
            "resampling": "paired_by_base_sample_id",
            "interval": "percentile",
        },
    }


def require_paired_panel(
    indexed: dict[str, Panel],
    *,
    condition: str,
    alphas: list[float],
) -> tuple[list[str], Panel]:
    if condition not in indexed:
        raise ValueError(f"Missing SIMID condition: {condition}")
    panel = indexed[condition]
    missing_alphas = [alpha for alpha in alphas if alpha not in panel]
    if missing_alphas:
        raise ValueError(
            f"Condition {condition} is missing alpha files: {missing_alphas}"
        )
    sample_sets = {alpha: set(panel[alpha]) for alpha in alphas}
    reference_alpha = alphas[0]
    reference = sample_sets[reference_alpha]
    for alpha, sample_ids in sample_sets.items():
        if sample_ids != reference:
            raise ValueError(
                f"Unpaired SIMID rows for condition={condition}: alpha={alpha} "
                f"has {len(sample_ids)} samples, alpha={reference_alpha} has "
                f"{len(reference)}. Missing vs reference: "
                f"{sorted(reference - sample_ids)[:5]}; extra: "
                f"{sorted(sample_ids - reference)[:5]}"
            )
        for sample_id in reference:
            reference_replicates = replicate_key_set(panel[reference_alpha][sample_id])
            alpha_replicates = replicate_key_set(panel[alpha][sample_id])
            if alpha_replicates != reference_replicates:
                raise ValueError(
                    f"Unpaired SIMID replicate rows for condition={condition}, "
                    f"sample_id={sample_id}, alpha={alpha}: "
                    f"missing vs alpha={reference_alpha}: "
                    f"{sorted(reference_replicates - alpha_replicates)[:5]}; "
                    f"extra: {sorted(alpha_replicates - reference_replicates)[:5]}"
                )
    if not reference:
        raise ValueError(f"Condition {condition} has no paired samples")
    return sorted(reference), panel


def rate_at_alpha(
    sample_ids: list[str],
    rows_by_id: dict[str, RowGroup],
    metric: MetricFn,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    values = [
        group_metric_value(rows_by_id[sample_id], metric) for sample_id in sample_ids
    ]
    return bootstrap_mean_summary(values, n_resamples=n_resamples, seed=seed)


def paired_bool_delta(
    sample_ids: list[str],
    baseline_rows: dict[str, RowGroup],
    comparison_rows: dict[str, RowGroup],
    metric: MetricFn,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    return paired_rate_delta(
        sample_ids,
        baseline_rows,
        comparison_rows,
        metric,
        n_resamples=n_resamples,
        seed=seed,
    )


def paired_continuous_delta(
    sample_ids: list[str],
    baseline_rows: dict[str, RowGroup],
    comparison_rows: dict[str, RowGroup],
    metric: MetricFn,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    baseline = np.array(
        [
            group_metric_value(baseline_rows[sample_id], metric)
            for sample_id in sample_ids
        ],
        dtype=float,
    )
    comparison = np.array(
        [
            group_metric_value(comparison_rows[sample_id], metric)
            for sample_id in sample_ids
        ],
        dtype=float,
    )
    return paired_bootstrap_continuous_mean_difference_raw(
        baseline,
        comparison,
        n_resamples=n_resamples,
        seed=seed,
    )


def summarize_sample_ids(
    panel: Panel,
    *,
    sample_ids: list[str],
    alphas: list[float],
    baseline_alpha: float,
    n_resamples: int,
    seed: int,
    bool_metrics: dict[str, MetricFn] | None = None,
) -> dict[str, Any]:
    if bool_metrics is None:
        bool_metrics = BOOL_METRICS
    baseline_rows = panel[baseline_alpha]
    rates: dict[str, Any] = {}
    deltas: dict[str, Any] = {}
    continuous_deltas: dict[str, Any] = {}
    interactions: dict[str, Any] = {}
    flips: dict[str, Any] = {}

    for alpha in alphas:
        rows_by_id = panel[alpha]
        rates[str(alpha)] = {
            name: rate_at_alpha(
                sample_ids,
                rows_by_id,
                metric,
                n_resamples=n_resamples,
                seed=seed,
            )
            for name, metric in bool_metrics.items()
        }
        interactions[str(alpha)] = interaction_counts(
            sample_ids,
            rows_by_id,
            n_resamples=n_resamples,
            seed=seed,
        )
        if alpha == baseline_alpha:
            continue
        comparison_rows = rows_by_id
        deltas[str(alpha)] = {
            name: paired_bool_delta(
                sample_ids,
                baseline_rows,
                comparison_rows,
                metric,
                n_resamples=n_resamples,
                seed=seed,
            )
            for name, metric in bool_metrics.items()
        }
        continuous_deltas[str(alpha)] = {
            name: paired_continuous_delta(
                sample_ids,
                baseline_rows,
                comparison_rows,
                metric,
                n_resamples=n_resamples,
                seed=seed,
            )
            for name, metric in CONTINUOUS_METRICS.items()
        }
        flips[str(alpha)] = flip_table(sample_ids, baseline_rows, comparison_rows)

    return {
        "n_paired_items": len(sample_ids),
        "pairing_unit": "base_sample_id",
        "n_rows_at_baseline": sum(
            len(baseline_rows[sample_id]) for sample_id in sample_ids
        ),
        "alphas": alphas,
        "baseline_alpha": baseline_alpha,
        "rates": rates,
        "paired_deltas_vs_baseline": deltas,
        "paired_margin_deltas_vs_baseline": continuous_deltas,
        "mc_open_interactions": interactions,
        "flip_tables_vs_baseline": flips,
    }


def analysis_stratum_sample_ids(
    sample_ids: list[str],
    panel: Panel,
    *,
    alphas: list[float],
    baseline_alpha: float,
) -> dict[StratumKey, list[str]]:
    strata: dict[StratumKey, list[str]] = defaultdict(list)
    for sample_id in sample_ids:
        reference = analysis_stratum_for_group(panel[baseline_alpha][sample_id])
        for alpha in alphas:
            alpha_value = analysis_stratum_for_group(panel[alpha][sample_id])
            if alpha_value != reference:
                raise ValueError(
                    f"SIMID sample_id={sample_id} changes analysis stratum "
                    f"between alpha={baseline_alpha} and alpha={alpha}: "
                    f"{reference} vs {alpha_value}"
                )
        strata[reference].append(sample_id)
    return {key: sorted(ids) for key, ids in strata.items()}


def summarize_condition(
    indexed: dict[str, Panel],
    *,
    condition: str,
    alphas: list[float],
    baseline_alpha: float,
    n_resamples: int,
    seed: int,
    bool_metrics: dict[str, MetricFn] | None = None,
) -> dict[str, Any]:
    if bool_metrics is None:
        bool_metrics = BOOL_METRICS
    sample_ids, panel = require_paired_panel(
        indexed, condition=condition, alphas=alphas
    )
    summary = summarize_sample_ids(
        panel,
        sample_ids=sample_ids,
        alphas=alphas,
        baseline_alpha=baseline_alpha,
        n_resamples=n_resamples,
        seed=seed,
        bool_metrics=bool_metrics,
    )
    strata: dict[str, Any] = {}
    for (
        dataset,
        mc_endpoint,
        leakage_metadata,
    ), stratum_ids in analysis_stratum_sample_ids(
        sample_ids,
        panel,
        alphas=alphas,
        baseline_alpha=baseline_alpha,
    ).items():
        stratum_summary = summarize_sample_ids(
            panel,
            sample_ids=stratum_ids,
            alphas=alphas,
            baseline_alpha=baseline_alpha,
            n_resamples=n_resamples,
            seed=seed,
            bool_metrics=bool_metrics,
        )
        stratum_summary["dataset"] = dataset
        stratum_summary["mc_endpoint"] = mc_endpoint
        if leakage_metadata:
            stratum_summary["truthfulqa_leakage_metadata"] = dict(leakage_metadata)
        strata[analysis_stratum_key(dataset, mc_endpoint, leakage_metadata)] = (
            stratum_summary
        )

    return {
        "condition": condition,
        "summary_scope": "pooled_across_dataset_and_mc_endpoint",
        **summary,
        "dataset_endpoint_summaries": strata,
    }


def interaction_counts(
    sample_ids: list[str],
    rows_by_id: dict[str, RowGroup],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    counts: defaultdict[str, float] = defaultdict(float)
    agreement_values: list[float] = []
    for sample_id in sample_ids:
        rows = rows_by_id[sample_id]
        per_group: Counter[str] = Counter()
        for row in rows:
            mc = bool(BOOL_METRICS[PRIMARY_MC_RATE_METRIC](row))
            open_correct = effective_open_correct(row)
            key = f"mc_{'R' if mc else 'W'}__open_{'R' if open_correct else 'W'}"
            per_group[key] += 1
        for key, count in per_group.items():
            counts[key] += float(count) / len(rows)
        agreement_values.append(
            float(per_group["mc_R__open_R"] + per_group["mc_W__open_W"]) / len(rows)
        )
    return {
        "counts": {key: round(value, 6) for key, value in sorted(counts.items())},
        "agreement_rate": bootstrap_mean_summary(
            agreement_values,
            n_resamples=n_resamples,
            seed=seed,
        ),
    }


def flip_table(
    sample_ids: list[str],
    baseline_rows: dict[str, RowGroup],
    comparison_rows: dict[str, RowGroup],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    counts: defaultdict[str, float] = defaultdict(float)
    for sample_id in sample_ids:
        paired_rows = pair_replicate_groups(
            baseline_rows[sample_id],
            comparison_rows[sample_id],
            sample_id=sample_id,
        )
        per_group: Counter[str] = Counter()
        for replicate, base, comp in paired_rows:
            base_mc = bool(BOOL_METRICS[PRIMARY_MC_RATE_METRIC](base))
            comp_mc = bool(BOOL_METRICS[PRIMARY_MC_RATE_METRIC](comp))
            base_open = effective_open_correct(base)
            comp_open = effective_open_correct(comp)
            if not base_mc and comp_mc and base_open and not comp_open:
                effective_grade = effective_open_grade(comp)
                failure_type = str(comp["open_grade"].get("failure_type"))
                if not bool(effective_grade["attempted"]):
                    bucket = "mc_W_to_R__open_R_to_not_attempted"
                elif failure_type == "wrong_entity":
                    bucket = "mc_W_to_R__open_R_to_wrong_entity"
                else:
                    bucket = "mc_W_to_R__open_R_to_alias_or_other"
                per_group[bucket] += 1
                rows.append(
                    {
                        "base_sample_id": sample_id,
                        "sample_id": comp["sample_id"],
                        "option_order_replicate": replicate,
                        "dataset": comp.get("dataset"),
                        "bucket": bucket,
                        "question": comp.get("question"),
                        "baseline_open_response": base["open_generation"]["response"],
                        "comparison_open_response": comp["open_generation"]["response"],
                        "comparison_open_grade": comp["open_grade"],
                        "comparison_effective_open_grade": effective_grade,
                    }
                )
        for bucket, count in per_group.items():
            counts[bucket] += float(count) / len(paired_rows)
    return {
        "counts": {key: round(value, 6) for key, value in sorted(counts.items())},
        "rows": rows,
        "count_unit": "base_sample_id_mean_over_option_order_replicates",
    }


def pair_replicate_groups(
    baseline: RowGroup,
    comparison: RowGroup,
    *,
    sample_id: str,
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    baseline_by_key = {replicate_key(row): row for row in baseline}
    comparison_by_key = {replicate_key(row): row for row in comparison}
    if set(baseline_by_key) != set(comparison_by_key):
        raise ValueError(
            f"Unpaired SIMID replicate rows for sample_id={sample_id}: "
            f"missing vs baseline={sorted(set(baseline_by_key) - set(comparison_by_key))[:5]}; "
            f"extra={sorted(set(comparison_by_key) - set(baseline_by_key))[:5]}"
        )
    return [
        (key, baseline_by_key[key], comparison_by_key[key])
        for key in sorted(baseline_by_key)
    ]


def per_item_slopes(
    sample_ids: list[str],
    panel: Panel,
    alphas: list[float],
    metric: MetricFn,
) -> np.ndarray:
    x = np.array(alphas, dtype=float)
    x_centered = x - x.mean()
    ss_x = float((x_centered**2).sum())
    if ss_x == 0.0:
        raise ValueError("Need at least two distinct alphas for slope summaries")
    slopes = np.empty(len(sample_ids), dtype=float)
    for idx, sample_id in enumerate(sample_ids):
        y = np.array(
            [group_metric_value(panel[alpha][sample_id], metric) for alpha in alphas],
            dtype=float,
        )
        slopes[idx] = float((x_centered * (y - y.mean())).sum() / ss_x)
    return slopes


def require_selected_control_replicate_pairing(
    *,
    selected_panel: Panel,
    control_panel: Panel,
    sample_ids: list[str],
    alphas: list[float],
    control: str,
) -> None:
    for alpha in alphas:
        for sample_id in sample_ids:
            selected_replicates = replicate_key_set(selected_panel[alpha][sample_id])
            control_replicates = replicate_key_set(control_panel[alpha][sample_id])
            if selected_replicates != control_replicates:
                raise ValueError(
                    f"Selected/control replicate sets differ for control={control}, "
                    f"sample_id={sample_id}, alpha={alpha}; "
                    f"missing vs selected="
                    f"{sorted(selected_replicates - control_replicates)[:5]}; "
                    f"extra={sorted(control_replicates - selected_replicates)[:5]}"
                )


def selected_minus_control_slope_summaries(
    indexed: dict[str, Panel],
    *,
    selected_condition: str,
    control_conditions: list[str],
    alphas: list[float],
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    selected_ids, selected_panel = require_paired_panel(
        indexed, condition=selected_condition, alphas=alphas
    )
    summaries: dict[str, Any] = {}
    for control in control_conditions:
        control_ids, control_panel = require_paired_panel(
            indexed, condition=control, alphas=alphas
        )
        if set(control_ids) != set(selected_ids):
            raise ValueError(
                f"Selected/control sample sets differ for control={control}; "
                "SIMID slope summaries require item-level pairing"
            )
        sample_ids = sorted(selected_ids)
        require_selected_control_replicate_pairing(
            selected_panel=selected_panel,
            control_panel=control_panel,
            sample_ids=sample_ids,
            alphas=alphas,
            control=control,
        )
        metric_summaries: dict[str, Any] = {}
        for metric_name in ("mc_full_margin", "open_first3_margin", "open_full_margin"):
            metric = CONTINUOUS_METRICS[metric_name]
            selected_slopes = per_item_slopes(
                sample_ids, selected_panel, alphas, metric
            )
            control_slopes = per_item_slopes(sample_ids, control_panel, alphas, metric)
            metric_summaries[metric_name] = (
                paired_bootstrap_continuous_mean_difference_raw(
                    control_slopes,
                    selected_slopes,
                    n_resamples=n_resamples,
                    seed=seed,
                )
            )
        summaries[control] = metric_summaries
    return summaries


def _audit_queue_append(
    queue: list[dict[str, Any]],
    seen: set[tuple[str, str, float, str]],
    row: dict[str, Any],
    *,
    reason: str,
    extra: dict[str, Any] | None = None,
) -> None:
    key = (
        str(row["sample_id"]),
        str(row["condition"]),
        float(row["alpha"]),
        reason,
    )
    if key in seen:
        return
    seen.add(key)
    item = {
        "sample_id": row["sample_id"],
        "condition": row["condition"],
        "alpha": row["alpha"],
        "dataset": row.get("dataset"),
        "mc_endpoint": row.get("mc_endpoint"),
        "question": row.get("question"),
        "response": row["open_generation"]["response"],
        "gold_aliases": row.get("gold_aliases"),
        "open_grade": row.get("open_grade"),
        "effective_open_grade": effective_open_grade(row),
        "open_adjudication": row.get("open_adjudication"),
        "mc_letter_likelihood": {
            "chosen_letter": row.get("mc_letter_likelihood", {}).get("chosen_letter"),
            "chosen_index": row.get("mc_letter_likelihood", {}).get("chosen_index"),
            "chosen_is_gold": row.get("mc_letter_likelihood", {}).get("chosen_is_gold"),
        },
        "reason": reason,
    }
    if extra:
        item.update(extra)
    queue.append(item)


def build_alias_audit_queue(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    seen: set[tuple[str, str, float, str]] = set()
    for row in rows:
        grade = row["open_grade"]
        adjudication = row.get("open_adjudication")
        effective_grade = effective_open_grade(row)
        mc_correct = bool(BOOL_METRICS[PRIMARY_MC_RATE_METRIC](row))
        open_correct = bool(grade["correct"])
        attempted = bool(grade["attempted"])

        if isinstance(adjudication, dict):
            if effective_grade["source"] == "adjudication_unknown":
                _audit_queue_append(
                    queue,
                    seen,
                    row,
                    reason="open_adjudication_unknown_or_error",
                )
            elif (
                effective_grade["source"] == "adjudication"
                and bool(effective_grade["correct"]) != open_correct
            ):
                _audit_queue_append(
                    queue,
                    seen,
                    row,
                    reason="judge_disagreed_with_deterministic_alias_grade",
                    extra={
                        "deterministic_open_correct": open_correct,
                        "adjudicated_open_correct": bool(effective_grade["correct"]),
                    },
                )
            continue

        if mc_correct != open_correct:
            _audit_queue_append(
                queue,
                seen,
                row,
                reason="unadjudicated_mc_open_disagreement_requires_adjudication",
            )

        audit_reason: str | None = None
        if bool(grade["correct"]):
            if row.get("dataset") != "triviaqa_bridge":
                audit_reason = (
                    "unadjudicated_non_bridge_deterministic_alias_correct_"
                    "requires_adjudication"
                )
            elif grade.get("match_tier") != "exact":
                audit_reason = (
                    "unadjudicated_non_exact_deterministic_alias_correct_requires_audit"
                )
            if audit_reason is not None:
                _audit_queue_append(
                    queue,
                    seen,
                    row,
                    reason=audit_reason,
                )
            continue
        if row.get("dataset") != "triviaqa_bridge":
            _audit_queue_append(
                queue,
                seen,
                row,
                reason=(
                    "unadjudicated_non_bridge_not_attempted_requires_adjudication"
                    if not attempted
                    else "unadjudicated_non_bridge_deterministic_alias_incorrect_"
                    "requires_adjudication"
                ),
            )
        margin = float(row["open_margins"]["full"]["margin"])
        if margin <= 0.0:
            continue
        _audit_queue_append(
            queue,
            seen,
            row,
            reason="unadjudicated_open_incorrect_but_gold_margin_positive",
            extra={"open_full_margin": margin},
        )
    return queue


def open_calibration_case_id(row: dict[str, Any]) -> str:
    payload = json.dumps(
        open_adjudication_dedup_key(row),
        ensure_ascii=True,
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"simid_open_cal_{digest}"


def open_calibration_stratum(row: dict[str, Any]) -> str:
    grade = effective_open_grade(row)
    return (
        f"dataset={row.get('dataset', 'unknown')}::"
        f"judge_grade={grade.get('judge_grade')}::"
        f"deterministic_correct={deterministic_open_correct(row)}"
    )


def make_open_calibration_queue_row(
    row: dict[str, Any],
    *,
    sample_source: str,
) -> dict[str, Any]:
    grade = effective_open_grade(row)
    return {
        "schema_version": SIMID_OPEN_CALIBRATION_QUEUE_SCHEMA_VERSION,
        "calibration_case_id": open_calibration_case_id(row),
        "sample_source": sample_source,
        "sampling_stratum": open_calibration_stratum(row),
        **open_adjudication_key_payload(row),
        "base_sample_id": row.get("base_sample_id"),
        "dataset": row.get("dataset"),
        "mc_endpoint": row.get("mc_endpoint"),
        "question": row.get("question"),
        "gold_aliases": row.get("gold_aliases"),
        "response": open_adjudication_response_payload(row),
        "deterministic_open_grade": row.get("open_grade"),
        "primary_judge_grade": grade.get("judge_grade"),
        "primary_effective_open_grade": grade,
        "primary_open_adjudication": row.get("open_adjudication"),
        "judge_disagrees_with_deterministic": (
            bool(grade["correct"]) != deterministic_open_correct(row)
            if grade["source"] == "adjudication"
            else None
        ),
    }


def _dedup_calibration_candidates(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_case_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not eligible_for_open_adjudication(row):
            continue
        grade = effective_open_grade(row)
        if grade["source"] != "adjudication":
            continue
        case_id = open_calibration_case_id(row)
        by_case_id.setdefault(case_id, row)
    return [by_case_id[case_id] for case_id in sorted(by_case_id)]


def build_open_calibration_queue(
    rows: list[dict[str, Any]],
    *,
    stratified_per_cell: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Build a second-rater queue with disagreement rows plus a stratified panel."""
    if stratified_per_cell < 0:
        raise ValueError("stratified_per_cell must be non-negative")

    candidates = _dedup_calibration_candidates(rows)
    audit_rows = [
        row
        for row in candidates
        if bool(effective_open_grade(row)["correct"]) != deterministic_open_correct(row)
    ]
    selected_by_case_id: dict[str, dict[str, Any]] = {}
    selected_sources: dict[str, str] = {}
    for row in audit_rows:
        case_id = open_calibration_case_id(row)
        selected_by_case_id[case_id] = row
        selected_sources[case_id] = "audit_queue_disagreement"

    if stratified_per_cell:
        rng = random.Random(seed)
        by_stratum: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            case_id = open_calibration_case_id(row)
            if case_id in selected_by_case_id:
                continue
            by_stratum[open_calibration_stratum(row)].append(row)
        for stratum, stratum_rows in sorted(by_stratum.items()):
            ordered = sorted(stratum_rows, key=open_calibration_case_id)
            n_take = min(stratified_per_cell, len(ordered))
            for row in sorted(
                rng.sample(ordered, n_take),
                key=open_calibration_case_id,
            ):
                case_id = open_calibration_case_id(row)
                selected_by_case_id[case_id] = row
                selected_sources[case_id] = f"stratified_panel::{stratum}"

    return [
        make_open_calibration_queue_row(
            selected_by_case_id[case_id],
            sample_source=selected_sources[case_id],
        )
        for case_id in sorted(selected_by_case_id)
    ]


def write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def chosen_letter_distribution(groups: dict[str, RowGroup]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    n_rows = 0
    for rows in groups.values():
        for row in rows:
            letter = row.get("mc_letter_likelihood", {}).get("chosen_letter")
            counts[str(letter) if letter is not None else "missing"] += 1
            n_rows += 1
    return {
        "counts": dict(sorted(counts.items())),
        "n_rows": n_rows,
    }


def gold_position_correctness_table(groups: dict[str, RowGroup]) -> dict[str, Any]:
    counts: Counter[int] = Counter()
    correct_counts: Counter[int] = Counter()
    n_options_values: set[int] = set()
    n_rows = 0
    for rows in groups.values():
        for row in rows:
            gold_indices = [int(idx) for idx in row.get("gold_option_indices", [])]
            if len(gold_indices) != 1:
                continue
            position = gold_indices[0]
            counts[position] += 1
            correct_counts[position] += int(BOOL_METRICS[PRIMARY_MC_RATE_METRIC](row))
            n_options_values.add(len(row.get("mc_options", [])))
            n_rows += 1

    n_options = max(n_options_values) if n_options_values else 0
    positions: dict[str, dict[str, Any]] = {}
    for position in range(n_options):
        n_at_position = counts.get(position, 0)
        correct = correct_counts.get(position, 0)
        positions[str(position)] = {
            "n": n_at_position,
            "correct": correct,
            "rate": correct / n_at_position if n_at_position else None,
        }
    return {
        "positions": positions,
        "n_rows": n_rows,
        "n_options_values": sorted(n_options_values),
        "rate_unit": PRIMARY_MC_RATE_METRIC,
    }


def option_order_stability_gate(groups: dict[str, RowGroup]) -> dict[str, Any]:
    replicate_values: defaultdict[int, list[float]] = defaultdict(list)
    replicate_sets_by_base_id: dict[str, set[int]] = {}
    flipped_base_sample_ids: list[str] = []
    n_rows = 0

    try:
        for base_sample_id, rows in sorted(groups.items()):
            rows_by_replicate: dict[int, dict[str, Any]] = {}
            for row in rows:
                replicate = require_option_order_replicate(row)
                if replicate in rows_by_replicate:
                    return {
                        "passed": False,
                        "reason": (
                            "duplicate option_order_replicate metadata for "
                            f"base_sample_id={base_sample_id!r}, replicate={replicate}"
                        ),
                    }
                rows_by_replicate[replicate] = row
                value = float(BOOL_METRICS[PRIMARY_MC_RATE_METRIC](row))
                replicate_values[replicate].append(value)
                n_rows += 1

            replicate_sets_by_base_id[base_sample_id] = set(rows_by_replicate)
            item_values = {
                bool(BOOL_METRICS[PRIMARY_MC_RATE_METRIC](row))
                for row in rows_by_replicate.values()
            }
            if len(item_values) > 1:
                flipped_base_sample_ids.append(base_sample_id)
    except ValueError as exc:
        return {
            "passed": False,
            "reason": str(exc),
            "rate_unit": PRIMARY_MC_RATE_METRIC,
        }

    rates = {
        f"ord:{replicate}": float(np.mean(values))
        for replicate, values in sorted(replicate_values.items())
    }
    if len(rates) < 2:
        return {
            "passed": False,
            "reason": "requires at least two option-order replicates",
            "replicate_rates": rates,
            "n_base_items": len(groups),
            "n_rows": n_rows,
            "rate_unit": PRIMARY_MC_RATE_METRIC,
        }

    expected_replicates = next(iter(replicate_sets_by_base_id.values()), set())
    mismatched = {
        base_sample_id: sorted(replicates)
        for base_sample_id, replicates in replicate_sets_by_base_id.items()
        if replicates != expected_replicates
    }
    if mismatched:
        return {
            "passed": False,
            "reason": "inconsistent option_order_replicate sets across base items",
            "expected_replicates": sorted(expected_replicates),
            "mismatched_base_items": dict(sorted(mismatched.items())),
            "replicate_rates": rates,
            "n_base_items": len(groups),
            "n_rows": n_rows,
            "rate_unit": PRIMARY_MC_RATE_METRIC,
        }

    spread = max(rates.values()) - min(rates.values())
    n_base_items = len(groups)
    item_flip_count = len(flipped_base_sample_ids)
    return {
        "passed": spread <= OPTION_ORDER_STABILITY_THRESHOLD,
        "replicate_rates": rates,
        "global_max_rate_spread": spread,
        "max_rate_spread": spread,
        "item_flip_count": item_flip_count,
        "item_flip_rate": item_flip_count / n_base_items if n_base_items else 0.0,
        "item_flip_base_sample_ids": flipped_base_sample_ids,
        "chosen_letter_distribution": chosen_letter_distribution(groups),
        "gold_position_correctness": gold_position_correctness_table(groups),
        "expected_replicates": sorted(expected_replicates),
        "n_base_items": n_base_items,
        "n_rows": n_rows,
        "threshold": OPTION_ORDER_STABILITY_THRESHOLD,
        "rate_unit": PRIMARY_MC_RATE_METRIC,
    }


def gold_position_balance_gate(groups: dict[str, RowGroup]) -> dict[str, Any]:
    counts: Counter[int] = Counter()
    n_options_values: set[int] = set()
    n_rows = 0
    for rows in groups.values():
        for row in rows:
            gold_indices = [int(idx) for idx in row.get("gold_option_indices", [])]
            if len(gold_indices) != 1:
                continue
            counts[gold_indices[0]] += 1
            n_options_values.add(len(row.get("mc_options", [])))
            n_rows += 1
    if not counts or len(n_options_values) != 1:
        return {
            "passed": False,
            "reason": "requires single-gold rows with a fixed option count",
            "counts": dict(sorted(counts.items())),
        }
    n_options = n_options_values.pop()
    expected = n_rows / n_options if n_options else 0.0
    max_abs_deviation = (
        max(abs(counts.get(idx, 0) - expected) for idx in range(n_options))
        if expected
        else 0.0
    )
    max_abs_deviation_share = max_abs_deviation / n_rows if n_rows else 0.0
    return {
        "passed": max_abs_deviation_share <= 0.20 or n_rows < 30,
        "counts": {str(idx): counts.get(idx, 0) for idx in range(n_options)},
        "n_rows": n_rows,
        "n_options": n_options,
        "max_abs_deviation_share": max_abs_deviation_share,
        "threshold": 0.20,
        "small_n_leniency": n_rows < 30,
    }


def option_length_balance_gate(groups: dict[str, RowGroup]) -> dict[str, Any]:
    diffs: list[float] = []
    for rows in groups.values():
        for row in rows:
            options = [str(option) for option in row.get("mc_options", [])]
            gold_indices = {int(idx) for idx in row.get("gold_option_indices", [])}
            if not options or not gold_indices:
                continue
            gold_lens = [
                len(options[idx].split()) for idx in gold_indices if idx < len(options)
            ]
            distractor_lens = [
                len(option.split())
                for idx, option in enumerate(options)
                if idx not in gold_indices
            ]
            if not gold_lens or not distractor_lens:
                continue
            diffs.append(float(np.mean(gold_lens) - np.mean(distractor_lens)))
    if not diffs:
        return {"passed": False, "reason": "no length-comparable MC rows"}
    mean_diff = float(np.mean(diffs))
    return {
        "passed": abs(mean_diff) <= 0.75,
        "mean_gold_minus_distractor_word_length": mean_diff,
        "n_rows": len(diffs),
        "threshold_abs_words": 0.75,
    }


def open_margin_alignment_gate(groups: dict[str, RowGroup]) -> dict[str, Any]:
    correct_margins: list[float] = []
    incorrect_margins: list[float] = []
    for rows in groups.values():
        for row in rows:
            margin = float(row["open_margins"]["full"]["margin"])
            if effective_open_correct(row):
                correct_margins.append(margin)
            else:
                incorrect_margins.append(margin)
    if not correct_margins or not incorrect_margins:
        return {
            "passed": False,
            "reason": "requires both open-correct and open-incorrect rows",
            "n_correct": len(correct_margins),
            "n_incorrect": len(incorrect_margins),
        }
    correct_mean = float(np.mean(correct_margins))
    incorrect_mean = float(np.mean(incorrect_margins))
    return {
        "passed": correct_mean > incorrect_mean,
        "correct_mean_open_full_margin": correct_mean,
        "incorrect_mean_open_full_margin": incorrect_mean,
        "n_correct": len(correct_margins),
        "n_incorrect": len(incorrect_margins),
    }


def run_phase0_gates(
    indexed: dict[str, Panel],
    *,
    noop_tolerance: float,
) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    if "unhooked" in indexed and "selected" in indexed:
        unhooked = indexed["unhooked"].get(0.0)
        selected = indexed["selected"].get(0.0)
        if unhooked is None or selected is None:
            gates["noop_equivalence"] = {
                "passed": False,
                "reason": "missing unhooked or selected alpha=0.0",
            }
        else:
            try:
                assert_noop_equivalence(
                    flatten_groups(unhooked),
                    flatten_groups(selected),
                    tolerance=noop_tolerance,
                )
            except AssertionError as exc:
                gates["noop_equivalence"] = {"passed": False, "reason": str(exc)}
            else:
                gates["noop_equivalence"] = {"passed": True}

    if "selected" in indexed and 0.0 in indexed["selected"]:
        groups = indexed["selected"][0.0]
        bridge_groups = [
            group
            for group in groups.values()
            if any(row.get("dataset") == "triviaqa_bridge" for row in group)
        ]
        if bridge_groups:
            correctness = [
                group_metric_value(group, BOOL_METRICS[PRIMARY_MC_RATE_METRIC])
                for group in bridge_groups
            ]
            n_options = max(
                len(row.get("mc_options", []))
                for group in bridge_groups
                for row in group
            )
            random_rate = 1.0 / n_options if n_options else 0.0
            observed = float(np.mean(correctness))
            gates["bridge_synthetic_mc_sanity"] = {
                "passed": observed > random_rate + 0.05 and observed < 0.98,
                "observed_rate": observed,
                "random_rate": random_rate,
                "n_base_items": len(bridge_groups),
                "replicate_policy": "mean_within_base_sample_id",
            }
            bridge_groups_by_id = {
                sample_id: group
                for sample_id, group in groups.items()
                if any(row.get("dataset") == "triviaqa_bridge" for row in group)
            }
            gates["bridge_option_order_stability"] = option_order_stability_gate(
                bridge_groups_by_id
            )
            gates["bridge_gold_position_balance"] = gold_position_balance_gate(
                bridge_groups_by_id
            )
            gates["bridge_option_length_balance"] = option_length_balance_gate(
                bridge_groups_by_id
            )
            gates["bridge_open_margin_alignment"] = open_margin_alignment_gate(
                bridge_groups_by_id
            )
    return gates


def format_ci(result: dict[str, Any], *, unit: str = "") -> str:
    if "estimate_pp" in result:
        ci = result["ci_pp"]
        return (
            f"{result['estimate_pp']:.2f}{unit} [{ci['lower']:.2f}, {ci['upper']:.2f}]"
        )
    ci = result["ci"]
    return f"{result['estimate']:.4f}{unit} [{ci['lower']:.4f}, {ci['upper']:.4f}]"


def format_rate(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def format_phase0_gate_line(name: str, gate: dict[str, Any]) -> str:
    status = "PASS" if gate.get("passed") else "FAIL"
    if name == "bridge_option_order_stability":
        spread = gate.get("global_max_rate_spread", gate.get("max_rate_spread"))
        flip_count = gate.get("item_flip_count")
        flip_rate = gate.get("item_flip_rate")
        n_base_items = gate.get("n_base_items")
        if spread is not None and flip_count is not None and flip_rate is not None:
            return (
                f"- {name}: {status} "
                f"(global replicate spread={float(spread):.4f}; "
                f"item flips={flip_count}/{n_base_items}, "
                f"rate={float(flip_rate):.4f})"
            )
    reason = gate.get("reason")
    suffix = f" ({reason})" if reason else ""
    return f"- {name}: {status}{suffix}"


def append_option_order_gate_details(
    lines: list[str],
    gate: dict[str, Any],
) -> None:
    rates = gate.get("replicate_rates", {})
    if rates:
        rendered_rates = ", ".join(
            f"{key}={format_rate(value)}" for key, value in sorted(rates.items())
        )
        lines.append(f"  - replicate rates: {rendered_rates}")
    distribution = gate.get("chosen_letter_distribution", {}).get("counts", {})
    if distribution:
        rendered_counts = ", ".join(
            f"{letter}={count}" for letter, count in sorted(distribution.items())
        )
        lines.append(f"  - chosen letters: {rendered_counts}")
    positions = gate.get("gold_position_correctness", {}).get("positions", {})
    if positions:
        rendered_positions = []
        for position, payload in sorted(
            positions.items(), key=lambda item: int(item[0])
        ):
            rendered_positions.append(
                f"pos {position}: {payload['correct']}/{payload['n']}="
                f"{format_rate(payload['rate'])}"
            )
        lines.append("  - gold-position correctness: " + "; ".join(rendered_positions))
    flipped = gate.get("item_flip_base_sample_ids", [])
    if flipped:
        preview = ", ".join(str(sample_id) for sample_id in flipped[:10])
        suffix = " ..." if len(flipped) > 10 else ""
        lines.append(f"  - flipped base items: {preview}{suffix}")


def write_report(results: dict[str, Any], path: Path) -> None:
    open_grading = results.get("open_grading", {})
    open_correct_metric = str(open_grading.get("metric_correct", "open_correct"))
    open_attempted_metric = str(open_grading.get("metric_attempted", "open_attempted"))
    judge_backed = open_correct_metric == "adjudicated_open_correct"
    claimable_open = open_grading.get("claimable_open_correctness") is True
    if judge_backed and claimable_open:
        open_grading_note = (
            "Open correctness is reported as adjudicated_open_correct. Judge "
            "adjudication is fully loaded for the analyzed rows, and the "
            "open-correctness calibration evidence passed the recorded "
            "claimability policy."
        )
    elif judge_backed:
        open_grading_note = (
            "Open correctness is reported as adjudicated_open_correct for "
            "diagnostics: judge "
            "adjudication is used when present, deterministic alias grading is only "
            "a fallback for unadjudicated rows, and deterministic alias diagnostics "
            "are kept separate. These open metrics remain diagnostic-only until "
            "judge calibration evidence is recorded."
        )
    else:
        open_grading_note = (
            "Open correctness is deterministic alias-grader correctness "
            "(not adjudicated human/judge correctness). Claimable open-ended results "
            "require adjudication or an explicit audit."
        )
    lines = [
        "# SIMID Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "All primary estimates use base-item pairing and bootstrap 95% CIs. "
        "The primary MC endpoint is the lettered forced-choice likelihood prompt.",
        open_grading_note,
        "",
    ]
    if open_grading:
        source_counts = open_grading.get("effective_grade_source_counts", {})
        if source_counts:
            rendered_counts = ", ".join(
                f"{key}={value}" for key, value in sorted(source_counts.items())
            )
            lines.append(f"Open grade sources: {rendered_counts}")
        if open_grading.get("claimable_open_correctness") is False:
            blocker = open_grading.get("claimability_blocker", "unspecified")
            lines.append(f"Open correctness claimability: blocked ({blocker}).")
        elif open_grading.get("claimable_open_correctness") is True:
            lines.append("Open correctness claimability: passed.")
            calibration = open_grading.get("calibration")
            if isinstance(calibration, dict):
                calibration_path = calibration.get("path")
                kappa = calibration.get("irr", {}).get("cohen_kappa")
                ac1 = calibration.get("irr", {}).get("gwet_ac1")
                if calibration_path or kappa is not None or ac1 is not None:
                    rendered = []
                    if calibration_path:
                        rendered.append(f"calibration={calibration_path}")
                    if kappa is not None:
                        rendered.append(f"kappa={float(kappa):.4f}")
                    if ac1 is not None:
                        rendered.append(f"AC1={float(ac1):.4f}")
                    lines.append("Open calibration evidence: " + ", ".join(rendered))
        claim_contract = open_grading.get("open_correctness_claim_contract")
        if claim_contract:
            lines.append(f"Open correctness claim contract: {claim_contract}.")
        lines.append("")
    for condition, summary in results["conditions"].items():
        lines.append(f"## {condition}")
        lines.append(
            "Pooled aggregate scope: all datasets and MC endpoints in the selected "
            "condition panel."
        )
        lines.append(f"Paired items: {summary['n_paired_items']}")
        baseline = str(summary["baseline_alpha"])
        baseline_open = summary["rates"][baseline][open_correct_metric]
        baseline_mc = summary["rates"][baseline][PRIMARY_MC_RATE_METRIC]
        lines.append(
            "Baseline rates: "
            f"lettered MC={format_ci(baseline_mc)}, "
            f"{open_correct_metric}={format_ci(baseline_open)}"
        )
        for alpha, deltas in summary["paired_deltas_vs_baseline"].items():
            mc = deltas[PRIMARY_MC_RATE_METRIC]
            open_delta = deltas[open_correct_metric]
            attempted = deltas[open_attempted_metric]
            lines.append(
                f"- alpha {alpha}: MC delta {format_ci(mc, unit=' pp')}; "
                f"{open_correct_metric} delta {format_ci(open_delta, unit=' pp')}; "
                f"attempt delta {format_ci(attempted, unit=' pp')}"
            )
        strata = summary.get("dataset_endpoint_summaries", {})
        if strata:
            lines.append("")
            lines.append(
                "Dataset / MC endpoint strata (TruthfulQA leakage split when available):"
            )
            for key in sorted(strata):
                stratum = strata[key]
                stratum_baseline = str(stratum["baseline_alpha"])
                stratum_open = stratum["rates"][stratum_baseline][open_correct_metric]
                stratum_mc = stratum["rates"][stratum_baseline][PRIMARY_MC_RATE_METRIC]
                leakage_metadata = stratum.get("truthfulqa_leakage_metadata", {})
                leakage_label = ""
                if leakage_metadata:
                    rendered = ", ".join(
                        f"{field.removeprefix('truthfulqa_')}={value}"
                        for field, value in leakage_metadata.items()
                    )
                    leakage_label = f" ({rendered})"
                lines.append(
                    f"- {stratum['dataset']} / {stratum['mc_endpoint']}"
                    f"{leakage_label}: "
                    f"n={stratum['n_paired_items']}; "
                    f"lettered MC={format_ci(stratum_mc)}; "
                    f"{open_correct_metric}={format_ci(stratum_open)}"
                )
                for alpha, deltas in stratum["paired_deltas_vs_baseline"].items():
                    mc = deltas[PRIMARY_MC_RATE_METRIC]
                    open_delta = deltas[open_correct_metric]
                    lines.append(
                        f"  - alpha {alpha}: MC delta {format_ci(mc, unit=' pp')}; "
                        f"{open_correct_metric} delta "
                        f"{format_ci(open_delta, unit=' pp')}"
                    )
        lines.append("")
    if results.get("selected_minus_control_slopes"):
        lines.append("## Selected Minus Control Slopes")
        for control, metrics in results["selected_minus_control_slopes"].items():
            lines.append(f"- {control}:")
            for metric_name, result in metrics.items():
                lines.append(f"  - {metric_name}: {format_ci(result)}")
        lines.append("")
    if results.get("phase0_gates"):
        lines.append("## Phase 0 Gates")
        for name, gate in results["phase0_gates"].items():
            lines.append(format_phase0_gate_line(name, gate))
            if name == "bridge_option_order_stability":
                append_option_order_gate_details(lines, gate)
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--baseline-alpha", type=float, default=0.0)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--alphas", type=float, nargs="+", default=None)
    parser.add_argument("--n-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--phase0-gates", action="store_true")
    parser.add_argument("--noop-tolerance", type=float, default=1e-5)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--report-md", default=None)
    parser.add_argument("--alias-audit-output", default=None)
    parser.add_argument(
        "--open-calibration-queue-output",
        default=None,
        help=(
            "Optional JSONL queue for a second-rater open-correctness calibration "
            "pass. Relative paths are resolved under --run-dir."
        ),
    )
    parser.add_argument(
        "--open-calibration-stratified-per-cell",
        type=int,
        default=3,
        help=(
            "Number of non-disagreement rows to sample per "
            "dataset/judge-grade/deterministic-correctness stratum when writing "
            "--open-calibration-queue-output."
        ),
    )
    parser.add_argument("--adjudicate-open", action="store_true")
    parser.add_argument("--judge-model", default=DEFAULT_SIMID_OPEN_JUDGE_MODEL)
    parser.add_argument(
        "--adjudication-mode",
        choices=["batch"],
        default="batch",
        help="SIMID open adjudication mode. Batch is the only supported mode.",
    )
    parser.add_argument(
        "--adjudication-output",
        default="open_adjudication.jsonl",
        help="Path for separate SIMID open adjudication JSONL output.",
    )
    parser.add_argument(
        "--open-calibration-summary",
        default=None,
        help=(
            "Optional L4-style SIMID open-correctness calibration summary JSON. "
            "Relative paths are resolved under --run-dir."
        ),
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--prompt-cache-retention",
        choices=["in-memory", "24h"],
        default=None,
    )
    parser.add_argument("--batch-max-enqueued-tokens", type=int, default=None)
    parser.add_argument(
        "--adjudication-limit",
        type=int,
        default=None,
        help="Limit new adjudication requests for canary runs.",
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args(argv)


def refuse_analysis_output_overwrite(
    paths: list[Path], *, allow_overwrite: bool
) -> None:
    if allow_overwrite:
        return
    existing = [path for path in dict.fromkeys(paths) if path.exists()]
    if existing:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing SIMID analysis outputs: {rendered}. "
            "Pass --allow-overwrite to replace them explicitly."
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    output_json = (
        Path(args.output_json) if args.output_json else run_dir / "results.json"
    )
    report_md = Path(args.report_md) if args.report_md else run_dir / "report.md"
    alias_queue_path = (
        Path(args.alias_audit_output)
        if args.alias_audit_output
        else run_dir / "alias_audit_queue.jsonl"
    )
    open_calibration_queue_path = (
        Path(args.open_calibration_queue_output)
        if args.open_calibration_queue_output
        else None
    )
    if (
        open_calibration_queue_path is not None
        and not open_calibration_queue_path.is_absolute()
    ):
        open_calibration_queue_path = run_dir / open_calibration_queue_path
    adjudication_output = Path(args.adjudication_output)
    if not adjudication_output.is_absolute():
        adjudication_output = run_dir / adjudication_output
    open_calibration_summary_path = (
        Path(args.open_calibration_summary) if args.open_calibration_summary else None
    )
    if (
        open_calibration_summary_path is not None
        and not open_calibration_summary_path.is_absolute()
    ):
        open_calibration_summary_path = run_dir / open_calibration_summary_path
    refuse_analysis_output_overwrite(
        [
            path
            for path in [
                output_json,
                report_md,
                alias_queue_path,
                open_calibration_queue_path,
            ]
            if path is not None
        ],
        allow_overwrite=args.allow_overwrite,
    )
    output_targets: list[str | os.PathLike[str]] = [
        output_json,
        report_md,
        alias_queue_path,
    ]
    if args.adjudicate_open:
        output_targets.append(adjudication_output)
    if open_calibration_queue_path is not None:
        output_targets.append(open_calibration_queue_path)
    provenance = start_run_provenance(
        args,
        primary_target=output_json,
        output_targets=output_targets,
    )
    status = "completed"
    extra: dict[str, Any] = {}
    try:
        rows = load_run_rows(run_dir)
        adjudication_run = None
        if args.adjudicate_open:
            if args.adjudication_mode != "batch":
                raise ValueError(
                    "SIMID open adjudication currently supports batch only"
                )
            adjudication_run = adjudicate_open_responses_batch(
                rows,
                output_path=adjudication_output,
                judge_model=args.judge_model,
                api_key=args.api_key,
                prompt_cache_retention=args.prompt_cache_retention,
                max_enqueued_tokens=args.batch_max_enqueued_tokens,
                limit=args.adjudication_limit,
            )
        calibration_summary = load_open_calibration_summary(
            open_calibration_summary_path
        )
        adjudications = load_open_adjudications(adjudication_output)
        n_attached_adjudications = attach_open_adjudications(rows, adjudications)
        bool_metrics = bool_metrics_for_rows(rows)
        open_grading_summary = summarize_open_grading(
            rows,
            adjudication_path=adjudication_output,
            adjudication_run=adjudication_run,
            calibration_summary=calibration_summary,
        )
        indexed = index_rows(rows)
        conditions = args.conditions or sorted(indexed)
        alphas = args.alphas or sorted(
            {alpha for condition in conditions for alpha in indexed[condition]}
        )
        if args.baseline_alpha not in alphas:
            raise ValueError(
                f"baseline alpha {args.baseline_alpha} is not in alpha grid {alphas}"
            )

        condition_summaries = {
            condition: summarize_condition(
                indexed,
                condition=condition,
                alphas=alphas,
                baseline_alpha=args.baseline_alpha,
                n_resamples=args.n_resamples,
                seed=args.seed,
                bool_metrics=bool_metrics,
            )
            for condition in conditions
        }
        controls = [
            condition
            for condition in conditions
            if condition not in {"selected", "unhooked"}
        ]
        slope_summaries = {}
        if "selected" in conditions and controls and len(alphas) >= 2:
            slope_summaries = selected_minus_control_slope_summaries(
                indexed,
                selected_condition="selected",
                control_conditions=controls,
                alphas=alphas,
                n_resamples=args.n_resamples,
                seed=args.seed,
            )
        alias_queue = build_alias_audit_queue(rows)
        write_jsonl_rows(alias_queue_path, alias_queue)
        open_calibration_queue = None
        if open_calibration_queue_path is not None:
            open_calibration_queue = build_open_calibration_queue(
                rows,
                stratified_per_cell=args.open_calibration_stratified_per_cell,
                seed=args.seed,
            )
            write_jsonl_rows(open_calibration_queue_path, open_calibration_queue)

        results = {
            "schema_version": "simid_analysis/v1",
            "run_dir": str(run_dir),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "baseline_alpha": args.baseline_alpha,
            "alphas": alphas,
            "conditions": condition_summaries,
            "selected_minus_control_slopes": slope_summaries,
            "open_grading": open_grading_summary,
            "alias_audit_queue": {
                "path": str(alias_queue_path),
                "n": len(alias_queue),
            },
            "bootstrap": {
                "n_resamples": args.n_resamples,
                "seed": args.seed,
                "confidence": 0.95,
            },
        }
        if (
            open_calibration_queue is not None
            and open_calibration_queue_path is not None
        ):
            results["open_calibration_queue"] = {
                "path": str(open_calibration_queue_path),
                "n": len(open_calibration_queue),
                "schema_version": SIMID_OPEN_CALIBRATION_QUEUE_SCHEMA_VERSION,
                "stratified_per_cell": args.open_calibration_stratified_per_cell,
            }
        if args.phase0_gates:
            results["phase0_gates"] = run_phase0_gates(
                indexed,
                noop_tolerance=args.noop_tolerance,
            )
        output_json.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_report(results, report_md)
        extra["n_rows"] = len(rows)
        extra["n_open_adjudications_loaded"] = n_attached_adjudications
        extra["n_alias_audit_rows"] = len(alias_queue)
        print(f"Wrote SIMID analysis to {output_json} and {report_md}")
    except BaseException as exc:
        status = provenance_status_for_exception(exc)
        extra["error"] = provenance_error_message(exc)
        raise
    finally:
        finish_run_provenance(provenance, status, extra)


if __name__ == "__main__":
    main()
