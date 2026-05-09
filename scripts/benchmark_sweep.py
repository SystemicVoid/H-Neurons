"""Shared alpha-sweep orchestration for claim-bearing benchmark runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
import time
from typing import Any

from tqdm import tqdm

from artifact_io import alpha_jsonl_path, append_jsonl_row
from uncertainty import build_rate_summary
from utils import format_alpha_label


@dataclass(frozen=True)
class AlphaSweepContext:
    benchmark: str
    alpha: float
    alpha_idx: int
    alpha_label: str
    output_path: str
    config_name: str | None = None


@dataclass(frozen=True)
class SweepRecord:
    payload: dict[str, Any]
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class AlphaSweepResult:
    results: dict[str, dict[str, Any]]
    output_paths: dict[str, str]


SampleIdFn = Callable[[Mapping[str, Any]], str]
GenerateRecordFn = Callable[[Mapping[str, Any], AlphaSweepContext], SweepRecord]
WriteRecordFn = Callable[[str, SweepRecord, AlphaSweepContext], None]
SummarizeAlphaFn = Callable[[str], dict[str, Any]]
SetAlphaFn = Callable[[float], None]
AlphaCompleteFn = Callable[[AlphaSweepContext, dict[str, Any], float], None]
PrintAlphaSummaryFn = Callable[[AlphaSweepContext, dict[str, Any]], None]


def alpha_output_path(output_dir: str, alpha: float) -> str:
    return str(alpha_jsonl_path(output_dir, alpha))


def load_existing_row_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()

    ids: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_id = record.get("id")
            if row_id is not None:
                ids.add(str(row_id))
    return ids


def append_jsonl_sweep_record(
    output_path: str,
    sweep_record: SweepRecord,
    _context: AlphaSweepContext,
) -> None:
    append_jsonl_row(output_path, sweep_record.payload, sort_keys=False)


def _default_sample_id(sample: Mapping[str, Any]) -> str:
    return str(sample["id"])


def _default_progress_description(context: AlphaSweepContext) -> str:
    if context.config_name:
        return f"[{context.config_name}] α={context.alpha_label}"
    return f"{context.benchmark} α={context.alpha_label}"


def run_alpha_sweep(
    *,
    benchmark: str,
    samples: Sequence[Mapping[str, Any]],
    alphas: Sequence[float],
    output_dir: str,
    generate_record: GenerateRecordFn,
    summarize_alpha: SummarizeAlphaFn,
    config_name: str | None = None,
    max_samples: int | None = None,
    sample_id: SampleIdFn = _default_sample_id,
    set_alpha: SetAlphaFn | None = None,
    write_record: WriteRecordFn = append_jsonl_sweep_record,
    print_alpha_summary: PrintAlphaSummaryFn | None = None,
    on_alpha_complete: AlphaCompleteFn | None = None,
    alpha_index_start: int = 1,
) -> AlphaSweepResult:
    os.makedirs(output_dir, exist_ok=True)
    selected_samples = (
        list(samples[:max_samples]) if max_samples is not None else list(samples)
    )
    results: dict[str, dict[str, Any]] = {}
    output_paths: dict[str, str] = {}

    for offset, alpha in enumerate(alphas):
        alpha_idx = alpha_index_start + offset
        alpha_label = format_alpha_label(alpha)
        output_path = alpha_output_path(output_dir, alpha)
        output_paths[str(alpha)] = output_path
        context = AlphaSweepContext(
            benchmark=benchmark,
            config_name=config_name,
            alpha=float(alpha),
            alpha_idx=alpha_idx,
            alpha_label=alpha_label,
            output_path=output_path,
        )
        existing_ids = load_existing_row_ids(output_path)
        if set_alpha is not None:
            set_alpha(float(alpha))

        alpha_t0 = time.perf_counter()
        for sample in tqdm(
            selected_samples,
            desc=_default_progress_description(context),
        ):
            if sample_id(sample) in existing_ids:
                continue
            sweep_record = generate_record(sample, context)
            write_record(output_path, sweep_record, context)

        alpha_result = summarize_alpha(output_path)
        results[str(alpha)] = alpha_result
        elapsed_s = round(time.perf_counter() - alpha_t0, 4)
        if print_alpha_summary is not None:
            print_alpha_summary(context, alpha_result)
        if on_alpha_complete is not None:
            on_alpha_complete(context, alpha_result, elapsed_s)

    return AlphaSweepResult(results=results, output_paths=output_paths)


def summarize_faitheval_alpha_file(path: str) -> dict[str, Any]:
    compliant = 0
    total = 0
    parse_failures = 0
    if not os.path.exists(path):
        total = 0
    else:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                if record.get("compliance"):
                    compliant += 1
                is_parse_failure = bool(record.get("parse_failure"))
                if not is_parse_failure and "chosen" in record:
                    is_parse_failure = record.get("chosen") is None
                if is_parse_failure:
                    parse_failures += 1

    return {
        "compliance_rate": round(compliant / total, 4) if total else 0,
        "n_compliant": compliant,
        "n_total": total,
        "parse_failures": parse_failures,
        "compliance": build_rate_summary(
            compliant,
            total,
            count_key="n_compliant",
            total_key="n_total",
        ),
        "parse_failure": build_rate_summary(
            parse_failures,
            total,
            count_key="count",
            total_key="n_total",
        ),
    }


def print_faitheval_alpha_summary(
    context: AlphaSweepContext,
    alpha_result: dict[str, Any],
) -> None:
    n_total = int(alpha_result["n_total"])
    n_compliant = int(alpha_result["n_compliant"])
    parse_failures = int(alpha_result["parse_failures"])
    rate = n_compliant / n_total if n_total else 0.0
    print(
        f"  α={context.alpha_label}: {rate:.1%} compliance "
        f"({n_compliant}/{n_total}), {parse_failures} parse failures"
    )
