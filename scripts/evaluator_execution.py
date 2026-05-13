"""Shared execution helpers for OpenAI-backed evaluator scripts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import time
from pathlib import Path
from typing import Any, cast


ParseResponse = Callable[[str], Any | None]
WriteSuccess = Callable[[dict[str, Any], Any], None]
WriteError = Callable[[dict[str, Any], str, str | None], None]


@dataclass(frozen=True)
class EvaluatorTask:
    """One evaluator request bound to a mutable output record."""

    custom_id: str
    record_index: int
    messages: Sequence[Mapping[str, object]]
    request_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluatorAdapter:
    """Evaluator-specific parser and record writeback hooks."""

    name: str
    parse_response: ParseResponse
    write_success: WriteSuccess
    write_error: WriteError


@dataclass(frozen=True)
class EvaluatorExecutionResult:
    """Counts and optional sync cache stats from a shared evaluator run."""

    submitted: int
    succeeded: int
    errors: int
    cache_stats: Any | None = None


def _record_for_task(
    records: list[dict[str, Any]], task: EvaluatorTask
) -> dict[str, Any]:
    try:
        return records[task.record_index]
    except IndexError as exc:
        raise IndexError(
            f"{task.custom_id}: record_index {task.record_index} is out of range "
            f"for {len(records)} records"
        ) from exc


def _merged_request_kwargs(
    common_request_kwargs: Mapping[str, Any],
    task: EvaluatorTask,
) -> dict[str, Any]:
    return {**common_request_kwargs, **task.request_kwargs}


def validate_unique_custom_ids(
    tasks: Sequence[EvaluatorTask], *, evaluator_name: str
) -> None:
    """Reject duplicate custom IDs before submitting a batch."""
    seen: set[str] = set()
    for task in tasks:
        if task.custom_id in seen:
            raise ValueError(
                f"Duplicate {evaluator_name} evaluator custom_id detected: "
                f"{task.custom_id!r}"
            )
        seen.add(task.custom_id)


def execute_batch_evaluator(
    *,
    records: list[dict[str, Any]],
    tasks: Sequence[EvaluatorTask],
    adapter: EvaluatorAdapter,
    client: Any,
    model: str,
    state_path: Path,
    metadata: Mapping[str, Any],
    common_request_kwargs: Mapping[str, Any],
    max_enqueued_tokens: int | None = None,
) -> EvaluatorExecutionResult:
    """Run evaluator tasks through the OpenAI Batch transport and write results."""
    import openai_batch

    validate_unique_custom_ids(tasks, evaluator_name=adapter.name)
    batch_requests = [
        openai_batch.build_chat_request(
            custom_id=task.custom_id,
            model=model,
            messages=task.messages,
            **_merged_request_kwargs(common_request_kwargs, task),
        )
        for task in tasks
    ]
    results = openai_batch.resume_or_submit(
        client,
        batch_requests,
        state_path,
        metadata=dict(metadata),
        max_enqueued_tokens=max_enqueued_tokens,
    )

    errors = 0
    succeeded = 0
    for task in tasks:
        record = _record_for_task(records, task)
        entry = results.get(task.custom_id)
        if entry is None:
            adapter.write_error(record, "missing_from_batch", None)
            errors += 1
            continue

        content = openai_batch.parse_chat_content(entry)
        if content is None:
            adapter.write_error(record, "batch_request_failed", None)
            errors += 1
            continue

        parsed = adapter.parse_response(content)
        if parsed is None:
            adapter.write_error(record, "parse_failed", content)
            errors += 1
            continue

        adapter.write_success(record, parsed)
        succeeded += 1

    return EvaluatorExecutionResult(
        submitted=len(tasks),
        succeeded=succeeded,
        errors=errors,
    )


def execute_sync_evaluator(
    *,
    records: list[dict[str, Any]],
    tasks: Sequence[EvaluatorTask],
    adapter: EvaluatorAdapter,
    client: Any,
    model: str,
    common_request_kwargs: Mapping[str, Any],
    max_retries: int = 5,
    retry_backoff_base: float = 2.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    cache_stats: Any | None = None,
) -> EvaluatorExecutionResult:
    """Run evaluator tasks through synchronous chat completions and write results."""
    from openai_batch import CacheStats

    if max_retries <= 0:
        raise ValueError(f"max_retries must be positive, got {max_retries}")
    stats = cache_stats if cache_stats is not None else CacheStats()

    errors = 0
    succeeded = 0
    for task in tasks:
        record = _record_for_task(records, task)
        kwargs = _merged_request_kwargs(common_request_kwargs, task)
        content: str | None = None
        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=cast(Any, list(task.messages)),
                    **kwargs,
                )
                stats.record(getattr(completion, "usage", None))
                raw_content = completion.choices[0].message.content
                if raw_content is not None:
                    content = str(raw_content).strip()
                    break
                print(f"  Empty content (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    sleep_fn(retry_backoff_base**attempt)
            except Exception as exc:
                print(f"  Judge API error (attempt {attempt + 1}): {exc}")
                if attempt < max_retries - 1:
                    sleep_fn(retry_backoff_base**attempt)

        if content is None:
            adapter.write_error(record, "api_failed", None)
            errors += 1
            continue

        parsed = adapter.parse_response(content)
        if parsed is None:
            adapter.write_error(record, "parse_failed", content)
            errors += 1
            continue

        adapter.write_success(record, parsed)
        succeeded += 1

    return EvaluatorExecutionResult(
        submitted=len(tasks),
        succeeded=succeeded,
        errors=errors,
        cache_stats=stats,
    )
