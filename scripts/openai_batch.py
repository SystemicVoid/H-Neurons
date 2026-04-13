"""OpenAI Batch API client for async chat completion requests.

Provides a drop-in alternative to synchronous chat.completions.create()
that uses the Batch API for 50% cost savings on non-time-critical evals.

Features:
- Crash-resumable: saves batch_id to a state file so interrupted polls
  can resume without resubmitting.
- Robust error handling: retries on transient network errors, surfaces
  per-request failures without losing successful results.
- Format-compatible: returns response bodies identical to synchronous API,
  so downstream parsing produces the same data fields.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

TERMINAL_STATUSES = frozenset({"completed", "failed", "expired", "cancelled"})
DEFAULT_POLL_INTERVAL = 30
DEFAULT_COMPLETION_WINDOW = "24h"
MAX_POLL_RETRIES = 10
MAX_UPLOAD_RETRIES = 5
UPLOAD_RETRY_BACKOFF = 2
CHARS_PER_TOKEN = 4
TOKENS_PER_MESSAGE_OVERHEAD = 4
DEFAULT_BATCH_QUEUE_SAFETY_MARGIN = 0.9
MAX_ENQUEUED_TOKENS_FALLBACK = 80_000
MAX_ENQUEUED_TOKENS_ENV = "OPENAI_BATCH_MAX_ENQUEUED_TOKENS"
BATCH_QUEUE_SAFETY_MARGIN_ENV = "OPENAI_BATCH_QUEUE_SAFETY_MARGIN"

# Tier-3 Batch queue limits verified against current OpenAI model pages on
# 2026-04-01. Keep this table local and update it when the docs change.
TIER3_BATCH_QUEUE_LIMITS = {
    "gpt-4o": 50_000_000,
    "gpt-4o-mini": 40_000_000,
    "gpt-4.1": 50_000_000,
    "gpt-5": 100_000_000,
    "gpt-5-mini": 40_000_000,
    "o3": 50_000_000,
    "o4-mini": 40_000_000,
}

MODEL_LIMIT_PREFIXES = (
    "gpt-5-mini",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4o",
    "gpt-5",
    "o4-mini",
    "o3",
)


# ---------------------------------------------------------------------------
# Prompt cache statistics
# ---------------------------------------------------------------------------


@dataclass
class CacheStats:
    """Accumulator for OpenAI prompt cache hit statistics."""

    total_prompt_tokens: int = field(default=0)
    cached_tokens: int = field(default=0)
    requests: int = field(default=0)

    def record(self, usage: Any) -> None:
        """Record usage from a synchronous completion response."""
        if usage is None:
            return
        self.requests += 1
        self.total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details:
            self.cached_tokens += getattr(details, "cached_tokens", 0) or 0

    def record_batch_entry(self, entry: dict[str, Any]) -> None:
        """Record usage from a batch result entry."""
        response = entry.get("response")
        if not isinstance(response, dict):
            return
        body = response.get("body")
        if not isinstance(body, dict):
            return
        status_code = response.get("status_code", 200)
        if status_code != 200:
            return
        usage = body.get("usage", {})
        if not isinstance(usage, dict):
            return
        self.requests += 1
        self.total_prompt_tokens += usage.get("prompt_tokens", 0)
        details = usage.get("prompt_tokens_details") or {}
        self.cached_tokens += details.get("cached_tokens", 0)

    @property
    def cache_rate(self) -> float:
        if self.total_prompt_tokens == 0:
            return 0.0
        return self.cached_tokens / self.total_prompt_tokens

    def summary(self) -> str:
        if self.requests == 0:
            return "  Cache stats: no requests recorded"
        return (
            f"  Cache stats: {self.cached_tokens:,}/{self.total_prompt_tokens:,} "
            f"prompt tokens cached ({self.cache_rate:.1%}) "
            f"across {self.requests} requests"
        )


def extract_batch_cache_stats(
    results: dict[str, dict[str, Any]],
) -> CacheStats:
    """Extract prompt cache statistics from batch results."""
    stats = CacheStats()
    for entry in results.values():
        stats.record_batch_entry(entry)
    return stats


# ---------------------------------------------------------------------------
# Request builder
# ---------------------------------------------------------------------------


def build_chat_request(
    custom_id: str,
    model: str,
    messages: Sequence[Mapping[str, object]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a single Batch API request line for /v1/chat/completions."""
    body: dict[str, Any] = {"model": model, "messages": list(messages), **kwargs}
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


# ---------------------------------------------------------------------------
# Token estimation & auto-chunking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaxEnqueuedTokensResolution:
    """Resolved chunking cap and how it was chosen."""

    value: int
    source: str
    models: tuple[str, ...] = ()
    queue_limit: int | None = None
    safety_margin: float | None = None

    def summary(self) -> str:
        if self.source == "explicit":
            return f"  Using explicit batch token cap {self.value:,}"
        if self.source == "env":
            return (
                f"  Using {MAX_ENQUEUED_TOKENS_ENV}={self.value:,} for batch token cap"
            )
        if self.source == "model":
            models = ", ".join(self.models) if self.models else "unknown"
            return (
                f"  Using model-aware batch token cap {self.value:,} "
                f"for {models} (Tier-3 queue {self.queue_limit:,} "
                f"x safety margin {self.safety_margin:.2f})"
            )
        return f"  Using fallback batch token cap {self.value:,} (unknown model)"


def _normalize_model_limit_key(model: str) -> str | None:
    """Map model aliases and snapshots onto the local queue-limit table."""
    normalized = model.strip().lower()
    for prefix in MODEL_LIMIT_PREFIXES:
        if normalized == prefix or normalized.startswith(f"{prefix}-"):
            return prefix
    return None


def _parse_max_enqueued_tokens_override(raw: str, source: str) -> int:
    """Parse and validate a max-enqueued-tokens override."""
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{source} must be an integer token count, got {raw!r}"
        ) from exc
    if value <= 0:
        raise ValueError(f"{source} must be > 0, got {value}")
    return value


def _resolve_batch_queue_safety_margin() -> float:
    """Resolve the safety margin used for model-aware queue limits."""
    raw = os.environ.get(BATCH_QUEUE_SAFETY_MARGIN_ENV)
    if raw is None:
        return DEFAULT_BATCH_QUEUE_SAFETY_MARGIN
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{BATCH_QUEUE_SAFETY_MARGIN_ENV} must be a float, got {raw!r}"
        ) from exc
    if not 0 < value <= 1:
        raise ValueError(
            f"{BATCH_QUEUE_SAFETY_MARGIN_ENV} must be in (0, 1], got {value}"
        )
    return value


def _resolve_max_enqueued_tokens(
    requests: Sequence[dict[str, Any]],
    max_enqueued_tokens: int | None = None,
) -> MaxEnqueuedTokensResolution:
    """Resolve the effective chunking cap for a request set."""
    if max_enqueued_tokens is not None:
        return MaxEnqueuedTokensResolution(
            value=_parse_max_enqueued_tokens_override(
                str(max_enqueued_tokens), "max_enqueued_tokens"
            ),
            source="explicit",
        )

    env_override = os.environ.get(MAX_ENQUEUED_TOKENS_ENV)
    if env_override is not None:
        return MaxEnqueuedTokensResolution(
            value=_parse_max_enqueued_tokens_override(
                env_override, MAX_ENQUEUED_TOKENS_ENV
            ),
            source="env",
        )

    models = sorted(
        {
            str(body["model"])
            for request in requests
            if isinstance((body := request.get("body")), dict) and "model" in body
        }
    )
    limit_keys = {
        limit_key
        for model in models
        if (limit_key := _normalize_model_limit_key(model)) is not None
    }
    if not limit_keys:
        return MaxEnqueuedTokensResolution(
            value=MAX_ENQUEUED_TOKENS_FALLBACK,
            source="fallback",
            models=tuple(models),
        )

    queue_limit = min(TIER3_BATCH_QUEUE_LIMITS[key] for key in limit_keys)
    safety_margin = _resolve_batch_queue_safety_margin()
    effective_limit = max(1, int(queue_limit * safety_margin))
    return MaxEnqueuedTokensResolution(
        value=effective_limit,
        source="model",
        models=tuple(models),
        queue_limit=queue_limit,
        safety_margin=safety_margin,
    )


def _estimate_request_tokens(request: dict[str, Any]) -> int:
    """Conservative estimate of input tokens for a batch request."""
    body = request.get("body", {})
    messages = body.get("messages", [])
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    return total_chars // CHARS_PER_TOKEN + len(messages) * TOKENS_PER_MESSAGE_OVERHEAD


def _chunk_requests(
    requests: list[dict[str, Any]], max_tokens: int
) -> list[list[dict[str, Any]]]:
    """Split requests into chunks that fit within the enqueued token limit."""
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    for req in requests:
        est = _estimate_request_tokens(req)
        if current and current_tokens + est > max_tokens:
            chunks.append(current)
            current = [req]
            current_tokens = est
        else:
            current.append(req)
            current_tokens += est
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_chat_content(result_entry: dict[str, Any]) -> str | None:
    """Extract assistant message content from a batch result entry.

    Returns None if the request errored or the response is malformed.
    The returned string is identical to what
    ``completion.choices[0].message.content`` would return from the
    synchronous API, ensuring no data-format difference between modes.
    """
    response = result_entry.get("response")
    if response is None:
        return None
    body = response.get("body") if isinstance(response, dict) else None
    if body is None:
        return None
    status_code = response.get("status_code", 0)
    if status_code != 200:
        return None
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


# ---------------------------------------------------------------------------
# State file helpers (crash-resumability)
# ---------------------------------------------------------------------------


def _write_state(state_path: Path, batch_id: str, input_file_id: str) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {"batch_id": batch_id, "input_file_id": input_file_id},
            indent=2,
        )
        + "\n"
    )


def _read_state(state_path: Path) -> dict[str, str] | None:
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _clear_state(state_path: Path) -> None:
    try:
        state_path.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Partial results file (crash-resumability across chunks)
# ---------------------------------------------------------------------------


def _load_partial_results(results_path: Path) -> dict[str, dict[str, Any]]:
    """Load previously completed chunk results."""
    if not results_path.exists():
        return {}
    results: dict[str, dict[str, Any]] = {}
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            results[entry["custom_id"]] = entry
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def _append_partial_results(
    results_path: Path, chunk_results: dict[str, dict[str, Any]]
) -> None:
    """Append chunk results to partial results file."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a") as f:
        for entry in chunk_results.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Upload with retry
# ---------------------------------------------------------------------------


def _upload_batch_file(
    client: OpenAI,
    input_path: Path,
) -> str:
    """Upload a JSONL file to OpenAI with retries on transient errors."""
    for attempt in range(MAX_UPLOAD_RETRIES):
        try:
            with open(input_path, "rb") as f:
                file_obj = client.files.create(file=f, purpose="batch")
            return file_obj.id
        except (APIConnectionError, APITimeoutError, RateLimitError, OSError) as exc:
            wait = UPLOAD_RETRY_BACKOFF ** (attempt + 1)
            print(
                f"  Upload error (attempt {attempt + 1}/{MAX_UPLOAD_RETRIES}): "
                f"{exc} — retrying in {wait}s",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise RuntimeError(
        f"Failed to upload batch file after {MAX_UPLOAD_RETRIES} attempts"
    )


# ---------------------------------------------------------------------------
# Poll with retry
# ---------------------------------------------------------------------------


def _poll_batch(
    client: OpenAI,
    batch_id: str,
    poll_interval: int,
) -> Any:
    """Poll a batch until it reaches a terminal status.

    Retries on transient network errors without counting them as permanent
    failures, up to MAX_POLL_RETRIES consecutive transient errors.
    """
    consecutive_errors = 0

    while True:
        try:
            batch = client.batches.retrieve(batch_id)
            consecutive_errors = 0
        except (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            OSError,
        ) as exc:
            consecutive_errors += 1
            if consecutive_errors > MAX_POLL_RETRIES:
                raise RuntimeError(
                    f"Lost contact with batch {batch_id} after "
                    f"{MAX_POLL_RETRIES} consecutive poll failures"
                ) from exc
            wait = min(poll_interval * 2, 120)
            print(
                f"  Poll error ({consecutive_errors}/{MAX_POLL_RETRIES}): "
                f"{exc} — retrying in {wait}s",
                file=sys.stderr,
            )
            time.sleep(wait)
            continue

        status = batch.status
        counts = batch.request_counts
        completed = counts.completed if counts else 0
        total = counts.total if counts else 0
        failed = counts.failed if counts else 0
        print(
            f"  Batch {batch_id}: {status} ({completed}/{total} done, {failed} failed)"
        )

        if status in TERMINAL_STATUSES:
            return batch

        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Download results
# ---------------------------------------------------------------------------


def _download_results(
    client: OpenAI,
    batch: Any,
) -> dict[str, dict[str, Any]]:
    """Download output + error files and merge into {custom_id: entry}."""
    results: dict[str, dict[str, Any]] = {}

    if batch.output_file_id:
        for attempt in range(MAX_UPLOAD_RETRIES):
            try:
                output = client.files.content(batch.output_file_id)
                break
            except (
                APIConnectionError,
                APITimeoutError,
                RateLimitError,
            ) as exc:
                if attempt == MAX_UPLOAD_RETRIES - 1:
                    raise RuntimeError(
                        f"Failed to download batch output after "
                        f"{MAX_UPLOAD_RETRIES} attempts"
                    ) from exc
                wait = UPLOAD_RETRY_BACKOFF ** (attempt + 1)
                print(
                    f"  Download error (attempt {attempt + 1}): {exc} "
                    f"— retrying in {wait}s",
                    file=sys.stderr,
                )
                time.sleep(wait)

        for line in output.text.strip().split("\n"):
            if not line.strip():
                continue
            entry = json.loads(line)
            results[entry["custom_id"]] = entry

    if batch.error_file_id:
        try:
            error_content = client.files.content(batch.error_file_id)
            for line in error_content.text.strip().split("\n"):
                if not line.strip():
                    continue
                entry = json.loads(line)
                cid = entry.get("custom_id")
                if cid and cid not in results:
                    results[cid] = entry
        except Exception as exc:
            print(
                f"  Warning: failed to download error file: {exc}",
                file=sys.stderr,
            )

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def submit_and_poll(
    client: OpenAI,
    requests: list[dict[str, Any]],
    *,
    metadata: dict[str, str] | None = None,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    state_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Submit a batch of chat completion requests and wait for results.

    Returns a dict mapping custom_id to the raw result entry. Use
    ``parse_chat_content(entry)`` to extract the assistant message.

    If ``state_path`` is provided, the batch_id is persisted so that a
    crashed/interrupted process can call ``resume_or_submit`` to pick up
    where it left off without resubmitting.
    """
    if not requests:
        return {}

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, dir="/tmp"
    ) as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
        input_path = Path(f.name)

    try:
        file_id = _upload_batch_file(client, input_path)

        batch = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window=DEFAULT_COMPLETION_WINDOW,
            metadata=metadata or {},
        )
        print(f"  Batch {batch.id} submitted ({len(requests)} requests)")

        if state_path:
            _write_state(state_path, batch.id, file_id)

        batch = _poll_batch(client, batch.id, poll_interval)

        if batch.status != "completed":
            counts = batch.request_counts
            completed = counts.completed if counts else 0
            failed = counts.failed if counts else 0
            msg = (
                f"Batch {batch.id} ended with status '{batch.status}' "
                f"({completed} completed, {failed} failed)"
            )
            if batch.status == "expired" and completed > 0:
                print(f"  Warning: {msg} — retrieving partial results")
            else:
                raise RuntimeError(msg)

        results = _download_results(client, batch)
    finally:
        input_path.unlink(missing_ok=True)

    if state_path:
        _clear_state(state_path)

    n_ok = sum(
        1
        for e in results.values()
        if (e.get("response") or {}).get("status_code") == 200
    )
    n_err = len(results) - n_ok
    print(f"  Batch complete: {n_ok} succeeded, {n_err} failed")
    print(extract_batch_cache_stats(results).summary())

    return results


def resume_or_submit(
    client: OpenAI,
    requests: list[dict[str, Any]],
    state_path: Path,
    *,
    metadata: dict[str, str] | None = None,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    max_enqueued_tokens: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Resume an existing batch if state file exists, otherwise submit new.

    Auto-chunks large request sets that exceed the enqueued token limit.
    Crash-resumable: completed chunk results are persisted to a partial
    results file so interrupted runs continue without resubmitting.
    """
    results_path = state_path.with_suffix(".results.jsonl")

    # 1. Load results from previously completed chunks
    all_results = _load_partial_results(results_path)

    # 2. Try to resume in-flight batch (current chunk)
    state = _read_state(state_path)
    if state and "batch_id" in state:
        batch_id = state["batch_id"]
        print(f"  Resuming batch {batch_id} from state file")
        try:
            batch = _poll_batch(client, batch_id, poll_interval)
            if batch.status == "completed" or (
                batch.status == "expired"
                and batch.request_counts
                and batch.request_counts.completed > 0
            ):
                chunk_results = _download_results(client, batch)
                all_results.update(chunk_results)
                _append_partial_results(results_path, chunk_results)
                _clear_state(state_path)
            elif batch.status in TERMINAL_STATUSES:
                print(
                    f"  Resumed batch {batch_id} has status '{batch.status}' "
                    f"— resubmitting chunk"
                )
                _clear_state(state_path)
        except (AuthenticationError, PermissionDeniedError):
            # Auth/permission errors are NOT transient — resubmitting would
            # just fail again (or worse, succeed and create a duplicate batch
            # once the scope recovers).  Fail hard so the operator can fix
            # the API key / project permissions and re-run safely.
            raise
        except Exception as exc:
            print(
                f"  Warning: resume failed ({exc}) — resubmitting",
                file=sys.stderr,
            )
            _clear_state(state_path)

    # 3. Filter out already-completed requests
    completed_ids = set(all_results.keys())
    remaining = [r for r in requests if r["custom_id"] not in completed_ids]

    if not remaining:
        try:
            results_path.unlink(missing_ok=True)
        except OSError:
            pass
        return all_results

    resolution = _resolve_max_enqueued_tokens(remaining, max_enqueued_tokens)
    print(resolution.summary())
    effective_max_enqueued_tokens = resolution.value

    # 4. Chunk remaining requests if needed
    total_est = sum(_estimate_request_tokens(r) for r in remaining)
    if total_est > effective_max_enqueued_tokens:
        chunks = _chunk_requests(remaining, effective_max_enqueued_tokens)
        print(
            f"  Auto-chunking: {len(remaining)} requests into {len(chunks)} "
            f"batches (~{total_est} est. tokens, "
            f"limit {effective_max_enqueued_tokens})"
        )
    else:
        chunks = [remaining]

    # 5. Submit each chunk
    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            print(f"\n  Chunk {i + 1}/{len(chunks)} ({len(chunk)} requests)...")
        chunk_results = submit_and_poll(
            client,
            chunk,
            metadata=metadata,
            poll_interval=poll_interval,
            state_path=state_path,
        )
        all_results.update(chunk_results)
        if len(chunks) > 1:
            _append_partial_results(results_path, chunk_results)

    # 6. Clean up partial results file
    try:
        results_path.unlink(missing_ok=True)
    except OSError:
        pass

    if len(chunks) > 1:
        n_ok = sum(
            1
            for e in all_results.values()
            if (e.get("response") or {}).get("status_code") == 200
        )
        n_err = len(all_results) - n_ok
        print(f"  All chunks complete: {n_ok} succeeded, {n_err} failed")
        print(extract_batch_cache_stats(all_results).summary())

    return all_results
