"""Tests for shared evaluator execution helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import openai_batch
from evaluator_execution import (
    EvaluatorAdapter,
    EvaluatorTask,
    execute_batch_evaluator,
    execute_sync_evaluator,
)


def _test_adapter() -> EvaluatorAdapter:
    def parse_response(raw: str) -> dict[str, str] | None:
        if raw == "OK":
            return {"value": raw}
        return None

    def write_success(record: dict[str, Any], parsed: Any) -> None:
        record["result"] = parsed

    def write_error(
        record: dict[str, Any], error_code: str, raw_response: str | None
    ) -> None:
        record["result"] = {"error": error_code, "raw_response": raw_response}

    return EvaluatorAdapter(
        name="unit",
        parse_response=parse_response,
        write_success=write_success,
        write_error=write_error,
    )


def _batch_entry(content: str) -> dict[str, Any]:
    return {
        "response": {
            "status_code": 200,
            "body": {"choices": [{"message": {"content": content}}]},
        }
    }


def test_batch_executor_writes_successes_and_failures(
    tmp_path: Path, monkeypatch
) -> None:
    records = [{}, {}, {}, {}]
    tasks = [
        EvaluatorTask("ok", 0, [{"role": "user", "content": "ok"}]),
        EvaluatorTask("parse", 1, [{"role": "user", "content": "parse"}]),
        EvaluatorTask("failed", 2, [{"role": "user", "content": "failed"}]),
        EvaluatorTask("missing", 3, [{"role": "user", "content": "missing"}]),
    ]
    state_path = tmp_path / ".batch_state.json"
    state_path.write_text(json.dumps({"batch_id": "existing"}), encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_resume_or_submit(
        client,
        batch_requests,
        state_path_arg,
        *,
        metadata,
        max_enqueued_tokens,
    ):
        captured["custom_ids"] = [request["custom_id"] for request in batch_requests]
        captured["state_path"] = state_path_arg
        captured["metadata"] = metadata
        captured["max_enqueued_tokens"] = max_enqueued_tokens
        assert state_path_arg.exists()
        return {
            "ok": _batch_entry("OK"),
            "parse": _batch_entry("NOT_OK"),
            "failed": {"response": {"status_code": 500, "body": {}}},
        }

    monkeypatch.setattr(openai_batch, "resume_or_submit", fake_resume_or_submit)

    result = execute_batch_evaluator(
        records=records,
        tasks=tasks,
        adapter=_test_adapter(),
        client=object(),
        model="gpt-test",
        state_path=state_path,
        metadata={"script": "unit"},
        common_request_kwargs={"temperature": 0.0},
        max_enqueued_tokens=123,
    )

    assert captured == {
        "custom_ids": ["ok", "parse", "failed", "missing"],
        "state_path": state_path,
        "metadata": {"script": "unit"},
        "max_enqueued_tokens": 123,
    }
    assert result.submitted == 4
    assert result.succeeded == 1
    assert result.errors == 3
    assert records[0]["result"] == {"value": "OK"}
    assert records[1]["result"] == {
        "error": "parse_failed",
        "raw_response": "NOT_OK",
    }
    assert records[2]["result"] == {
        "error": "batch_request_failed",
        "raw_response": None,
    }
    assert records[3]["result"] == {
        "error": "missing_from_batch",
        "raw_response": None,
    }


class _Completion:
    def __init__(self, content: str | None) -> None:
        message = type("Message", (), {"content": content})()
        self.choices = [type("Choice", (), {"message": message})()]
        self.usage = None


class _FakeCompletions:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)

    def create(self, **_kwargs: Any) -> Any:
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        completions = _FakeCompletions(responses)
        self.chat = type("Chat", (), {"completions": completions})()


def test_sync_executor_writes_success_parse_failure_and_api_failure() -> None:
    records = [{}, {}, {}]
    tasks = [
        EvaluatorTask("ok", 0, [{"role": "user", "content": "ok"}]),
        EvaluatorTask("parse", 1, [{"role": "user", "content": "parse"}]),
        EvaluatorTask("api", 2, [{"role": "user", "content": "api"}]),
    ]
    client = _FakeClient(
        [
            _Completion("OK"),
            _Completion("NOT_OK"),
            RuntimeError("network"),
        ]
    )

    result = execute_sync_evaluator(
        records=records,
        tasks=tasks,
        adapter=_test_adapter(),
        client=client,
        model="gpt-test",
        common_request_kwargs={"temperature": 0.0},
        max_retries=1,
        sleep_fn=lambda _seconds: None,
    )

    assert result.submitted == 3
    assert result.succeeded == 1
    assert result.errors == 2
    assert records[0]["result"] == {"value": "OK"}
    assert records[1]["result"] == {
        "error": "parse_failed",
        "raw_response": "NOT_OK",
    }
    assert records[2]["result"] == {
        "error": "api_failed",
        "raw_response": None,
    }
