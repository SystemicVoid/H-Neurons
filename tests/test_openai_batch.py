"""Tests for scripts/openai_batch.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from openai_batch import (
    _chunk_requests,
    _estimate_request_tokens,
    build_chat_request,
    parse_chat_content,
)


class TestParseChatContent:
    """Regression tests for batch result parsing, especially null-response entries."""

    def test_successful_response(self):
        entry = {
            "custom_id": "req-1",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [
                        {"message": {"role": "assistant", "content": "ACCEPTED"}}
                    ]
                },
            },
            "error": None,
        }
        assert parse_chat_content(entry) == "ACCEPTED"

    def test_null_response_returns_none(self):
        """Batch API error entries have 'response': null (e.g. expired requests)."""
        entry = {
            "custom_id": "req-2",
            "response": None,
            "error": {"code": "batch_expired", "message": "Request expired."},
        }
        assert parse_chat_content(entry) is None

    def test_missing_response_key_returns_none(self):
        entry = {"custom_id": "req-3", "error": {"code": "server_error"}}
        assert parse_chat_content(entry) is None

    def test_non_200_status_returns_none(self):
        entry = {
            "custom_id": "req-4",
            "response": {"status_code": 400, "body": {"error": {"message": "bad"}}},
            "error": None,
        }
        assert parse_chat_content(entry) is None


class TestNokCounting:
    """Regression: n_ok counting must not crash on response:null entries (P1 fix)."""

    def _count_ok(self, results: dict) -> int:
        return sum(
            1
            for e in results.values()
            if (e.get("response") or {}).get("status_code") == 200
        )

    def test_mixed_success_and_null_response(self):
        results = {
            "ok": {
                "response": {"status_code": 200, "body": {}},
                "error": None,
            },
            "expired": {
                "response": None,
                "error": {"code": "batch_expired"},
            },
            "missing": {
                "error": {"code": "server_error"},
            },
        }
        assert self._count_ok(results) == 1

    def test_all_null_responses(self):
        results = {
            "a": {"response": None, "error": {"code": "batch_expired"}},
            "b": {"response": None, "error": {"code": "batch_expired"}},
        }
        assert self._count_ok(results) == 0


class TestTokenEstimationAndChunking:
    """Tests for auto-chunking logic."""

    def _make_request(self, custom_id: str, content: str) -> dict:
        return build_chat_request(
            custom_id, "gpt-4o", [{"role": "user", "content": content}]
        )

    def test_estimate_tokens_basic(self):
        req = self._make_request("r1", "Hello world")
        est = _estimate_request_tokens(req)
        # 11 chars / 4 + 1 message * 4 overhead = 6
        assert est > 0

    def test_single_chunk_when_under_limit(self):
        reqs = [self._make_request(f"r{i}", "short") for i in range(3)]
        chunks = _chunk_requests(reqs, max_tokens=100_000)
        assert len(chunks) == 1
        assert len(chunks[0]) == 3

    def test_splits_when_over_limit(self):
        content = "x" * 400  # ~100 tokens each
        reqs = [self._make_request(f"r{i}", content) for i in range(10)]
        chunks = _chunk_requests(reqs, max_tokens=300)
        assert len(chunks) > 1
        assert sum(len(c) for c in chunks) == 10

    def test_single_large_request_not_dropped(self):
        content = "x" * 4000  # ~1000 tokens, exceeds limit alone
        reqs = [self._make_request("big", content)]
        chunks = _chunk_requests(reqs, max_tokens=100)
        assert len(chunks) == 1
        assert len(chunks[0]) == 1
