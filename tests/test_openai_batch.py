"""Tests for scripts/openai_batch.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from openai_batch import (
    CacheStats,
    _chunk_requests,
    _estimate_request_tokens,
    build_chat_request,
    extract_batch_cache_stats,
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


class TestCacheStats:
    """Tests for prompt cache statistics collection."""

    def test_record_batch_entry_with_cache(self):
        stats = CacheStats()
        entry = {
            "custom_id": "r1",
            "response": {
                "status_code": 200,
                "body": {
                    "usage": {
                        "prompt_tokens": 2000,
                        "completion_tokens": 100,
                        "total_tokens": 2100,
                        "prompt_tokens_details": {"cached_tokens": 1500},
                    }
                },
            },
        }
        stats.record_batch_entry(entry)
        assert stats.requests == 1
        assert stats.total_prompt_tokens == 2000
        assert stats.cached_tokens == 1500
        assert stats.cache_rate == 1500 / 2000

    def test_record_batch_entry_no_cache(self):
        stats = CacheStats()
        entry = {
            "custom_id": "r1",
            "response": {
                "status_code": 200,
                "body": {
                    "usage": {
                        "prompt_tokens": 500,
                        "completion_tokens": 50,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    }
                },
            },
        }
        stats.record_batch_entry(entry)
        assert stats.cached_tokens == 0
        assert stats.cache_rate == 0.0

    def test_record_batch_entry_null_response(self):
        stats = CacheStats()
        stats.record_batch_entry({"custom_id": "r1", "response": None})
        assert stats.requests == 0

    def test_record_batch_entry_missing_details(self):
        stats = CacheStats()
        entry = {
            "custom_id": "r1",
            "response": {
                "status_code": 200,
                "body": {"usage": {"prompt_tokens": 1000, "completion_tokens": 50}},
            },
        }
        stats.record_batch_entry(entry)
        assert stats.requests == 1
        assert stats.total_prompt_tokens == 1000
        assert stats.cached_tokens == 0

    def test_extract_batch_cache_stats_multiple(self):
        results = {
            "r1": {
                "response": {
                    "status_code": 200,
                    "body": {
                        "usage": {
                            "prompt_tokens": 2000,
                            "prompt_tokens_details": {"cached_tokens": 1500},
                        }
                    },
                }
            },
            "r2": {
                "response": {
                    "status_code": 200,
                    "body": {
                        "usage": {
                            "prompt_tokens": 2000,
                            "prompt_tokens_details": {"cached_tokens": 1800},
                        }
                    },
                }
            },
            "r3": {"response": None},
        }
        stats = extract_batch_cache_stats(results)
        assert stats.requests == 2
        assert stats.total_prompt_tokens == 4000
        assert stats.cached_tokens == 3300
        assert abs(stats.cache_rate - 3300 / 4000) < 1e-9

    def test_summary_no_requests(self):
        stats = CacheStats()
        assert "no requests" in stats.summary()

    def test_summary_with_data(self):
        stats = CacheStats(total_prompt_tokens=4000, cached_tokens=3000, requests=2)
        s = stats.summary()
        assert "3,000" in s
        assert "4,000" in s
        assert "75.0%" in s

    def test_record_sync_usage(self):
        """Test recording from a sync completion usage object."""

        class MockDetails:
            cached_tokens = 800

        class MockUsage:
            prompt_tokens = 1200
            prompt_tokens_details = MockDetails()

        stats = CacheStats()
        stats.record(MockUsage())
        assert stats.requests == 1
        assert stats.total_prompt_tokens == 1200
        assert stats.cached_tokens == 800

    def test_record_sync_usage_no_details(self):
        class MockUsage:
            prompt_tokens = 500
            prompt_tokens_details = None

        stats = CacheStats()
        stats.record(MockUsage())
        assert stats.requests == 1
        assert stats.total_prompt_tokens == 500
        assert stats.cached_tokens == 0

    def test_record_none_usage(self):
        stats = CacheStats()
        stats.record(None)
        assert stats.requests == 0


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
