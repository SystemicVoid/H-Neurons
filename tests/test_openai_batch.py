"""Tests for scripts/openai_batch.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import openai_batch
from openai_batch import (
    CacheStats,
    DEFAULT_BATCH_QUEUE_SAFETY_MARGIN,
    DEFAULT_OPENAI_BATCH_USAGE_TIER,
    MAX_ENQUEUED_TOKENS_FALLBACK,
    OPENAI_BATCH_USAGE_TIER_ENV,
    _chunk_requests,
    _estimate_request_tokens,
    _normalize_model_limit_key,
    _resolve_max_enqueued_tokens,
    _resolve_openai_batch_usage_tier,
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


class TestMaxEnqueuedTokenResolution:
    def _make_request(self, custom_id: str, model: str = "gpt-4o") -> dict:
        return build_chat_request(
            custom_id,
            model,
            [{"role": "user", "content": "short"}],
        )

    def test_known_model_uses_tier4_limit_with_default_margin(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BATCH_MAX_ENQUEUED_TOKENS", raising=False)
        monkeypatch.delenv("OPENAI_BATCH_QUEUE_SAFETY_MARGIN", raising=False)
        monkeypatch.delenv("OPENAI_BATCH_USAGE_TIER", raising=False)

        resolution = _resolve_max_enqueued_tokens([self._make_request("r1", "gpt-4o")])

        assert resolution.source == "model"
        assert resolution.usage_tier == DEFAULT_OPENAI_BATCH_USAGE_TIER
        assert resolution.queue_limit == 200_000_000
        assert resolution.safety_margin == DEFAULT_BATCH_QUEUE_SAFETY_MARGIN
        assert resolution.value == int(200_000_000 * DEFAULT_BATCH_QUEUE_SAFETY_MARGIN)

    def test_usage_tier_env_can_select_tier3(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BATCH_MAX_ENQUEUED_TOKENS", raising=False)
        monkeypatch.delenv("OPENAI_BATCH_QUEUE_SAFETY_MARGIN", raising=False)
        monkeypatch.setenv("OPENAI_BATCH_USAGE_TIER", "3")

        resolution = _resolve_max_enqueued_tokens([self._make_request("r1", "gpt-4o")])

        assert resolution.source == "model"
        assert resolution.usage_tier == 3
        assert resolution.queue_limit == 50_000_000
        assert resolution.value == int(50_000_000 * DEFAULT_BATCH_QUEUE_SAFETY_MARGIN)

    def test_snapshot_alias_resolves_to_base_model_limit(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BATCH_MAX_ENQUEUED_TOKENS", raising=False)
        monkeypatch.delenv("OPENAI_BATCH_QUEUE_SAFETY_MARGIN", raising=False)
        monkeypatch.delenv("OPENAI_BATCH_USAGE_TIER", raising=False)

        resolution = _resolve_max_enqueued_tokens(
            [self._make_request("r1", "gpt-4o-2024-11-20")]
        )

        assert resolution.source == "model"
        assert resolution.queue_limit == 200_000_000
        assert resolution.models == ("gpt-4o-2024-11-20",)

    def test_explicit_override_beats_env_and_model_defaults(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BATCH_MAX_ENQUEUED_TOKENS", "999999")

        resolution = _resolve_max_enqueued_tokens(
            [self._make_request("r1", "gpt-4o")],
            max_enqueued_tokens=123_456,
        )

        assert resolution.source == "explicit"
        assert resolution.value == 123_456

    def test_env_override_beats_model_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BATCH_MAX_ENQUEUED_TOKENS", "654321")
        monkeypatch.delenv("OPENAI_BATCH_QUEUE_SAFETY_MARGIN", raising=False)

        resolution = _resolve_max_enqueued_tokens([self._make_request("r1", "gpt-4o")])

        assert resolution.source == "env"
        assert resolution.value == 654_321

    def test_unknown_model_falls_back_to_legacy_safe_cap(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BATCH_MAX_ENQUEUED_TOKENS", raising=False)
        monkeypatch.delenv("OPENAI_BATCH_QUEUE_SAFETY_MARGIN", raising=False)
        monkeypatch.delenv("OPENAI_BATCH_USAGE_TIER", raising=False)

        resolution = _resolve_max_enqueued_tokens(
            [self._make_request("r1", "custom-eval-model")]
        )

        assert resolution.source == "fallback"
        assert resolution.value == MAX_ENQUEUED_TOKENS_FALLBACK

    def test_invalid_env_override_raises(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BATCH_MAX_ENQUEUED_TOKENS", "oops")

        with pytest.raises(ValueError, match="OPENAI_BATCH_MAX_ENQUEUED_TOKENS"):
            _resolve_max_enqueued_tokens([self._make_request("r1", "gpt-4o")])

    def test_invalid_usage_tier_env_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BATCH_MAX_ENQUEUED_TOKENS", raising=False)
        monkeypatch.setenv("OPENAI_BATCH_USAGE_TIER", "2")

        with pytest.raises(ValueError, match=OPENAI_BATCH_USAGE_TIER_ENV):
            _resolve_openai_batch_usage_tier()

    def test_gpt4o_tier4_cap_keeps_old_17_chunk_case_to_one_chunk(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BATCH_MAX_ENQUEUED_TOKENS", raising=False)
        monkeypatch.delenv("OPENAI_BATCH_QUEUE_SAFETY_MARGIN", raising=False)
        monkeypatch.delenv("OPENAI_BATCH_USAGE_TIER", raising=False)

        reqs = [self._make_request(f"r{i}", "gpt-4o") for i in range(500)]
        estimates = [2568] * 499 + [2790]  # total = 1,284,222 tokens

        def fake_estimate(request: dict) -> int:
            idx = int(request["custom_id"][1:])
            return estimates[idx]

        monkeypatch.setattr(openai_batch, "_estimate_request_tokens", fake_estimate)

        resolution = _resolve_max_enqueued_tokens(reqs)
        chunks = _chunk_requests(reqs, max_tokens=resolution.value)

        assert resolution.value == int(200_000_000 * DEFAULT_BATCH_QUEUE_SAFETY_MARGIN)
        assert sum(fake_estimate(r) for r in reqs) == 1_284_222
        assert len(chunks) == 1


class TestNormalizeModelLimitKey:
    """Coverage for `_normalize_model_limit_key`, including dotted gpt-5 aliases."""

    def test_plain_gpt5_resolves_to_gpt5(self):
        assert _normalize_model_limit_key("gpt-5") == "gpt-5"

    def test_dotted_gpt5_minor_resolves_like_gpt5(self):
        assert _normalize_model_limit_key("gpt-5.5") == "gpt-5"
        assert _normalize_model_limit_key("gpt-5.1") == "gpt-5"
        assert _normalize_model_limit_key("gpt-5.4") == "gpt-5"

    def test_dotted_gpt5_pro_resolves_to_gpt5_family(self):
        # No distinct "pro" key in the local table, so pro variants should
        # land on the gpt-5 base family — same as ``gpt-5-pro`` does today.
        assert _normalize_model_limit_key("gpt-5.5-pro") == _normalize_model_limit_key(
            "gpt-5-pro"
        )
        assert _normalize_model_limit_key("gpt-5.5-pro") == "gpt-5"

    def test_dotted_gpt5_mini_resolves_to_gpt5_mini(self):
        assert _normalize_model_limit_key("gpt-5.5-mini") == "gpt-5-mini"
        assert _normalize_model_limit_key("gpt-5.1-mini") == "gpt-5-mini"
        assert _normalize_model_limit_key("gpt-5.4-mini") == "gpt-5-mini"

    def test_dotted_gpt5_with_snapshot_suffix(self):
        assert _normalize_model_limit_key("gpt-5.5-2026-04-01") == "gpt-5"
        assert _normalize_model_limit_key("gpt-5.5-mini-2026-04-01") == "gpt-5-mini"

    def test_case_insensitive(self):
        assert _normalize_model_limit_key("GPT-5.5") == "gpt-5"
        assert _normalize_model_limit_key("Gpt-5.5-Mini") == "gpt-5-mini"

    def test_gpt4o_unchanged(self):
        assert _normalize_model_limit_key("gpt-4o") == "gpt-4o"
        assert _normalize_model_limit_key("gpt-4o-mini") == "gpt-4o-mini"
        assert _normalize_model_limit_key("gpt-4o-2024-08-06") == "gpt-4o"

    def test_gpt41_unchanged(self):
        # gpt-4.1 is an explicit prefix — must still resolve and must not be
        # affected by the dotted-gpt5 collapse.
        assert _normalize_model_limit_key("gpt-4.1") == "gpt-4.1"
        assert _normalize_model_limit_key("gpt-4.1-2025-01-01") == "gpt-4.1"

    def test_o_series_unchanged(self):
        assert _normalize_model_limit_key("o3") == "o3"
        assert _normalize_model_limit_key("o4-mini") == "o4-mini"

    def test_unknown_model_returns_none(self):
        assert _normalize_model_limit_key("claude-3-opus") is None
        assert _normalize_model_limit_key("gemini-1.5-pro") is None

    def test_dotted_gpt5_bogus_suffix_consistent_with_undotted(self):
        # ``gpt-5.5-bogus`` should resolve identically to ``gpt-5-bogus`` —
        # we are not broadening recognition beyond what the existing prefix
        # match already accepts for the base family.
        assert _normalize_model_limit_key(
            "gpt-5.5-bogus"
        ) == _normalize_model_limit_key("gpt-5-bogus")

    def test_garbage_with_gpt5_substring_returns_none(self):
        # Tokens like ``gpt-5x`` or ``gpt-5.5x`` (no proper separator after
        # the version) must not slip through.
        assert _normalize_model_limit_key("gpt-5x") is None
        assert _normalize_model_limit_key("gpt-5.5x") is None
        assert _normalize_model_limit_key("gpt-55") is None
