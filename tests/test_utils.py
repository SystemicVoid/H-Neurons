"""Tests for scripts/utils.py -- shared evaluation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# scripts/ uses flat sibling imports; add it to sys.path for test discovery.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_intervention import (
    HNeuronScaler,
    build_alpha_throughput_payload,
    build_alpha_throughput_summary,
    build_sample_throughput_payload,
)
from utils import extract_mc_answer, normalize_answer


# ---------------------------------------------------------------------------
# normalize_answer
# ---------------------------------------------------------------------------


class TestNormalizeAnswer:
    def test_lowercases(self):
        assert normalize_answer("PARIS") == "paris"

    def test_strips_articles(self):
        assert normalize_answer("The Eiffel Tower") == "eiffel tower"

    def test_strips_punctuation(self):
        assert normalize_answer("hello, world!") == "hello world"

    def test_collapses_whitespace(self):
        assert normalize_answer("  lots   of   spaces  ") == "lots of spaces"

    def test_underscores_to_spaces(self):
        assert normalize_answer("new_york_city") == "new york city"

    def test_curly_quotes_stripped(self):
        assert normalize_answer("\u2018quoted\u2019") == "quoted"

    def test_empty_string(self):
        assert normalize_answer("") == ""

    def test_none_returns_empty(self):
        assert normalize_answer(None) == ""

    def test_numeric_passthrough(self):
        assert normalize_answer("42") == "42"

    def test_combined_normalization(self):
        assert normalize_answer("The  'quick_brown' Fox!") == "quick brown fox"


# ---------------------------------------------------------------------------
# extract_mc_answer
# ---------------------------------------------------------------------------

VALID = ["A", "B", "C", "D"]


class TestExtractMcAnswer:
    def test_starts_with_letter(self):
        assert extract_mc_answer("B) some explanation", VALID) == "B"

    def test_answer_is_pattern(self):
        assert extract_mc_answer("The answer is C", VALID) == "C"

    def test_parenthesized(self):
        assert extract_mc_answer("(A)", VALID) == "A"

    def test_bold_markdown(self):
        assert extract_mc_answer("I think **D** is correct", VALID) == "D"

    def test_dot_pattern(self):
        assert extract_mc_answer("A. The first option", VALID) == "A"

    def test_standalone_letter_fallback(self):
        assert extract_mc_answer("Hmm, probably B here", VALID) == "B"

    def test_no_match_returns_none(self):
        assert extract_mc_answer("I have no idea", VALID) is None

    def test_empty_response(self):
        assert extract_mc_answer("", VALID) is None

    def test_invalid_letter_ignored(self):
        assert extract_mc_answer("E is my choice", VALID) is None

    def test_lowercase_input_still_matches(self):
        assert extract_mc_answer("the answer is b", VALID) == "B"

    def test_correct_is_pattern(self):
        assert extract_mc_answer("The correct answer is A", VALID) == "A"


class DummyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.down_proj.weight.copy_(torch.eye(4))


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([DummyBlock()])

    def forward(self, x):
        return self.layers[0].down_proj(x)


class TestHNeuronScalerSampleStats:
    def test_tracks_hook_time_and_calls_for_noop_and_active_alpha(self):
        model = DummyModel()
        scaler = HNeuronScaler(model, {0: [1, 3]}, device=torch.device("cpu"))
        x = torch.ones(1, 2, 4)

        scaler.alpha = 1.0
        scaler.reset_sample_stats()
        y_noop = model(x.clone())
        noop_stats = scaler.consume_sample_stats()

        assert torch.allclose(y_noop, x)
        assert noop_stats["hook_calls"] == 1
        assert noop_stats["hook_s"] >= 0.0

        scaler.alpha = 2.0
        scaler.reset_sample_stats()
        y_scaled = model(x.clone())
        active_stats = scaler.consume_sample_stats()

        assert y_scaled.tolist() == [[[1.0, 2.0, 1.0, 2.0], [1.0, 2.0, 1.0, 2.0]]]
        assert active_stats["hook_calls"] == 1
        assert active_stats["hook_s"] >= 0.0

        reset_stats = scaler.consume_sample_stats()
        assert reset_stats == {"hook_s": 0.0, "hook_calls": 0}

        scaler.remove()


class TestThroughputHelpers:
    def test_build_alpha_throughput_summary_uses_instrumented_rows_only(self):
        records = [
            {
                "id": "s1",
                "timings": {
                    "wall_start_ts": 10.0,
                    "wall_end_ts": 11.0,
                    "wall_total_s": 1.0,
                    "generate_s": 0.6,
                    "prompt_tokens": 5,
                    "generated_tokens": 10,
                    "hook_s": 0.1,
                },
            },
            {"id": "legacy-row"},
            {
                "id": "s2",
                "timings": {
                    "wall_start_ts": 13.0,
                    "wall_end_ts": 15.0,
                    "wall_total_s": 2.0,
                    "generate_s": 1.4,
                    "prompt_tokens": 7,
                    "generated_tokens": 20,
                    "hook_s": 0.3,
                },
            },
        ]

        summary = build_alpha_throughput_summary(records)

        assert summary["samples_completed"] == 2
        assert summary["record_count_total"] == 3
        assert summary["instrumented_row_count"] == 2
        assert summary["instrumented_coverage"] == pytest.approx(0.6667, abs=1e-4)
        assert summary["generated_tokens_total"] == 30
        assert summary["prompt_tokens_total"] == 12
        assert summary["wall_total_s"] == pytest.approx(5.0, abs=1e-4)
        assert summary["wall_measured_s"] == pytest.approx(3.0, abs=1e-4)
        assert summary["generate_total_s"] == pytest.approx(2.0, abs=1e-4)
        assert summary["samples_per_s_wall"] == pytest.approx(0.4, abs=1e-4)
        assert summary["tokens_per_s_wall"] == pytest.approx(6.0, abs=1e-4)
        assert summary["tokens_per_s_generate"] == pytest.approx(15.0, abs=1e-4)
        assert summary["inter_sample_gap_count"] == 1
        assert summary["inter_sample_gap_mean_s"] == pytest.approx(2.0, abs=1e-6)
        assert summary["inter_sample_gap_p50_s"] == pytest.approx(2.0, abs=1e-6)
        assert summary["inter_sample_gap_p95_s"] == pytest.approx(2.0, abs=1e-6)
        assert summary["hook_total_s"] == pytest.approx(0.4, abs=1e-6)
        assert summary["hook_mean_s"] == pytest.approx(0.2, abs=1e-6)
        assert summary["hook_frac_of_generate"] == pytest.approx(0.2, abs=1e-6)

    def test_build_alpha_throughput_summary_omits_hook_aggregates_without_stats(self):
        records = [
            {
                "id": "sae-1",
                "timings": {
                    "wall_start_ts": 10.0,
                    "wall_end_ts": 11.0,
                    "wall_total_s": 1.0,
                    "generate_s": 0.6,
                    "prompt_tokens": 5,
                    "generated_tokens": 10,
                },
            },
            {
                "id": "sae-2",
                "timings": {
                    "wall_start_ts": 12.0,
                    "wall_end_ts": 14.0,
                    "wall_total_s": 2.0,
                    "generate_s": 1.4,
                    "prompt_tokens": 7,
                    "generated_tokens": 20,
                },
            },
        ]

        summary = build_alpha_throughput_summary(records)

        assert "hook_total_s" not in summary
        assert "hook_mean_s" not in summary
        assert "hook_frac_of_generate" not in summary

    def test_build_alpha_throughput_summary_ignores_cross_session_downtime(self):
        records = [
            {
                "id": "run-1-a",
                "timings": {
                    "wall_start_ts": 10.0,
                    "wall_end_ts": 11.0,
                    "wall_total_s": 1.0,
                    "generate_s": 0.5,
                    "prompt_tokens": 4,
                    "generated_tokens": 8,
                    "throughput_session_id": "session-a",
                },
            },
            {
                "id": "run-1-b",
                "timings": {
                    "wall_start_ts": 13.0,
                    "wall_end_ts": 14.0,
                    "wall_total_s": 1.0,
                    "generate_s": 0.5,
                    "prompt_tokens": 4,
                    "generated_tokens": 8,
                    "throughput_session_id": "session-a",
                },
            },
            {
                "id": "run-2-a",
                "timings": {
                    "wall_start_ts": 100.0,
                    "wall_end_ts": 101.0,
                    "wall_total_s": 1.0,
                    "generate_s": 0.5,
                    "prompt_tokens": 4,
                    "generated_tokens": 8,
                    "throughput_session_id": "session-b",
                },
            },
            {
                "id": "run-2-b",
                "timings": {
                    "wall_start_ts": 103.0,
                    "wall_end_ts": 104.0,
                    "wall_total_s": 1.0,
                    "generate_s": 0.5,
                    "prompt_tokens": 4,
                    "generated_tokens": 8,
                    "throughput_session_id": "session-b",
                },
            },
        ]

        summary = build_alpha_throughput_summary(records)

        assert summary["wall_measured_s"] == pytest.approx(4.0, abs=1e-4)
        assert summary["wall_total_s"] == pytest.approx(8.0, abs=1e-4)
        assert summary["samples_per_s_wall"] == pytest.approx(0.5, abs=1e-4)
        assert summary["tokens_per_s_wall"] == pytest.approx(4.0, abs=1e-4)
        assert summary["inter_sample_gap_count"] == 2
        assert summary["inter_sample_gap_mean_s"] == pytest.approx(2.0, abs=1e-6)
        assert summary["inter_sample_gap_p50_s"] == pytest.approx(2.0, abs=1e-6)
        assert summary["inter_sample_gap_p95_s"] == pytest.approx(2.0, abs=1e-6)

    def test_build_alpha_throughput_summary_uses_caller_wall_total_override(self):
        records = [
            {
                "id": "s1",
                "timings": {
                    "wall_start_ts": 10.0,
                    "wall_end_ts": 11.0,
                    "wall_total_s": 1.0,
                    "generate_s": 0.6,
                    "prompt_tokens": 5,
                    "generated_tokens": 10,
                },
            },
            {
                "id": "s2",
                "timings": {
                    "wall_start_ts": 13.0,
                    "wall_end_ts": 14.0,
                    "wall_total_s": 1.0,
                    "generate_s": 0.6,
                    "prompt_tokens": 5,
                    "generated_tokens": 10,
                },
            },
        ]

        summary = build_alpha_throughput_summary(records, alpha_wall_total_s=10.0)

        assert summary["wall_total_s"] == pytest.approx(10.0, abs=1e-4)
        assert summary["samples_per_s_wall"] == pytest.approx(0.2, abs=1e-4)
        assert summary["tokens_per_s_wall"] == pytest.approx(2.0, abs=1e-4)

    def test_build_sample_throughput_payload(self):
        payload = build_sample_throughput_payload(
            benchmark="jailbreak",
            alpha=1.5,
            alpha_idx=2,
            sample_idx=7,
            timings={
                "generate_s": 4.0,
                "wall_total_s": 5.0,
                "generated_tokens": 20,
                "prompt_tokens": 8,
                "hook_s": 0.4,
                "hook_calls": 23,
                "hook_frac_of_generate": 0.1,
                "hit_token_cap": True,
            },
        )

        assert payload == {
            "throughput/sample_idx": 7,
            "throughput/sample/benchmark": "jailbreak",
            "throughput/sample/alpha": 1.5,
            "throughput/sample/alpha_idx": 2,
            "throughput/sample/generate_s": 4.0,
            "throughput/sample/wall_total_s": 5.0,
            "throughput/sample/generated_tokens": 20,
            "throughput/sample/prompt_tokens": 8,
            "throughput/sample/tokens_per_s_generate": 5.0,
            "throughput/sample/tokens_per_s_wall": 4.0,
            "throughput/sample/hook_s": 0.4,
            "throughput/sample/hook_calls": 23,
            "throughput/sample/hook_frac_of_generate": 0.1,
            "throughput/sample/hit_token_cap": 1,
        }

    def test_build_sample_throughput_payload_omits_hook_metrics_without_stats(self):
        payload = build_sample_throughput_payload(
            benchmark="jailbreak",
            alpha=1.5,
            alpha_idx=2,
            sample_idx=7,
            timings={
                "generate_s": 4.0,
                "wall_total_s": 5.0,
                "generated_tokens": 20,
                "prompt_tokens": 8,
                "hit_token_cap": False,
            },
        )

        assert "throughput/sample/hook_s" not in payload
        assert "throughput/sample/hook_calls" not in payload
        assert "throughput/sample/hook_frac_of_generate" not in payload

    def test_build_alpha_throughput_payload_omits_hook_metrics_without_stats(self):
        payload = build_alpha_throughput_payload(
            alpha=3.0,
            alpha_idx=4,
            throughput_summary={
                "samples_completed": 12,
                "wall_total_s": 30.5,
                "samples_per_s_wall": 0.3934,
                "tokens_per_s_wall": 18.2,
                "tokens_per_s_generate": 19.7,
                "mean_generated_tokens": 555.0,
                "inter_sample_gap_mean_s": 0.021,
                "instrumented_coverage": 1.0,
            },
        )

        assert "throughput/alpha/hook_frac_of_generate" not in payload
