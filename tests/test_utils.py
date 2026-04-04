"""Tests for scripts/utils.py -- shared evaluation helpers."""

from __future__ import annotations

import json
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# scripts/ uses flat sibling imports; add it to sys.path for test discovery.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

evaluate_intervention = importlib.import_module("evaluate_intervention")
extract_direction = importlib.import_module("extract_direction")
run_intervention = importlib.import_module("run_intervention")
utils = importlib.import_module("utils")

evaluate_direction_sanity_gate = extract_direction.evaluate_direction_sanity_gate
HNeuronScaler = run_intervention.HNeuronScaler
aggregate_results = run_intervention.aggregate_results
build_direction_output_suffix = run_intervention.build_direction_output_suffix
build_iti_output_suffix = run_intervention.build_iti_output_suffix
build_alpha_throughput_payload = run_intervention.build_alpha_throughput_payload
build_alpha_throughput_summary = run_intervention.build_alpha_throughput_summary
build_sample_throughput_payload = run_intervention.build_sample_throughput_payload
resolve_output_dir = run_intervention.resolve_output_dir
load_truthfulqa_mc = run_intervention.load_truthfulqa_mc
load_triviaqa_bridge = run_intervention.load_triviaqa_bridge
run_truthfulqa_mc = run_intervention.run_truthfulqa_mc
triviaqa_bridge_attempted = run_intervention.triviaqa_bridge_attempted
run_faitheval_mc_logprob = run_intervention.run_faitheval_mc_logprob
extract_mc_answer = utils.extract_mc_answer
format_alpha_label = utils.format_alpha_label
normalize_answer = utils.normalize_answer


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


class TestFormatAlphaLabel:
    def test_keeps_one_decimal_for_canonical_grid(self):
        assert format_alpha_label(0.0) == "0.0"
        assert format_alpha_label(1.5) == "1.5"
        assert format_alpha_label(3.0) == "3.0"

    def test_preserves_micro_beta_precision_without_collision(self):
        assert format_alpha_label(0.02) == "0.02"
        assert format_alpha_label(0.125) == "0.125"
        assert format_alpha_label(0.0) != format_alpha_label(0.02)


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
        block = self.layers[0]
        assert isinstance(block, DummyBlock)
        return block.down_proj(x)


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


class TestDirectionOutputDir:
    def test_direction_output_suffix_changes_when_config_changes(self):
        base = build_direction_output_suffix(
            "data/contrastive/refusal/directions/refusal_directions.pt",
            "ablate",
            None,
        )
        different_mode = build_direction_output_suffix(
            "data/contrastive/refusal/directions/refusal_directions.pt",
            "add",
            None,
        )
        different_layers = build_direction_output_suffix(
            "data/contrastive/refusal/directions/refusal_directions.pt",
            "ablate",
            "20,21,22",
        )
        different_path = build_direction_output_suffix(
            "data/contrastive/refusal/directions/alt_refusal_directions.pt",
            "ablate",
            None,
        )

        assert len({base, different_mode, different_layers, different_path}) == 4

    def test_resolve_output_dir_uses_config_specific_direction_default(self):
        args = SimpleNamespace(
            output_dir=None,
            intervention_mode="direction",
            benchmark="jailbreak",
            direction_path="data/contrastive/refusal/directions/refusal_directions.pt",
            direction_mode="ablate",
            direction_layers="20,21,22",
        )

        output_dir = resolve_output_dir(args)

        assert output_dir.startswith("data/gemma3_4b/intervention/jailbreak_")
        assert output_dir.endswith("/experiment")
        assert "direction_ablate_layers-20-21-22" in output_dir

    def test_resolve_output_dir_requires_direction_path_for_default(self):
        args = SimpleNamespace(
            output_dir=None,
            intervention_mode="direction",
            benchmark="jailbreak",
            direction_path=None,
            direction_mode="ablate",
            direction_layers=None,
        )

        with pytest.raises(ValueError, match="requires --direction_path"):
            resolve_output_dir(args)

    def test_iti_output_suffix_changes_when_config_changes(self):
        base = build_iti_output_suffix(
            "data/contrastive/truthfulness/iti_triviaqa/iti_heads.pt",
            "triviaqa_transfer",
            16,
            "ranked",
            42,
        )
        different_family = build_iti_output_suffix(
            "data/contrastive/truthfulness/iti_triviaqa/iti_heads.pt",
            "context_grounded",
            16,
            "ranked",
            42,
        )
        different_k = build_iti_output_suffix(
            "data/contrastive/truthfulness/iti_triviaqa/iti_heads.pt",
            "triviaqa_transfer",
            32,
            "ranked",
            42,
        )
        different_strategy = build_iti_output_suffix(
            "data/contrastive/truthfulness/iti_triviaqa/iti_heads.pt",
            "triviaqa_transfer",
            16,
            "random",
            42,
        )
        different_seed = build_iti_output_suffix(
            "data/contrastive/truthfulness/iti_triviaqa/iti_heads.pt",
            "triviaqa_transfer",
            16,
            "ranked",
            7,
        )
        different_path = build_iti_output_suffix(
            "data/contrastive/truthfulness/iti_context/iti_heads.pt",
            "triviaqa_transfer",
            16,
            "ranked",
            42,
        )
        different_scope = build_iti_output_suffix(
            "data/contrastive/truthfulness/iti_triviaqa/iti_heads.pt",
            "triviaqa_transfer",
            16,
            "ranked",
            42,
            iti_decode_scope="first_3_tokens",
        )

        assert (
            len(
                {
                    base,
                    different_family,
                    different_k,
                    different_strategy,
                    different_seed,
                    different_path,
                    different_scope,
                }
            )
            == 7
        )

    def test_resolve_output_dir_uses_config_specific_iti_default(self):
        args = SimpleNamespace(
            output_dir=None,
            intervention_mode="iti_head",
            benchmark="faitheval",
            iti_head_path="data/contrastive/truthfulness/iti_triviaqa/iti_heads.pt",
            iti_family="triviaqa_transfer",
            iti_k=16,
            iti_selection_strategy="ranked",
            iti_random_seed=42,
            iti_direction_mode="artifact",
            iti_direction_random_seed=None,
            iti_decode_scope="full_decode",
            truthfulqa_variant="mc1",
        )

        output_dir = resolve_output_dir(args)

        assert output_dir.startswith("data/gemma3_4b/intervention/faitheval_")
        assert output_dir.endswith("/experiment")
        assert "iti-head_triviaqa-transfer_k-16_ranked_seed-42" in output_dir
        assert "scope-full-decode" in output_dir

    def test_resolve_output_dir_requires_iti_path_for_default(self):
        args = SimpleNamespace(
            output_dir=None,
            intervention_mode="iti_head",
            benchmark="faitheval",
            iti_head_path=None,
            iti_family="triviaqa_transfer",
            iti_k=16,
            iti_selection_strategy="ranked",
            iti_random_seed=42,
            iti_direction_mode="artifact",
            iti_direction_random_seed=None,
            iti_decode_scope="full_decode",
            truthfulqa_variant="mc1",
        )

        with pytest.raises(ValueError, match="requires --iti_head_path"):
            resolve_output_dir(args)

    @pytest.mark.parametrize(
        ("intervention_mode", "expected_fragment"),
        [
            ("neuron", "truthfulqa_mc_mc1/experiment"),
            ("sae", "truthfulqa_mc_mc1_sae/experiment"),
            ("direction", "truthfulqa_mc_mc1_direction_ablate"),
        ],
    )
    def test_resolve_output_dir_separates_truthfulqa_variants_for_default_dirs(
        self, intervention_mode, expected_fragment
    ):
        common_args = dict(
            output_dir=None,
            intervention_mode=intervention_mode,
            benchmark="truthfulqa_mc",
            truthfulqa_variant="mc1",
            iti_head_path="data/contrastive/truthfulness/iti_triviaqa/iti_heads.pt",
            iti_family="triviaqa_transfer",
            iti_k=16,
            iti_selection_strategy="ranked",
            iti_random_seed=42,
            iti_direction_mode="artifact",
            iti_direction_random_seed=None,
            iti_decode_scope="full_decode",
            direction_path="data/contrastive/refusal/directions/refusal_directions.pt",
            direction_mode="ablate",
            direction_layers=None,
        )

        mc1_dir = resolve_output_dir(SimpleNamespace(**common_args))
        mc2_dir = resolve_output_dir(
            SimpleNamespace(**{**common_args, "truthfulqa_variant": "mc2"})
        )

        assert mc1_dir != mc2_dir
        assert expected_fragment in mc1_dir
        assert "truthfulqa_mc_mc2" in mc2_dir


class TestTruthfulqaMetrics:
    def test_run_truthfulqa_mc_records_observed_mc1_accuracy(
        self, tmp_path, monkeypatch
    ):
        output_dir = tmp_path / "truthfulqa"
        output_dir.mkdir()

        scores_by_choice = {
            " False answer": {
                "log_likelihood": -2.0,
                "total_logprob": -2.0,
                "avg_logprob": -0.1,
                "token_count": 20,
            },
            " True answer": {
                "log_likelihood": -0.4,
                "total_logprob": -0.4,
                "avg_logprob": -0.2,
                "token_count": 2,
            },
        }

        monkeypatch.setattr(
            "run_intervention.score_continuation_decode_only",
            lambda _model, _tokenizer, _prompt, continuation, scaler=None: (
                scores_by_choice[continuation]
            ),
        )

        run_truthfulqa_mc(
            model=object(),
            tokenizer=object(),
            scaler=SimpleNamespace(alpha=1.0),
            samples=[
                {
                    "id": "truthfulqa_mc1_0",
                    "question": "Which answer is true?",
                    "choices": ["False answer", "True answer"],
                    "labels": [0, 1],
                    "variant": "mc1",
                }
            ],
            alpha=0.0,
            output_dir=str(output_dir),
        )

        records = [
            json.loads(line)
            for line in (output_dir / "alpha_0.0.jsonl").read_text().splitlines()
        ]

        assert records[0]["compliance"] is True
        assert records[0]["metric_name"] == "mc1_accuracy"
        assert records[0]["metric_value"] == 1.0
        assert records[0]["chosen"] == "True answer"
        assert records[0]["choice_log_likelihoods"] == [-2.0, -0.4]

        aggregation = aggregate_results(str(output_dir), [0.0])

        assert aggregation["results"]["0.0"]["metric_mean"] == 1.0
        assert aggregation["results"]["0.0"]["compliance_rate"] == 1.0

    def test_run_truthfulqa_mc_uses_log_likelihood_mass_for_mc2(
        self, tmp_path, monkeypatch
    ):
        output_dir = tmp_path / "truthfulqa_mc2"
        output_dir.mkdir()

        scores_by_choice = {
            " Truthful one": {
                "log_likelihood": -1.0,
                "total_logprob": -1.0,
                "avg_logprob": -0.5,
                "token_count": 2,
            },
            " Truthful two": {
                "log_likelihood": -3.0,
                "total_logprob": -3.0,
                "avg_logprob": -1.0,
                "token_count": 3,
            },
            " False but short": {
                "log_likelihood": -4.0,
                "total_logprob": -4.0,
                "avg_logprob": -0.2,
                "token_count": 20,
            },
        }

        monkeypatch.setattr(
            "run_intervention.score_continuation_decode_only",
            lambda _model, _tokenizer, _prompt, continuation, scaler=None: (
                scores_by_choice[continuation]
            ),
        )

        run_truthfulqa_mc(
            model=object(),
            tokenizer=object(),
            scaler=SimpleNamespace(alpha=1.0),
            samples=[
                {
                    "id": "truthfulqa_mc2_0",
                    "question": "Which answers are true?",
                    "choices": ["Truthful one", "Truthful two", "False but short"],
                    "labels": [1, 1, 0],
                    "variant": "mc2",
                }
            ],
            alpha=0.0,
            output_dir=str(output_dir),
        )

        record = json.loads(
            (output_dir / "alpha_0.0.jsonl").read_text().splitlines()[0]
        )
        expected_mass = float(
            (
                torch.exp(torch.tensor(-1.0, dtype=torch.float64))
                + torch.exp(torch.tensor(-3.0, dtype=torch.float64))
            )
            / (
                torch.exp(torch.tensor(-1.0, dtype=torch.float64))
                + torch.exp(torch.tensor(-3.0, dtype=torch.float64))
                + torch.exp(torch.tensor(-4.0, dtype=torch.float64))
            )
        )

        assert record["metric_name"] == "mc2_truthful_mass"
        assert record["metric_value"] == pytest.approx(expected_mass)
        assert record["choice_log_likelihoods"] == [-1.0, -3.0, -4.0]

    def test_load_truthfulqa_mc_from_csv(self, tmp_path):
        csv_content = (
            "\ufeffType,Category,Question,Best Answer,Correct Answers,Incorrect Answers,Source\n"
            "Adversarial,Misc,Is the sky blue?,Yes the sky is blue,"
            "Yes the sky is blue; The sky appears blue,No the sky is red; The sky is green,"
            "http://example.com\n"
            "Adversarial,Misc,Is water wet?,Water is wet,"
            "Water is wet; Yes,No water is not wet,"
            "http://example.com\n"
        )
        csv_path = tmp_path / "TruthfulQA.csv"
        csv_path.write_text(csv_content, encoding="utf-8")

        mc1 = load_truthfulqa_mc(variant="mc1", csv_path=str(csv_path))
        mc2 = load_truthfulqa_mc(variant="mc2", csv_path=str(csv_path))

        assert len(mc1) == 2
        assert len(mc2) == 2

        # MC1: best answer first with label 1, rest 0
        assert mc1[0]["choices"][0] == "Yes the sky is blue"
        assert mc1[0]["labels"][0] == 1
        assert all(label == 0 for label in mc1[0]["labels"][1:])
        assert mc1[0]["variant"] == "mc1"
        assert mc1[0]["id"] == "truthfulqa_mc1_0"

        # MC2: all correct first (label 1), all incorrect last (label 0)
        assert mc2[0]["choices"] == [
            "Yes the sky is blue",
            "The sky appears blue",
            "No the sky is red",
            "The sky is green",
        ]
        assert mc2[0]["labels"] == [1, 1, 0, 0]
        assert mc2[0]["id"] == "truthfulqa_mc2_0"

        # No BOM or whitespace artifacts
        assert not mc1[0]["question"].startswith("\ufeff")
        assert not any(c.startswith(" ") or c.endswith(" ") for c in mc1[0]["choices"])

    def test_aggregate_results_rejects_mixed_metric_definitions(self, tmp_path):
        output_dir = tmp_path / "mixed_metrics"
        output_dir.mkdir()
        alpha_path = output_dir / "alpha_0.0.jsonl"
        alpha_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": "sample_a",
                            "compliance": True,
                            "metric_name": "mc1_accuracy",
                            "metric_value": 1.0,
                        }
                    ),
                    json.dumps(
                        {
                            "id": "sample_b",
                            "compliance": False,
                            "metric_name": "mc2_truthful_mass",
                            "metric_value": 0.4,
                        }
                    ),
                ]
            )
            + "\n"
        )

        with pytest.raises(ValueError, match="Mixed metric definitions"):
            aggregate_results(str(output_dir), [0.0])

    def test_aggregate_results_exports_triviaqa_bridge_deterministic_metrics(
        self, tmp_path
    ):
        output_dir = tmp_path / "triviaqa_bridge"
        output_dir.mkdir()
        alpha_path = output_dir / "alpha_0.0.jsonl"
        alpha_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": "sample_a",
                            "deterministic_correct": True,
                            "attempted": True,
                            "compliance": True,
                        }
                    ),
                    json.dumps(
                        {
                            "id": "sample_b",
                            "deterministic_correct": False,
                            "attempted": False,
                            "compliance": False,
                        }
                    ),
                ]
            )
            + "\n"
        )

        aggregation = aggregate_results(str(output_dir), [0.0])
        result = aggregation["results"]["0.0"]

        assert result["metric_name"] == "deterministic_accuracy"
        assert result["metric_mean"] == 0.5
        assert result["deterministic_accuracy_rate"] == 0.5
        assert result["n_deterministic_correct"] == 1
        assert result["attempt_rate"] == 0.5
        assert result["n_attempted"] == 1

    def test_aggregate_results_prefers_judge_grade_for_triviaqa_attempts(
        self, tmp_path
    ):
        output_dir = tmp_path / "triviaqa_bridge_judged"
        output_dir.mkdir()
        alpha_path = output_dir / "alpha_0.0.jsonl"
        alpha_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": "sample_a",
                            "deterministic_correct": False,
                            "attempted": True,
                            "triviaqa_bridge_grade": "NOT_ATTEMPTED",
                            "compliance": False,
                        }
                    ),
                    json.dumps(
                        {
                            "id": "sample_b",
                            "deterministic_correct": False,
                            "attempted": False,
                            "triviaqa_bridge_grade": "INCORRECT",
                            "compliance": False,
                        }
                    ),
                ]
            )
            + "\n"
        )

        aggregation = aggregate_results(str(output_dir), [0.0])
        result = aggregation["results"]["0.0"]

        assert result["attempt_rate"] == 0.5
        assert result["n_attempted"] == 1


class TestTriviaQaBridge:
    def test_attempt_detector_rejects_abstentions_and_multi_answers(self):
        assert triviaqa_bridge_attempted("Paris") is True
        assert triviaqa_bridge_attempted("I think Paris") is True
        assert triviaqa_bridge_attempted("I don't know.") is False
        assert triviaqa_bridge_attempted("Need more context.") is False
        assert triviaqa_bridge_attempted("Paris or London") is False

    def test_load_triviaqa_bridge_raises_when_manifest_ids_missing(self, tmp_path):
        pandas = pytest.importorskip("pandas")

        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(["q1", "q_missing"]))

        dataframe = pandas.DataFrame(
            [
                {
                    "question_id": "q1",
                    "question": "Capital of France?",
                    "answer": {"aliases": ["Paris"]},
                }
            ]
        )

        original_read_parquet = pandas.read_parquet
        pandas.read_parquet = lambda _path: dataframe
        try:
            with pytest.raises(ValueError, match="manifest/parquet mismatch"):
                load_triviaqa_bridge(str(manifest_path), parquet_path="dummy.parquet")
        finally:
            pandas.read_parquet = original_read_parquet

    def test_evaluate_triviaqa_bridge_batch_retries_prior_error_records(
        self, tmp_path, monkeypatch
    ):
        output_dir = tmp_path / "triviaqa_bridge_eval"
        output_dir.mkdir()
        alpha_path = output_dir / "alpha_0.0.jsonl"
        alpha_path.write_text(
            json.dumps(
                {
                    "id": "tqa_bridge_q1",
                    "question": "Capital of France?",
                    "response": "Paris",
                    "ground_truth_aliases": ["Paris"],
                    "match_tier": "no_match",
                    "deterministic_correct": False,
                    "judge": "ERROR",
                    "judge_audit_type": "nonmatch",
                    "triviaqa_bridge_grade": "ERROR",
                    "compliance": False,
                }
            )
            + "\n"
        )

        submitted_requests = []
        openai_batch = importlib.import_module("openai_batch")

        monkeypatch.setattr(
            openai_batch,
            "build_chat_request",
            lambda **kwargs: submitted_requests.append(kwargs) or kwargs,
        )
        monkeypatch.setattr(
            openai_batch,
            "resume_or_submit",
            lambda *_args, **_kwargs: {"tqa_0.0_nm_0": {"content": "A"}},
        )
        monkeypatch.setattr(
            openai_batch,
            "parse_chat_content",
            lambda entry: entry["content"],
        )

        evaluate_intervention.evaluate_triviaqa_bridge_batch(
            input_dir=str(output_dir),
            alphas=[0.0],
            client=object(),
            judge_model="gpt-test",
        )

        records = [
            json.loads(line)
            for line in alpha_path.read_text().splitlines()
            if line.strip()
        ]

        assert len(submitted_requests) == 1
        assert records[0]["judge"] == "CORRECT"
        assert records[0]["judge_audit_type"] == "nonmatch"
        assert records[0]["triviaqa_bridge_grade"] == "CORRECT"
        assert records[0]["compliance"] is True


class TestDirectionSanityGate:
    def test_requires_positive_ablation_effect_even_below_threshold(self):
        gate = evaluate_direction_sanity_gate(
            harmful_baseline_refusals=4,
            harmful_ablated_refusals=4,
            n_harmful=10,
        )

        assert gate["passes_refusal_threshold"] is True
        assert gate["has_positive_ablation_effect"] is False
        assert gate["ready_for_d3"] is False

    def test_passes_only_when_ablation_reduces_refusal_and_clears_threshold(self):
        gate = evaluate_direction_sanity_gate(
            harmful_baseline_refusals=8,
            harmful_ablated_refusals=3,
            n_harmful=10,
        )

        assert gate["passes_refusal_threshold"] is True
        assert gate["has_positive_ablation_effect"] is True
        assert gate["refusal_rate_reduction"] == pytest.approx(0.5, abs=1e-6)
        assert gate["refusal_count_reduction"] == 5
        assert gate["ready_for_d3"] is True


# ---------------------------------------------------------------------------
# FaithEval MC log-prob scoring
# ---------------------------------------------------------------------------


class TestFaithEvalMcLogprob:
    def test_picks_highest_logprob_letter_and_flags_compliance(
        self, tmp_path, monkeypatch
    ):
        output_dir = tmp_path / "faitheval_mc"
        output_dir.mkdir()

        # D has highest log-likelihood, which matches counterfactual_key → compliant
        scores_by_continuation = {
            " A": {
                "log_likelihood": -3.0,
                "total_logprob": -3.0,
                "avg_logprob": -3.0,
                "token_count": 1,
            },
            " B": {
                "log_likelihood": -2.0,
                "total_logprob": -2.0,
                "avg_logprob": -2.0,
                "token_count": 1,
            },
            " C": {
                "log_likelihood": -4.0,
                "total_logprob": -4.0,
                "avg_logprob": -4.0,
                "token_count": 1,
            },
            " D": {
                "log_likelihood": -0.5,
                "total_logprob": -0.5,
                "avg_logprob": -0.5,
                "token_count": 1,
            },
        }

        monkeypatch.setattr(
            "run_intervention.score_continuation_decode_only",
            lambda _model, _tokenizer, _prompt, continuation, scaler=None: (
                scores_by_continuation[continuation]
            ),
        )

        run_faitheval_mc_logprob(
            model=object(),
            tokenizer=object(),
            scaler=SimpleNamespace(alpha=1.0),
            samples=[
                {
                    "id": "test_sample_1",
                    "context": "Some misleading context.",
                    "question": "What is the capital of France?",
                    "choices_text": "A) Berlin\nB) Madrid\nC) Rome\nD) London",
                    "valid_letters": ["A", "B", "C", "D"],
                    "counterfactual_key": "D",
                    "num_options": 4,
                }
            ],
            alpha=0.0,
            output_dir=str(output_dir),
        )

        records = [
            json.loads(line)
            for line in (output_dir / "alpha_0.0.jsonl").read_text().splitlines()
        ]
        assert len(records) == 1
        rec = records[0]
        assert rec["chosen"] == "D"
        assert rec["chosen_index"] == 3
        assert rec["compliance"] is True
        assert rec["metric_name"] == "compliance"
        assert rec["metric_value"] == 1.0
        assert rec["choice_log_likelihoods"] == [-3.0, -2.0, -4.0, -0.5]
        assert len(rec["choice_scores"]) == 4
        assert rec["choice_scores"][0]["letter"] == "A"

    def test_non_compliant_when_model_picks_non_counterfactual(
        self, tmp_path, monkeypatch
    ):
        output_dir = tmp_path / "faitheval_mc_nc"
        output_dir.mkdir()

        # A has highest log-likelihood, but counterfactual_key is C → not compliant
        scores_by_continuation = {
            " A": {
                "log_likelihood": -0.1,
                "total_logprob": -0.1,
                "avg_logprob": -0.1,
                "token_count": 1,
            },
            " B": {
                "log_likelihood": -2.0,
                "total_logprob": -2.0,
                "avg_logprob": -2.0,
                "token_count": 1,
            },
            " C": {
                "log_likelihood": -3.0,
                "total_logprob": -3.0,
                "avg_logprob": -3.0,
                "token_count": 1,
            },
            " D": {
                "log_likelihood": -4.0,
                "total_logprob": -4.0,
                "avg_logprob": -4.0,
                "token_count": 1,
            },
        }

        monkeypatch.setattr(
            "run_intervention.score_continuation_decode_only",
            lambda _model, _tokenizer, _prompt, continuation, scaler=None: (
                scores_by_continuation[continuation]
            ),
        )

        run_faitheval_mc_logprob(
            model=object(),
            tokenizer=object(),
            scaler=SimpleNamespace(alpha=1.0),
            samples=[
                {
                    "id": "test_sample_2",
                    "context": "Some context.",
                    "question": "What color is the sky?",
                    "choices_text": "A) Blue\nB) Red\nC) Green\nD) Yellow",
                    "valid_letters": ["A", "B", "C", "D"],
                    "counterfactual_key": "C",
                    "num_options": 4,
                }
            ],
            alpha=0.0,
            output_dir=str(output_dir),
        )

        rec = json.loads((output_dir / "alpha_0.0.jsonl").read_text().splitlines()[0])
        assert rec["chosen"] == "A"
        assert rec["compliance"] is False
        assert rec["metric_value"] == 0.0

    def test_no_parse_failures_by_construction(self, tmp_path, monkeypatch):
        """Log-prob scoring always produces a valid chosen letter."""
        output_dir = tmp_path / "faitheval_mc_nopf"
        output_dir.mkdir()

        scores_by_continuation = {
            " A": {
                "log_likelihood": -1.0,
                "total_logprob": -1.0,
                "avg_logprob": -1.0,
                "token_count": 1,
            },
            " B": {
                "log_likelihood": -1.0,
                "total_logprob": -1.0,
                "avg_logprob": -1.0,
                "token_count": 1,
            },
        }

        monkeypatch.setattr(
            "run_intervention.score_continuation_decode_only",
            lambda _model, _tokenizer, _prompt, continuation, scaler=None: (
                scores_by_continuation[continuation]
            ),
        )

        run_faitheval_mc_logprob(
            model=object(),
            tokenizer=object(),
            scaler=SimpleNamespace(alpha=1.0),
            samples=[
                {
                    "id": "test_nopf",
                    "context": "Context.",
                    "question": "Q?",
                    "choices_text": "A) X\nB) Y",
                    "valid_letters": ["A", "B"],
                    "counterfactual_key": "A",
                    "num_options": 2,
                }
            ],
            alpha=0.0,
            output_dir=str(output_dir),
        )

        rec = json.loads((output_dir / "alpha_0.0.jsonl").read_text().splitlines()[0])
        # chosen is always a valid letter, never None
        assert rec["chosen"] in ["A", "B"]
        assert rec["chosen"] is not None
